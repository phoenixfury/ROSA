#!/usr/bin/env python3
"""Profile/convert/measure an ONNX candidate without replacing deployment defaults."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

ROSA=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROSA))
deps=ROSA/'work_dirs/onnx_deps_py38'
if sys.version_info[:2]==(3,8) and deps.is_dir():
    sys.path.insert(0,str(deps))
import numpy as np
from deployment.runtime import create_session,load_npz,compare_outputs
from benchmark_fisheye_backends import sha,stats,DeviceMemory


def summarize_profile(events):
    runs=[event for event in events if event.get('name')=='model_run']
    if runs:
        last=max(runs,key=lambda event:event['ts'])
        start,stop=last['ts'],last['ts']+last['dur']
        events=[event for event in events if start<=event.get('ts',-1)<stop]
    ops,providers,scopes=Counter(),Counter(),Counter()
    rows=[]
    for event in events:
        if event.get('cat')!='Node' or not event.get('name','').endswith('_kernel_time'):
            continue
        data=event.get('args',{})
        duration=event.get('dur',0)
        key=data.get('op_name','unknown')
        ops[key]+=duration
        providers[data.get('provider','unknown')]+=duration
        scope='backbone' if 'img_backbone' in event['name'] else 'neck' if 'img_neck' in event['name'] else 'head/other'
        scopes[scope]+=duration
        rows.append(dict(name=event['name'],op=key,microseconds=duration,provider=data.get('provider')))
    return dict(operator_microseconds=dict(ops.most_common()),provider_microseconds=dict(providers),
        scope_microseconds=dict(scopes),slowest_nodes=sorted(rows,key=lambda r:-r['microseconds'])[:30],
        gpu_kernel_events=sum(event.get('cat')=='Kernel' for event in events),
        note='ORT host operator durations, including scheduling/synchronization; not pure GPU kernel time. '
             'Scope sums can overlap for control-flow nodes. Use synchronized encoder/head assays for '
             'component wall-time attribution and non-profiled distinct-frame runs for latency.')


def graph_nodes(graph):
    import onnx
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.type==onnx.AttributeProto.GRAPH:
                yield from graph_nodes(attribute.g)
            elif attribute.type==onnx.AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    yield from graph_nodes(nested)


def convert_backbone_only(model, fp32_scopes=()):
    """Convert an extracted image encoder, never the decoder's nested If graphs."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16,DEFAULT_OP_BLOCK_LIST
    from onnxruntime.transformers.onnx_model import OnnxModel
    encoder_all={n.name for n in model.graph.node if any(scope in n.name for scope in ('img_backbone','img_neck'))}
    encoder={name for name in encoder_all if not any(scope in name for scope in fp32_scopes)}
    consumers={}
    for node in model.graph.node:
        for value in node.input:
            consumers.setdefault(value,[]).append(node.name)
    # Extract the WHOLE encoder, even if a protected middle stage also feeds
    # a lateral FPN branch. Otherwise that branch can receive a half tensor
    # without the required FP32 bridge outside the extracted partition.
    boundaries=[value for node in model.graph.node if node.name in encoder_all for value in node.output
                if any(name not in encoder_all for name in consumers.get(value,[]))]
    if not boundaries:
        raise ValueError('Could not identify image-encoder/decoder boundary')
    from deployment.onnx_static import indexed_extractor
    part=indexed_extractor(model).extract_model(['images'],boundaries)
    produced={value for node in part.graph.node for value in node.output}
    replaced_initializers={value.name for value in part.graph.initializer}
    remaining=[node for node in model.graph.node if not any(value in produced for value in node.output)]
    shared=replaced_initializers&{value for node in remaining for value in node.input}
    # Simplifiers deduplicate scalar constants across encoder and geometry.
    # Clone shared initializers before FP16 conversion; never round the head's copy.
    renames={name:'encoder_only/'+name for name in shared}
    for initializer in part.graph.initializer:
        if initializer.name in renames:
            initializer.name=renames[initializer.name]
    for node in part.graph.node:
        for index,name in enumerate(node.input):
            node.input[index]=renames.get(name,name)
    replaced_initializers-=shared
    # Preprocessing remains FP32, with explicit FP32 bridges into the unchanged head.
    blocked=[node.name for node in part.graph.node if node.name not in encoder]
    half=convert_float_to_float16(part,keep_io_types=True,node_block_list=blocked,
        op_block_list=list(DEFAULT_OP_BLOCK_LIST))
    OnnxModel(half).topological_sort()
    onnx.checker.check_model(half,full_check=True)
    initializers=[value for value in model.graph.initializer if value.name not in replaced_initializers]
    model.graph.ClearField('node');model.graph.node.extend(half.graph.node);model.graph.node.extend(remaining)
    model.graph.ClearField('initializer');model.graph.initializer.extend(initializers);model.graph.initializer.extend(half.graph.initializer)
    model.graph.ClearField('value_info');model.graph.value_info.extend(half.graph.value_info)
    OnnxModel(model).topological_sort()
    onnx.checker.check_model(model,full_check=True)
    return model,len(blocked)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('profile','convert','simplify','specialize-shapes','trt-prepare','gpu-atan','gpu-argmax','split-encoder','benchmark'))
    p.add_argument('--model',type=Path,default=ROSA/'work_dirs/onnx_singleframe_e1/rosa_e1.onnx')
    p.add_argument('--input',type=Path,default=ROSA/'work_dirs/onnx_singleframe_e1/example_0_input.npz')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-model',type=Path)
    p.add_argument('--reference',type=Path,help='Saved same-input FP32 reference, avoids loading a second session')
    p.add_argument('--precision',choices=('backbone-fp16','mixed-fp16'),default='backbone-fp16')
    p.add_argument('--fp32-encoder-scope',action='append',default=[],help='Keep matching encoder scopes in FP32, e.g. img_neck')
    p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--cuda-tuning',choices=('default','throughput'),default='default')
    p.add_argument('--iobinding',action='store_true')
    p.add_argument('--cuda-graph',action='store_true',help='Capture/replay a fully CUDA graph; requires --iobinding')
    p.add_argument('--engine',type=Path,help='Benchmark a TensorRT engine built from --model')
    p.add_argument('--split-encoder',type=Path,help='Exact encoder/head ONNX partition directory')
    p.add_argument('--encoder-engine',type=Path,help='Optional TensorRT encoder; the head stays in ORT')
    args=p.parse_args()
    if args.repeats<1 or args.threads<1:
        p.error('repeats and threads must be positive')
    if args.reference and not args.reference.is_file():
        p.error('Reference file does not exist: '+str(args.reference))
    if args.cuda_graph and ((not args.iobinding and not args.split_encoder) or args.engine):p.error('--cuda-graph requires ORT --iobinding or --split-encoder')
    if args.encoder_engine and not args.split_encoder:p.error('--encoder-engine requires --split-encoder')
    if args.split_encoder and (args.iobinding or args.engine):p.error('Split encoder handles its own GPU I/O')
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(args.model.with_suffix('.json').read_text())
    if metadata['onnx_sha256']!=sha(args.model):
        raise ValueError('Source model checksum differs')
    if args.stage=='split-encoder':
        import onnx
        from deployment.split_runtime import split_encoder_model
        encoder,head,boundaries=split_encoder_model(onnx.load(str(args.model)))
        for name,model in [('encoder',encoder),('head',head)]:
            destination=args.output/(name+'.onnx')
            if destination.exists() or destination.with_suffix('.json').exists():raise FileExistsError(destination)
            onnx.save(model,str(destination))
            manifest=copy.deepcopy(metadata)
            manifest.update(onnx_sha256=sha(destination),source_onnx=str(args.model.resolve()),source_onnx_sha256=sha(args.model),
                split_role=name,feature_boundaries=boundaries,
                inputs=[dict(name=v.name,shape=[d.dim_value for d in v.type.tensor_type.shape.dim]) for v in model.graph.input],
                outputs=[dict(name=v.name,shape=[d.dim_value for d in v.type.tensor_type.shape.dim]) for v in model.graph.output],
                deployment_validation_status='UNVALIDATED_DO_NOT_PROMOTE')
            destination.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('Split',len(boundaries),'GPU feature tensors:',args.output,flush=True);return
    if args.stage=='specialize-shapes':
        import onnx
        from onnx import helper,TensorProto
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
        sys.path.insert(0,str(ROSA/'work_dirs/onnx_opt_deps_py38'))
        from onnxsim import simplify
        model=onnx.load(str(args.model));before=len(model.graph.node)
        if any(not v.type.tensor_type.HasField('shape') or any(d.dim_value<=0 for d in v.type.tensor_type.shape.dim)
               for v in model.graph.input):raise ValueError('Shape specialization requires a fixed input contract')
        model=SymbolicShapeInference.infer_shapes(model,auto_merge=True,guess_output_rank=False,verbose=0)
        shapes={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim]
                for v in [*model.graph.input,*model.graph.value_info,*model.graph.output]
                if v.type.HasField('tensor_type') and v.type.tensor_type.HasField('shape') and
                all(d.dim_value>0 for d in v.type.tensor_type.shape.dim)}
        proofs=[];nodes=[]
        for node in model.graph.node:
            if node.op_type=='Shape' and node.input[0] in shapes:
                shape=shapes[node.input[0]];attributes={a.name:helper.get_attribute_value(a) for a in node.attribute}
                value=shape[attributes.get('start',0):attributes.get('end',len(shape))]
                tensor=helper.make_tensor(node.output[0]+'_proven',TensorProto.INT64,[len(value)],value)
                nodes.append(helper.make_node('Constant',[],list(node.output),name=node.name,value=tensor))
                proofs.append(dict(input=node.input[0],shape=shape,start=attributes.get('start'),end=attributes.get('end')))
            else:nodes.append(node)
        model.graph.ClearField('node');model.graph.node.extend(nodes)
        model,checked=simplify(model,check_n=0)
        if not checked:raise RuntimeError('Specialized graph structural check failed')
        onnx.checker.check_model(model,full_check=True)
        destination=args.output/'static_shapes.onnx'
        if destination.exists() or destination.with_suffix('.json').exists():raise FileExistsError(destination)
        onnx.save(model,str(destination));manifest=copy.deepcopy(metadata)
        manifest.update(onnx_sha256=sha(destination),source_onnx=str(args.model.resolve()),source_onnx_sha256=sha(args.model),
            shape_specialization=dict(method='Symbolically inferred fixed shapes, no runtime-value assumptions',
                nodes_before=before,nodes_after=len(model.graph.node),folded_shape_proofs=proofs),
            deployment_validation_status='UNVALIDATED_DO_NOT_PROMOTE')
        destination.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('Specialized',len(proofs),'Shape nodes;',before,'->',len(model.graph.node),destination,flush=True);return
    if args.stage=='gpu-atan':
        import onnx
        from deployment.onnx_math import replace_fp32_atan
        converted,replaced=replace_fp32_atan(onnx.load(str(args.model)))
        destination=args.output/'gpu_atan.onnx'
        if destination.exists() or destination.with_suffix('.json').exists():raise FileExistsError(destination)
        onnx.save(converted,str(destination))
        manifest=copy.deepcopy(metadata)
        manifest.update(onnx_sha256=sha(destination),source_onnx=str(args.model.resolve()),source_onnx_sha256=sha(args.model),
            math_approximation=dict(method='Range-reduced odd degree17 FP32 atan polynomial',replaced_nodes=replaced,
                validation='Dense scalar numerical tests plus paired real-frame/official metrics required; not exact arithmetic'),
            references=[],deployment_validation_status='UNVALIDATED_DO_NOT_PROMOTE')
        destination.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('Replaced',len(replaced),'CPU Atan candidates:',destination,flush=True);return
    if args.stage=='gpu-argmax':
        import onnx
        from deployment.onnx_math import replace_argmax_topk
        converted,replaced=replace_argmax_topk(onnx.load(str(args.model)))
        destination=args.output/'gpu_argmax.onnx'
        if destination.exists() or destination.with_suffix('.json').exists():raise FileExistsError(destination)
        onnx.save(converted,str(destination));manifest=copy.deepcopy(metadata)
        manifest.update(onnx_sha256=sha(destination),source_onnx=str(args.model.resolve()),source_onnx_sha256=sha(args.model),
            argmax_optimization=dict(method='TopK(1), lowest-index tie breaking for finite scores',replaced_nodes=replaced),
            deployment_validation_status='UNVALIDATED_DO_NOT_PROMOTE')
        destination.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('Replaced',len(replaced),'ArgMax nodes:',destination,flush=True);return
    if args.stage=='trt-prepare':
        import onnx
        from onnx import helper,TensorProto
        model=onnx.load(str(args.model))
        if next(value for value in model.graph.input if value.name=='images').type.tensor_type.elem_type!=TensorProto.UINT8:
            raise ValueError('Expected canonical uint8 input')
        # Exact commuting rewrite: uint8 channel permutation then float cast
        # becomes float cast then channel permutation. All uint8 values are
        # represented exactly in FP32; TRT cannot Gather a UINT8 tensor.
        for node in model.graph.node:
            for index,value in enumerate(node.input):
                if value=='images':
                    node.input[index]='images_fp32_for_trt'
        model.graph.node.insert(0,helper.make_node('Cast',['images'],['images_fp32_for_trt'],
            name='canonical_uint8_to_fp32',to=TensorProto.FLOAT))
        model.graph.ClearField('value_info')
        model=onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model,full_check=True)
        destination=args.output/'trt_prepared.onnx'
        if destination.exists():
            raise FileExistsError(destination)
        onnx.save(model,str(destination))
        meta=copy.deepcopy(metadata)
        meta.update(onnx_sha256=sha(destination),source_onnx_sha256=sha(args.model),source_onnx=str(args.model.resolve()),
            optimization=dict(method='Exact cast-before-channel-Gather rewrite; uint8 I/O contract preserved'),
            deployment_validation_status='UNVALIDATED_DO_NOT_PROMOTE')
        destination.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(destination,flush=True)
        return
    if args.stage=='simplify':
        import onnx
        sys.path.insert(0,str(ROSA/'work_dirs/onnx_opt_deps_py38'))
        from onnxsim import simplify
        model=onnx.load(str(args.model))
        before=len(model.graph.node)
        shapes={item.name:[dim.dim_value for dim in item.type.tensor_type.shape.dim] for item in model.graph.input}
        converted,checked=simplify(model,check_n=0,overwrite_input_shapes=shapes)
        if not checked:
            raise RuntimeError('Simplifier structural check failed')
        onnx.checker.check_model(converted,full_check=True)
        destination=args.output/'simplified.onnx'
        if destination.exists():
            raise FileExistsError(destination)
        onnx.save(converted,str(destination))
        manifest=copy.deepcopy(metadata)
        manifest.update(onnx_sha256=sha(destination),source_onnx_sha256=sha(args.model),source_onnx=str(args.model.resolve()),
            optimization=dict(method='onnxsim static shape/control flow folding',nodes_before=before,nodes_after=len(converted.graph.node)),
            deployment_validation_status='UNVALIDATED_DO_NOT_PROMOTE')
        destination.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('Simplified',before,'->',len(converted.graph.node),destination,flush=True)
        return
    if args.stage=='convert':
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16,DEFAULT_OP_BLOCK_LIST
        model=onnx.load(str(args.model))
        if args.precision=='backbone-fp16':
            converted,blocked_count=convert_backbone_only(model,args.fp32_encoder_scope)
        else:
            # Numerically sensitive projection/angle operations stay FP32.
            blocked=[n.name for n in graph_nodes(model.graph) if any(scope in n.name.lower() for scope in
                ('projection','ocam','polynomial','refinement','keypoint','decode'))]
            converted=convert_float_to_float16(model,keep_io_types=True,node_block_list=blocked,
                op_block_list=list(set(DEFAULT_OP_BLOCK_LIST)|{'GridSample','LayerNormalization','Softmax','Sin','Cos','Atan','Acos','Asin','Exp','Log','Pow'}))
            blocked_count=len(blocked)
        destination=args.output/(args.precision+'.onnx')
        if destination.exists():
            raise FileExistsError(destination)
        from onnxruntime.transformers.onnx_model import OnnxModel
        OnnxModel(converted).topological_sort()
        onnx.checker.check_model(converted,full_check=True)
        onnx.save(converted,str(destination))
        manifest=copy.deepcopy(metadata)
        manifest.update(onnx_sha256=sha(destination),precision=args.precision,
            source_onnx=str(args.model.resolve()),source_onnx_sha256=sha(args.model),
            optimization=dict(method='Isolated FP16 image encoder with FP32 decoder bridges',blocked_nodes=blocked_count,
                fp32_encoder_scopes=args.fp32_encoder_scope,keep_io_types=True),
            references=[],deployment_validation_status='UNVALIDATED_DO_NOT_PROMOTE')
        destination.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('Converted candidate',destination,flush=True)
        return
    feed=load_npz(args.input)
    if args.stage=='profile':
        session=create_session(args.model,'cuda',threads=args.threads,profile=True,
            profile_prefix=str(args.output/'ort_profile'),cuda_tuning=args.cuda_tuning)
        for _ in range(2):
            session.run(None,feed)
        profile=Path(session.end_profiling())
        report=summarize_profile(json.loads(profile.read_text()))
        report.update(model=str(args.model.resolve()),model_sha256=sha(args.model),profile=str(profile))
        (args.output/'profile_summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        return
    with DeviceMemory() as memory:
        engine_report=None
        if args.split_encoder:
            from deployment.split_runtime import SplitEncoderRunner
            runner=SplitEncoderRunner(args.model,args.split_encoder,args.encoder_engine,args.threads,
                args.cuda_graph,args.cuda_tuning)
            run=lambda:runner.run(feed)
            session=None
        elif args.engine:
            from deployment.tensorrt_runtime import TensorRTRunner
            engine_report=json.loads((args.engine.parent/'build_report.json').read_text())
            if engine_report['onnx_sha256']!=sha(args.model) or engine_report['engine_sha256']!=sha(args.engine):
                raise ValueError('TensorRT engine/source checksum differs')
            runner=TensorRTRunner(args.engine)
            run=lambda:runner.run(feed)
            session=None
        else:
            session=create_session(args.model,'cuda',threads=args.threads,cuda_tuning=args.cuda_tuning,cuda_graph=args.cuda_graph)
        if args.iobinding and not args.engine:
            from deployment.runtime import CudaIOBindingRunner
            runner=CudaIOBindingRunner(session,cuda_graph=args.cuda_graph)
            run=lambda:runner.run(feed)
        elif not args.engine and not args.split_encoder:
            run=lambda:session.run(None,feed)
        for _ in range(3):
            raw=run()
        timings=[]
        for _ in range(args.repeats):
            started=time.perf_counter();raw=run();timings.append(time.perf_counter()-started)
    report=dict(model=str(args.model.resolve()),onnx_sha256=sha(args.model),precision=metadata['precision'],
        iobinding=args.iobinding,cuda_graph=args.cuda_graph,cuda_tuning=args.cuda_tuning,threads=args.threads,
        repeats=args.repeats,latency=stats(timings),
        peak_gpu_device_mib=memory.peak/1024**2,baseline_gpu_device_mib=memory.baseline/1024**2)
    if args.split_encoder:
        report.update(split_encoder=str(args.split_encoder.resolve()),
            encoder_precision=runner.encoder_precision,
            encoder_engine=str(args.encoder_engine.resolve()) if args.encoder_engine else None,
            encoder_engine_sha256=sha(args.encoder_engine) if args.encoder_engine else None)
    if engine_report:
        report.update(engine=str(args.engine.resolve()),engine_precision=engine_report['precision'])
    np.savez(args.output/'raw_outputs.npz',**dict(zip(('boxes','scores','labels'),raw)))
    del session
    if args.iobinding or args.engine or args.split_encoder:
        del runner
    del run
    import gc
    gc.collect()
    if args.reference:
        expected=load_npz(args.reference)
        report['raw_fp32_parity']=compare_outputs(raw,expected,box_atol=.05,score_atol=.005,yaw_atol=.02)
        from deployment.runtime import CLASSES
        profile=json.loads((ROSA/'work_dirs/threshold_tuning_e1_20261103/balanced_profile.json').read_text())['score_thresholds']
        limits=np.asarray([profile[name] for name in CLASSES])
        mask=(np.asarray(raw[1])>=limits[np.asarray(raw[2])])|(expected['scores']>=limits[expected['labels']])
        report['retained_query_count']=int(mask.sum())
        if mask.any():
            retained=[np.asarray(value)[mask] for value in raw]
            ref={key:value[mask] for key,value in expected.items()}
            report['retained_fp32_parity']=compare_outputs(retained,ref,box_atol=.05,score_atol=.005,yaw_atol=.02)
    if args.reference_model:
        reference_session=create_session(args.reference_model,'cuda')
        reference=reference_session.run(None,feed)
        # Candidate FP16 tolerance is declared explicitly, never silently accepted as FP32 parity.
        reference=dict(zip(('boxes','scores','labels'),reference))
        report['raw_fp32_parity']=compare_outputs(raw,reference,box_atol=.05,score_atol=.005,yaw_atol=.02)
    path=args.output/('benchmark_iobinding.json' if args.iobinding else 'benchmark.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()

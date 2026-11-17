#!/usr/bin/env python3
"""Build an isolated TensorRT engine, or parse only to report unsupported nodes."""
import argparse
import json
from pathlib import Path
import sys
import time

ROSA=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROSA),str(ROSA/'work_dirs/tensorrt_deps_py38')]
from benchmark_fisheye_backends import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parse-only',action='store_true')
    p.add_argument('--fold-static-squeezes',action='store_true',help='Prove fixed-shape squeeze branches using TensorRT shape inference')
    p.add_argument('--workspace-mib',type=int,default=768)
    p.add_argument('--precision',choices=('backbone-fp16','fp16','fp32','tf32'),default='backbone-fp16')
    p.add_argument('--fp32-scope',action='append',default=[],help='Additional named layer scopes kept FP32 in FP16 builds')
    args=p.parse_args()
    import torch
    import tensorrt as trt
    metadata=json.loads(args.model.with_suffix('.json').read_text())
    if metadata['onnx_sha256']!=sha(args.model):
        raise ValueError('ONNX provenance mismatch')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    destination=args.output/'rosa.engine'
    if destination.exists() and not args.parse_only:
        raise FileExistsError(destination)
    logger=trt.Logger(trt.Logger.WARNING)
    builder=trt.Builder(logger)
    import onnx
    from deployment.onnx_static import fold_static_squeezes
    model=onnx.load(str(args.model))
    proofs=[]
    for attempt in range(25):
        network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser=trt.OnnxParser(network,logger)
        parsed=parser.parse(model.SerializeToString())
        if parsed or not args.fold_static_squeezes:
            break
        needed=set()
        for node in model.graph.node:
            if node.op_type=='If':
                for attr in node.attribute:
                    if attr.type==onnx.AttributeProto.GRAPH:
                        for child in attr.g.node:
                            if child.op_type=='Squeeze':
                                needed.add(child.input[0])
        shapes={}
        for i in range(network.num_layers):
            layer=network.get_layer(i)
            if layer.type==trt.LayerType.CONDITIONAL_OUTPUT:
                continue
            for j in range(layer.num_outputs):
                tensor=layer.get_output(j)
                if tensor.name in needed:
                    shape=tuple(tensor.shape)
                    if all(dim>0 for dim in shape):
                        shapes[tensor.name]=shape
        folded=fold_static_squeezes(model,shapes)
        if not folded:
            break
        onnx.checker.check_model(model)
        proofs.extend(folded)
        print('Proved and folded',len(folded),'static squeeze branch(es)',flush=True)
        del parser,network
    errors=[str(parser.get_error(i)) for i in range(parser.num_errors)]
    report=dict(model=str(args.model.resolve()),onnx_sha256=sha(args.model),checkpoint_sha256=metadata['checkpoint_sha256'],
        tensorrt_version=trt.__version__,parsed=bool(parsed),errors=errors,layers=network.num_layers,
        gpu=torch.cuda.get_device_name(0),precision=args.precision,workspace_mib=args.workspace_mib,
        fp32_scopes=args.fp32_scope,
        static_shape_proofs=proofs)
    (args.output/'build_report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not parsed:
        raise RuntimeError('TensorRT parse rejected: '+str(errors[:5]))
    if proofs:
        model.graph.ClearField('value_info')
        model=onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model,full_check=True)
        static_path=args.output/'trt_static.onnx'
        onnx.save(model,str(static_path))
        static_meta=dict(metadata,onnx_sha256=sha(static_path),source_onnx_sha256=sha(args.model),
            optimization=dict(method='Compiler-proven fixed-contract squeeze folding',static_shape_proofs=proofs),
            deployment_validation_status='UNVALIDATED_DO_NOT_PROMOTE')
        static_path.with_suffix('.json').write_text(json.dumps(static_meta,indent=2)+'\n')
        report['static_onnx']=str(static_path)
    if args.parse_only:
        print(json.dumps(report,indent=2),flush=True);return
    config=builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,args.workspace_mib*1024**2)
    config.clear_flag(trt.BuilderFlag.TF32)
    if args.precision=='tf32':config.set_flag(trt.BuilderFlag.TF32)
    config.max_aux_streams=0
    if args.precision not in ('fp32','tf32'):
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        protected=0
        for i in range(network.num_layers):
            layer=network.get_layer(i)
            name=layer.name.lower()
            keep_float=(not any(scope in name for scope in ('img_backbone','img_neck')) if args.precision=='backbone-fp16'
                        else any(scope in name for scope in ('projector','projection','ocam','refinement','atan','sin','cos')))
            keep_float=keep_float or any(scope.lower() in name for scope in args.fp32_scope)
            if keep_float and any(layer.get_output(j).dtype==trt.float32 for j in range(layer.num_outputs)):
                layer.precision=trt.float32
                for j in range(layer.num_outputs):
                    if layer.get_output(j).dtype==trt.float32:
                        layer.set_output_type(j,trt.float32)
                protected+=1
        report['fp32_protected_layers']=protected
    started=time.monotonic()
    serialized=builder.build_serialized_network(network,config)
    if serialized is None:
        raise RuntimeError('TensorRT build failed; inspect the build log')
    destination.write_bytes(bytes(serialized))
    report.update(engine=str(destination),engine_sha256=sha(destination),build_seconds=time.monotonic()-started,
        deployment_validation_status='UNVALIDATED_DO_NOT_PROMOTE',
        note='Engine is GPU/TensorRT-version-specific. Benchmark and real-frame parity required before deployment.')
    (args.output/'build_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()

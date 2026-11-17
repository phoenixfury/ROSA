#!/usr/bin/env python3
"""Check an ORT/TensorRT deployment on real camera frames against paired FP32.

Accuracy gate is explicit for the checkpoint-bound operating thresholds. Raw
low-score query differences are reported separately, never silently hidden.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time

ROSA=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROSA),str(ROSA/'work_dirs/onnx_deps_py38')]
import numpy as np
from scipy.optimize import linear_sum_assignment
from deployment.runtime import CLASSES,create_session,load_npz,image_input,postprocess,compare_outputs,CudaIOBindingRunner
from deployment.tensorrt_runtime import TensorRTRunner
from benchmark_fisheye_backends import sha


def feed_for(sample):
    if 'input' in sample:
        return load_npz(sample['input'])
    feed=load_npz(sample['calibration'])
    feed['images']=image_input(sample['image_paths'])
    return feed


def matching_cache_indices(cases,cached):
    identities=[json.dumps(sample,sort_keys=True) for sample in cached]
    if len(set(identities))!=len(identities):raise ValueError('Duplicate cached case identities')
    lookup={identity:index for index,identity in enumerate(identities)}
    try:return [lookup[json.dumps(sample,sort_keys=True)] for sample in cases]
    except KeyError:raise ValueError('Requested case is not present with identical metadata in the raw cache')


def reuse_fp32_references(source,cases,reference_sha256,destination):
    """Reuse only identical cases and checksum-verified original FP32 arrays."""
    report=json.loads((source/'validation.json').read_text())
    if report['reference_onnx_sha256']!=reference_sha256:
        raise ValueError('Cached FP32 references belong to another model')
    cached=json.loads((source/'cases.json').read_text())['samples']
    indices=matching_cache_indices(cases,cached)
    checks={row['index']:row['fp32_sha256'] for row in report['raw_output_checksums']}
    for index,old in enumerate(indices):
        path=source/'fp32_references'/('%03d.npz'%old)
        if sha(path)!=checks.get(old):raise ValueError('Cached FP32 array checksum differs')
        target=destination/('%03d.npz'%index)
        if target.exists():raise FileExistsError(target)
        shutil.copyfile(path,target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--engine',type=Path)
    p.add_argument('--split-encoder',type=Path)
    p.add_argument('--encoder-engine',type=Path)
    p.add_argument('--iobinding',action='store_true',help='Persistent CUDA buffers for ONNX Runtime')
    p.add_argument('--cuda-graph',action='store_true')
    p.add_argument('--custom-op-library',type=Path)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--cuda-tuning',choices=('default','throughput'),default='default')
    p.add_argument('--reference-model',type=Path,required=True)
    p.add_argument('--threshold-profile',type=Path,required=True)
    p.add_argument('--manifest',action='append',required=True,metavar='NAME=JSON')
    p.add_argument('--per-manifest',type=int,default=4)
    p.add_argument('--max-heading-difference-deg',type=float,default=2.,
        help='Explicit retained-query maximum versus FP32; default2, user-approved mixed precision may use5')
    p.add_argument('--analyze-cache',type=Path,
        help='Recompute accuracy from saved FP32/candidate NPZ files and identical cases, without GPU inference; not a timing run')
    p.add_argument('--reference-cache',type=Path,
        help='Reuse verified original FP32 arrays for identical cases; run only the new candidate on GPU')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.custom_op_library and args.engine:p.error('ORT custom-op library cannot be used as a TensorRT plugin')
    if args.analyze_cache and args.reference_cache:p.error('Reference reuse and cache-only analysis are mutually exclusive')
    if not 0<args.max_heading_difference_deg<=180:p.error('Heading difference limit must be in (0,180]')
    yaw_atol=np.deg2rad(args.max_heading_difference_deg)
    if args.cuda_graph and ((not args.iobinding and not args.split_encoder) or args.engine):p.error('--cuda-graph requires ORT --iobinding or --split-encoder')
    if args.encoder_engine and not args.split_encoder:p.error('--encoder-engine requires --split-encoder')
    if args.split_encoder and (args.engine or args.iobinding):p.error('Split encoder handles its own GPU I/O')
    reference_meta=json.loads(args.reference_model.with_suffix('.json').read_text())
    model_meta=json.loads(args.model.with_suffix('.json').read_text())
    engine_meta=json.loads((args.engine.parent/'build_report.json').read_text()) if args.engine else model_meta
    profile=json.loads(args.threshold_profile.read_text())
    if len({reference_meta['checkpoint_sha256'],model_meta['checkpoint_sha256'],engine_meta['checkpoint_sha256'],profile['checkpoint_sha256']})!=1:
        raise ValueError('References/profile/engine must use the same checkpoint')
    artifacts=[(args.model,model_meta['onnx_sha256']),(args.reference_model,reference_meta['onnx_sha256'])]
    if args.engine:artifacts.append((args.engine,engine_meta['engine_sha256']))
    for path,expected in artifacts:
        if sha(path)!=expected:
            raise ValueError('Artifact checksum differs')
    if engine_meta['onnx_sha256']!=model_meta['onnx_sha256']:
        raise ValueError('Engine came from a different ONNX model')
    cases=[]
    for value in args.manifest:
        name,path=value.split('=',1);path=Path(path).resolve()
        document=json.loads(path.read_text());root=Path(document.get('data_root','.'))
        samples=document['samples']
        for index in np.unique(np.linspace(0,len(samples)-1,min(args.per_manifest,len(samples))).astype(int)):
            sample=dict(samples[index]);sample.update(bag=name,source_manifest=str(path),source_index=int(index))
            if 'image_relative_paths' in sample:
                sample['image_paths']=[str((root/p).resolve()) for p in sample['image_relative_paths']]
            if 'calibration' in sample:
                calibration=Path(sample['calibration'])
                sample['calibration']=str(calibration if calibration.is_absolute() else (path.parent/calibration).resolve())
            cases.append(sample)
    cache_indices=list(range(len(cases)))
    if args.analyze_cache:
        cached=json.loads((args.analyze_cache/'cases.json').read_text())['samples']
        cache_indices=matching_cache_indices(cases,cached)
    (out/'cases.json').write_text(json.dumps(dict(samples=cases),indent=2)+'\n')
    cache=args.analyze_cache.resolve() if args.analyze_cache else out
    refs=cache/'fp32_references';refs.mkdir(exist_ok=True)
    if not args.analyze_cache:
        if args.reference_cache:
            reuse_fp32_references(args.reference_cache,cases,sha(args.reference_model),refs)
        else:
            session=create_session(args.reference_model,'cuda')
            for index,sample in enumerate(cases):
                raw=session.run(None,feed_for(sample))
                np.savez(refs/('%03d.npz'%index),**dict(zip(('boxes','scores','labels'),raw)))
            del session;gc.collect()
        import torch
        torch.cuda.empty_cache()
        if args.split_encoder:
            from deployment.split_runtime import SplitEncoderRunner
            runner=SplitEncoderRunner(args.model,args.split_encoder,args.encoder_engine,args.threads,
                args.cuda_graph,args.cuda_tuning,args.custom_op_library);run=runner.run
        elif args.engine:
            runner=TensorRTRunner(args.engine);run=runner.run
        else:
            session=create_session(args.model,'cuda',threads=args.threads,cuda_tuning=args.cuda_tuning,cuda_graph=args.cuda_graph,
                custom_op_library=args.custom_op_library)
            if args.iobinding:
                runner=CudaIOBindingRunner(session,cuda_graph=args.cuda_graph);run=runner.run
            else:run=lambda feed:session.run(None,feed)
    limits=np.asarray([profile['score_thresholds'][name] for name in CLASSES])
    rows=[];actual_retained=[[],[],[]];reference_retained=[[],[],[]]
    final_yaw_errors=[];final_box_errors=[];final_score_errors=[]
    candidates=cache/'candidate_raw_outputs';candidates.mkdir(exist_ok=True)
    cache_checksums=[]
    total_before,total_after,total_matched=0,0,0
    for index,sample in enumerate(cases):
        cache_index=cache_indices[index]
        expected=load_npz(refs/('%03d.npz'%cache_index))
        if args.analyze_cache:
            saved=load_npz(candidates/('%03d.npz'%cache_index));raw=[saved[key] for key in ('boxes','scores','labels')]
            seconds=None
        else:
            started=time.perf_counter();raw=run(feed_for(sample));seconds=time.perf_counter()-started
            np.savez(candidates/('%03d.npz'%index),**dict(zip(('boxes','scores','labels'),raw)))
        cache_checksums.append(dict(index=index,cache_index=cache_index,fp32_sha256=sha(refs/('%03d.npz'%cache_index)),
            candidate_sha256=sha(candidates/('%03d.npz'%cache_index))))
        mask=(raw[1]>=limits[raw[2]])|(expected['scores']>=limits[expected['labels']])
        for j,key in enumerate(('boxes','scores','labels')):
            actual_retained[j].append(raw[j][mask]);reference_retained[j].append(expected[key][mask])
        a=postprocess(*[expected[key] for key in ('boxes','scores','labels')],class_score_thresholds=profile['score_thresholds'])
        b=postprocess(*raw,class_score_thresholds=profile['score_thresholds'])
        matched=0
        if a['boxes_3d'] and b['boxes_3d']:
            cost=np.linalg.norm(np.asarray(a['boxes_3d'])[:,None,:2]-np.asarray(b['boxes_3d'])[None,:,:2],axis=-1)
            cost[np.asarray(a['labels'])[:,None]!=np.asarray(b['labels'])[None]]=1e6
            cost[cost>=.5]=1e6
            ia,ib=linear_sum_assignment(cost);matched=int((cost[ia,ib]<.5).sum())
            valid=cost[ia,ib]<.5;ia,ib=ia[valid],ib[valid]
            before,after=np.asarray(a['boxes_3d'])[ia],np.asarray(b['boxes_3d'])[ib]
            final_yaw_errors.extend(np.abs((before[:,6]-after[:,6]+np.pi)%(2*np.pi)-np.pi).tolist())
            final_box_errors.extend(np.max(np.abs(before[:,:6]-after[:,:6]),axis=-1).tolist())
            final_score_errors.extend(np.abs(np.asarray(a['scores'])[ia]-np.asarray(b['scores'])[ib]).tolist())
        total_before+=len(a['scores']);total_after+=len(b['scores']);total_matched+=matched
        rows.append(dict(bag=sample['bag'],token=str(sample['token']),network_with_image_decode_seconds=seconds,
            original_detections=len(a['scores']),engine_detections=len(b['scores']),matched=matched,
            all_raw_query_parity=compare_outputs(raw,expected,box_atol=.1,score_atol=.005,yaw_atol=yaw_atol)))
        print(sample['bag'],index+1,'/',len(cases),'matches',matched,'/',len(a['scores']),flush=True)
    actual=[np.concatenate(value) for value in actual_retained]
    expected={key:np.concatenate(reference_retained[j]) for j,key in enumerate(('boxes','scores','labels'))}
    if not len(actual[0]):
        raise ValueError('Empty retained-query validation is not an accuracy proof')
    parity=compare_outputs(actual,expected,box_atol=.1,score_atol=.005,yaw_atol=yaw_atol)
    yaw=np.abs((actual[0][:,6]-expected['boxes'][:,6]+np.pi)%(2*np.pi)-np.pi)
    retention=total_matched/max(total_before,1)
    precision=total_matched/max(total_after,1)
    passed=bool(parity['passed'] and np.percentile(yaw,95)<=np.deg2rad(.5) and retention>=.99 and precision>=.99)
    report=dict(passed=passed,checkpoint_sha256=model_meta['checkpoint_sha256'],onnx_sha256=sha(args.model),
        reference_onnx_sha256=sha(args.reference_model),
        engine_sha256=sha(args.engine) if args.engine else None,
        backend='TensorRT encoder + ONNX Runtime CUDA head' if args.encoder_engine else 'TensorRT' if args.engine else 'ONNX Runtime CUDA',
        iobinding=args.iobinding,
        cuda_tuning=args.cuda_tuning,cuda_graph=args.cuda_graph,threads=args.threads,
        custom_ops=model_meta.get('custom_ops'),
        split_encoder=str(args.split_encoder.resolve()) if args.split_encoder else None,
        encoder_precision=runner.encoder_precision if args.split_encoder and not args.analyze_cache else None,
        encoder_engine_sha256=sha(args.encoder_engine) if args.encoder_engine else None,
        cases=rows,retained_raw_query_count=len(actual[0]),retained_fp32_parity=parity,
        retained_yaw_p95_degrees=float(np.degrees(np.percentile(yaw,95))),
        maximum_heading_difference_deg=args.max_heading_difference_deg,
        heading_difference_reference='Original FP32 ONNX model, not absolute ground-truth accuracy',
        validation_mode='saved_raw_output_analysis_no_timing' if args.analyze_cache else 'paired_GPU_inference',
        reference_cache=str(args.reference_cache.resolve()) if args.reference_cache else None,
        raw_output_checksums=cache_checksums,
        original_detections=total_before,engine_detections=total_after,matched_detections=total_matched,
        baseline_retention=retention,candidate_match_fraction=precision,
        final_matched_max_box_error_m=max(final_box_errors,default=None),
        final_matched_max_score_error=max(final_score_errors,default=None),
        final_matched_max_yaw_error_degrees=float(np.degrees(max(final_yaw_errors))) if final_yaw_errors else None,
        final_matched_yaw_p95_degrees=float(np.degrees(np.percentile(final_yaw_errors,95))) if final_yaw_errors else None,
        scope=('Up to %d selected real frames per source manifest; thresholds fixed to the inherited checkpoint profile. '%args.per_manifest)+
              'Not a complete detection benchmark or manual annotation validation. Low-score raw-query discrepancies are retained in each case.')
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ('cases','raw_output_checksums')},indent=2),flush=True)
    if not passed:
        raise RuntimeError('Deployment accuracy gate failed; retain FP32/PyTorch baseline')


if __name__=='__main__':
    main()

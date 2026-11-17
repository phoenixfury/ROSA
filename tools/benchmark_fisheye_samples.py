#!/usr/bin/env python3
"""Measure warmed inference, JPEG loading and NMS on distinct real samples."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from benchmark_fisheye_backends import DeviceMemory,stats,np,sha,ROSA


def select_samples(manifests,per_manifest):
    selected=[];counts={}
    for value in manifests:
        name,path=value.split('=',1);path=Path(path).resolve()
        doc=json.loads(path.read_text());root=Path(doc.get('data_root','.'))
        samples=doc['samples'];indices=np.unique(np.linspace(0,len(samples)-1,min(per_manifest,len(samples))).astype(int))
        if name in counts:raise ValueError('Duplicate source name '+name)
        counts[name]=len(indices)
        for index in indices:
            sample=dict(samples[index]);sample['bag']=name
            if 'image_relative_paths' in sample:
                sample['image_paths']=[str((root/p).resolve()) for p in sample.pop('image_relative_paths')]
            if 'calibration' in sample:
                p=Path(sample['calibration']);sample['calibration']=str(p if p.is_absolute() else (path.parent/p).resolve())
            selected.append(sample)
    if not selected:raise ValueError('No benchmark samples')
    return selected,counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--threshold-profile',type=Path,required=True)
    p.add_argument('--manifest',action='append',required=True,metavar='NAME=JSON')
    p.add_argument('--per-manifest',type=int,default=20)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split-encoder',type=Path)
    p.add_argument('--encoder-engine',type=Path)
    p.add_argument('--iobinding',action='store_true')
    p.add_argument('--cuda-graph',action='store_true')
    p.add_argument('--custom-op-library',type=Path)
    p.add_argument('--cuda-tuning',choices=('default','throughput'),default='default')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--prefetch',action='store_true')
    args=p.parse_args()
    if args.per_manifest<1:p.error('--per-manifest must be positive')
    if args.output.exists():raise FileExistsError(args.output)
    selected,counts=select_samples(args.manifest,args.per_manifest)
    out=args.output.resolve();out.mkdir(parents=True)
    samples=out/'samples.json';samples.write_text(json.dumps(dict(samples=selected),indent=2)+'\n')
    command=[sys.executable,str(ROSA/'tools/infer_fisheye_onnx.py'),'--model',str(args.model.resolve()),
        '--threshold-profile',str(args.threshold_profile.resolve()),'--samples',str(samples),
        '--warmup','3','--threads',str(args.threads),'--cuda-tuning',args.cuda_tuning,
        '--output',str(out/'predictions.json')]
    for name in ('split_encoder','encoder_engine','custom_op_library'):
        if getattr(args,name):command+=['--'+name.replace('_','-'),str(getattr(args,name).resolve())]
    for name in ('iobinding','cuda_graph','prefetch'):
        if getattr(args,name):command+=['--'+name.replace('_','-')]
    started=time.perf_counter()
    with DeviceMemory() as memory:
        subprocess.run(command,check=True)
    predictions=json.loads((out/'predictions.json').read_text())
    report=dict(model=str(args.model.resolve()),onnx_sha256=sha(args.model),checkpoint_sha256=predictions['checkpoint_sha256'],
        samples=len(selected),source_counts=counts,providers=predictions['providers'],
        prefetch=args.prefetch,
        warmup_seconds=predictions['warmup_seconds'],process_seconds=time.perf_counter()-started,
        peak_gpu_device_mib=memory.peak/1024**2,baseline_gpu_device_mib=memory.baseline/1024**2,
        timings={name:stats(predictions[key]) for name,key in dict(network='seconds_per_frame',
            decode='decode_seconds_per_frame',postprocess='postprocess_seconds_per_frame',
            end_to_end='end_to_end_seconds_per_frame').items()},
        network_frames_under350_ms=sum(t<.350 for t in predictions['seconds_per_frame']),
        end_to_end_frames_under350_ms=sum(t<.350 for t in predictions['end_to_end_seconds_per_frame']),
        scope='Distinct four-camera 800x800 samples, batch1. Network includes host/device I/O. '
              'End-to-end adds consumer input wait and class-wise rotated NMS; --prefetch overlaps CPU image/calibration loading. '
              'Rendering and cold session/engine initialization excluded from per-frame figures; no concurrent GPU jobs.')
    (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

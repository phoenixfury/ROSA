#!/usr/bin/env python3
"""Run isolated epoch6 deployment comparisons; never select a default model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROSA=Path(__file__).resolve().parents[1]


def variants(root):
    return dict(
        fp32=(root/'fp32/rosa_epoch6.onnx',['--iobinding','--threads','2']),
        fast_ort=(root/'fast_ort/model/backbone-fp16.onnx',['--iobinding','--threads','2']),
        gpu_fp32=(root/'gpu_full_fp32/gpu_argmax.onnx',['--iobinding','--threads','2']),
        gpu_fp32_graph=(root/'gpu_full_fp32/gpu_argmax.onnx',['--iobinding','--cuda-graph','--threads','2']),
        gpu_fp16=(root/'gpu_encoder_fp16/backbone-fp16.onnx',['--iobinding','--threads','2']),
        gpu_fp16_graph=(root/'gpu_encoder_fp16/backbone-fp16.onnx',['--iobinding','--cuda-graph','--threads','2']),
        gpu_fp16_graph_prefetch=(root/'gpu_encoder_fp16/backbone-fp16.onnx',['--iobinding','--cuda-graph','--threads','2','--prefetch']),
        gpu_fp16_graph_tuned=(root/'gpu_encoder_fp16/backbone-fp16.onnx',['--iobinding','--cuda-graph','--threads','2','--prefetch','--cuda-tuning','throughput']),
        protected_graph=(root/'protected_encoder_fp16/backbone-fp16.onnx',['--iobinding','--cuda-graph','--threads','2']),
        protected_tail_graph=(root/'protected_tail_encoder_fp16/backbone-fp16.onnx',['--iobinding','--cuda-graph','--threads','2']),
        protected_stage4_graph=(root/'protected_stage4_encoder_fp16/backbone-fp16.onnx',['--iobinding','--cuda-graph','--threads','2']),
        trt_encoder=(root/'gpu_full_fp32/gpu_argmax.onnx',['--split-encoder',str(root/'gpu_split_fp32'),
            '--encoder-engine',str(root/'trt_encoder/rosa.engine'),'--cuda-graph','--threads','1']),
        trt_encoder_prefetch=(root/'gpu_full_fp32/gpu_argmax.onnx',['--split-encoder',str(root/'gpu_split_fp32'),
            '--encoder-engine',str(root/'trt_encoder/rosa.engine'),'--cuda-graph','--threads','1','--prefetch']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROSA/'work_dirs/epoch6_deployment_20261107')
    parser.add_argument('--stage',choices=('check','bench'),required=True)
    parser.add_argument('--candidate',action='append',required=True)
    parser.add_argument('--per-manifest',type=int,default=20)
    args=parser.parse_args();root=args.root.resolve();options=variants(root)
    checks=root/'checks_first3';reference=root/'fp32/rosa_epoch6.onnx'
    if args.per_manifest<1:parser.error('per-manifest must be positive')
    for tag in args.candidate:
        if tag not in options:parser.error('Unknown candidate '+tag)
        model,flags=options[tag]
        if not model.is_file():raise FileNotFoundError(model)
        # Decoding prefetch does not change the network accuracy test.
        if args.stage=='check':flags=[f for f in flags if f!='--prefetch']
        folder=root/f'{tag}_{args.stage}_{args.per_manifest}'
        script='validate_fisheye_deployment.py' if args.stage=='check' else 'benchmark_fisheye_samples.py'
        command=[sys.executable,str(ROSA/'tools'/script),'--model',str(model),*flags,
            '--threshold-profile',str(checks/'uniform_profile.json'),'--per-manifest',str(args.per_manifest),
            '--output',str(folder)]
        if args.stage=='check':
            command+=['--reference-model',str(reference),'--max-heading-difference-deg','10',
                      '--reference-cache',str(root/'raw_encoder_first3_validation')]
        for sequence in ('16_09','17_09','30_09'):
            command+=['--manifest',sequence+'='+str(checks/(sequence+'.json'))]
        record=root/(folder.name+'_invocation.json')
        if record.exists():raise FileExistsError(record)
        print('START',tag,args.stage,'log',str(root/(folder.name+'.log')),flush=True)
        with (root/(folder.name+'.log')).open('w') as stream:
            completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        record.write_text(json.dumps(dict(command=command,return_code=completed.returncode),indent=2)+'\n')
        print('FINISH',tag,'return_code',completed.returncode,flush=True)


if __name__=='__main__':main()

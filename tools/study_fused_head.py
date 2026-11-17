#!/usr/bin/env python3
"""Reproducible, sequential first-three-bag fusion ablations (no promotion)."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROSA=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROSA/'work_dirs/epoch6_fused_head_20261108')
    p.add_argument('--reference-root',type=Path,default=ROSA/'work_dirs/epoch6_deployment_20261107')
    p.add_argument('--stage',choices=('check','bench'),required=True)
    p.add_argument('--candidate',action='append',required=True)
    p.add_argument('--per-manifest',type=int,default=20)
    args=p.parse_args();root=args.root.resolve();ref=args.reference_root.resolve()
    for tag in args.candidate:
        if tag not in ('baseline','sampling','sampling_attention','all','all_nhwc','all_nhwc_release'):
            p.error('Unknown candidate '+tag)
        model=ref/'gpu_encoder_fp16/backbone-fp16.onnx' if tag=='baseline' else root/tag/'model.onnx'
        out=root/(tag+'_'+args.stage+'_'+str(args.per_manifest))
        if out.exists():raise FileExistsError(out)
        script='validate_fisheye_deployment.py' if args.stage=='check' else 'benchmark_fisheye_samples.py'
        command=[sys.executable,str(ROSA/'tools'/script),'--model',str(model),
            '--threshold-profile',str(ref/'checks_first3/uniform_profile.json'),
            '--iobinding','--cuda-graph','--threads','2','--per-manifest',str(args.per_manifest),'--output',str(out)]
        if tag!='baseline':
            library=json.loads(model.with_suffix('.json').read_text())['custom_ops']['library']
            command+=['--custom-op-library',library]
        if args.stage=='check':
            command+=['--reference-model',str(ref/'fp32/rosa_epoch6.onnx'),
                '--reference-cache',str(ref/'raw_encoder_first3_validation'),
                '--max-heading-difference-deg','10']
        else:command+=['--prefetch']
        for bag in ('16_09','17_09','30_09'):
            command+=['--manifest',bag+'='+str(ref/'checks_first3'/(bag+'.json'))]
        log=out.with_suffix('.log');print('START',tag,args.stage,str(log),flush=True)
        with log.open('w') as stream:
            status=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT).returncode
        out.with_suffix('.invocation.json').write_text(json.dumps(dict(command=command,return_code=status),indent=2)+'\n')
        print('FINISH',tag,args.stage,'exit',status,flush=True)


if __name__=='__main__':main()

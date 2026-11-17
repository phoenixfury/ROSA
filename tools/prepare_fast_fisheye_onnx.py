#!/usr/bin/env python3
"""Prepare a reproducible split-backend candidate without replacing a baseline.

This compiles a candidate, NOT an accuracy certificate. Run the paired-frame
validator and official held-out metrics before promoting a new checkpoint.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROSA=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True,help='Canonical FP32 export with its JSON metadata')
    p.add_argument('--output',type=Path,required=True,help='New build directory; existing directories are refused')
    p.add_argument('--backend',choices=('ort','split-tensorrt'),default='ort',
        help='Default: the user-selected FP16 encoder/FP32 head in plain ORT. TensorRT remains experimental.')
    p.add_argument('--encoder-precision',choices=('fp32','backbone-fp16','tf32'),default='backbone-fp16')
    p.add_argument('--workspace-mib',type=int,default=768)
    args=p.parse_args()
    if not args.model.is_file():p.error('Missing model')
    metadata=json.loads(args.model.with_suffix('.json').read_text())
    if not metadata['precision'].lower().startswith('fp32'):p.error('Start with the canonical FP32 export')
    if args.workspace_mib<1:p.error('Workspace must be positive')
    if args.output.exists():raise FileExistsError(args.output)
    out=args.output.resolve();out.mkdir(parents=True)
    commands=[]
    def run(script,*arguments):
        command=[sys.executable,str(ROSA/'tools'/script),*map(str,arguments)]
        commands.append(command)
        print('$',' '.join(command),flush=True)
        subprocess.run(command,check=True)
    def optimize(stage,source,folder):
        run('optimize_fisheye_onnx.py',stage,'--model',source,'--output',out/folder)
    if args.backend=='ort':
        optimize('simplify',args.model.resolve(),'simplify')
        run('optimize_fisheye_onnx.py','convert','--model',out/'simplify/simplified.onnx',
            '--precision','backbone-fp16','--output',out/'model')
        report=dict(status='UNVALIDATED_DO_NOT_PROMOTE',source=str(args.model.resolve()),
            model=str(out/'model/backbone-fp16.onnx'),commands=commands,
            runtime_flags=['--iobinding','--threads','2'],
            validation_flags=['--max-heading-difference-deg','5'],
            note='FP16 image encoder, FP32 head, standard ONNX only. The 5-degree maximum is the user-approved '
                 'difference versus original FP32, not an absolute ground-truth guarantee. Validate each new checkpoint.')
        (out/'candidate.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True);return
    optimize('trt-prepare',args.model.resolve(),'uint8')
    optimize('simplify',out/'uint8/trt_prepared.onnx','simplify')
    run('build_fisheye_tensorrt.py','--model',out/'simplify/simplified.onnx',
        '--parse-only','--fold-static-squeezes','--output',out/'shape_proofs')
    source=out/'shape_proofs/trt_static.onnx'
    if not source.exists():source=out/'simplify/simplified.onnx'
    optimize('specialize-shapes',source,'shapes')
    optimize('gpu-atan',out/'shapes/static_shapes.onnx','atan')
    optimize('gpu-argmax',out/'atan/gpu_atan.onnx','full')
    optimize('split-encoder',out/'full/gpu_argmax.onnx','split')
    run('build_fisheye_tensorrt.py','--model',out/'split/encoder.onnx',
        '--precision',args.encoder_precision,'--workspace-mib',args.workspace_mib,'--output',out/'encoder_engine')
    report=dict(status='UNVALIDATED_DO_NOT_PROMOTE',source=str(args.model.resolve()),
        model=str(out/'full/gpu_argmax.onnx'),split_encoder=str(out/'split'),
        encoder_engine=str(out/'encoder_engine/rosa.engine'),commands=commands,
        runtime_flags=['--split-encoder',str(out/'split'),'--encoder-engine',str(out/'encoder_engine/rosa.engine'),
                       '--cuda-graph','--threads','1'],
        note='Weights unchanged. Standard ONNX atan approximation and TopK class selection; '
             'TensorRT image encoder only; FP32 ONNX detection head. Validate on the target GPU. '
             'No baseline artifacts are overwritten; intermediate graphs are kept for reproducibility.')
    (out/'candidate.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

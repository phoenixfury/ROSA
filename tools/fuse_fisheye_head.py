#!/usr/bin/env python3
"""Create a separate checksum-bound fused CUDA ONNX candidate; no default changes."""
import argparse
import json
from pathlib import Path
import sys

ROSA=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROSA),str(ROSA/'work_dirs/onnx_deps_py38')]
import onnx
from deployment.fused_head import fuse
from benchmark_fisheye_backends import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--technique',action='append',choices=('sampling','projection','attention'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sampling-layout',choices=('nchw','nhwc'),default='nchw')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    metadata=json.loads(args.model.with_suffix('.json').read_text())
    if metadata['onnx_sha256']!=sha(args.model):raise ValueError('Source checksum mismatch')
    build=json.loads((args.library.parent/'build.json').read_text())
    if sha(args.library)!=build['library_sha256']:raise ValueError('Library checksum mismatch')
    model,report=fuse(onnx.load(str(args.model)),args.technique,args.sampling_layout)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,str(args.output))
    metadata.update(onnx_sha256=sha(args.output),node_count=len(model.graph.node),
        custom_ops=dict(domain='com.rosa.cuda',library=str(args.library.resolve()),
                        library_sha256=sha(args.library),api_version=19),
        head_fusion=report,source_onnx_sha256=sha(args.model),
        custom_op_policy='Opt-in ORT CUDA only; unchanged learned RFL and FP32 head; not TensorRT portable')
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='boundaries'}),flush=True)


if __name__=='__main__':main()

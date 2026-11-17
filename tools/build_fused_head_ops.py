#!/usr/bin/env python3
"""Build opt-in ORT CUDA operators without a PyTorch dependency in the .so."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROSA = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function(source, signature):
    start = source.index(signature)
    body = source.index('{', start)
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end] + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ort-include', type=Path, default=Path('/opt/onnxruntime/include'))
    p.add_argument('--nvcc', default=shutil.which('nvcc'))
    p.add_argument('--arch', default='86', help='CUDA compute capability, RTX3060/3080Ti=86')
    p.add_argument('--host-compiler', help='nvcc-compatible host C++ compiler')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rebuild', action='store_true', help='Explicitly rebuild this generated library')
    args = p.parse_args()
    if not args.nvcc: p.error('nvcc is required; activate the CUDA toolkit environment')
    if not args.arch.isdigit(): p.error('arch must be numeric, e.g.86')
    if not (args.ort_include/'onnxruntime_c_api.h').is_file(): p.error('Missing ORT C API header')
    args.output.mkdir(parents=True, exist_ok=True)
    library = args.output/'librosa_head_ort.so'
    if library.exists() and not args.rebuild: raise FileExistsError(library)
    native = ROSA/'ops/src/deformable_aggregation_cuda.cu'
    text = native.read_text()
    signatures = ['__device__ __forceinline__ float v49_safe_value(',
        '__device__ __forceinline__ void v49_normalize3(',
        '__global__ void v49_point_attention_kernel(',
        '__global__ void v49_project_geometry_kernel(']
    generated = args.output/'native_kernels.inc'
    generated.write_text('// Generated from maintained native CUDA, not a separate implementation.\n'+
                         '\n'.join(function(text, s) for s in signatures))
    source = ROSA/'deployment/cuda/fused_head.cu'
    command = [str(args.nvcc), '-std=c++17', '-O3', '--shared', '-Xcompiler=-fPIC',
        '--fmad=false', '-gencode', f'arch=compute_{args.arch},code=sm_{args.arch}',
        '-gencode', f'arch=compute_{args.arch},code=compute_{args.arch}',
        '-I'+str(args.ort_include.resolve()), '-I'+str(args.output.resolve()),
        str(source), '-o', str(library.resolve())]
    if args.host_compiler: command += ['-ccbin', args.host_compiler]
    print(' '.join(command), flush=True)
    subprocess.run(command, check=True)
    dependencies = subprocess.check_output(['ldd', str(library)], text=True)
    if any(s in dependencies for s in ('libtorch', 'libpython', 'libc10')):
        raise RuntimeError('Unexpected training framework dependency')
    record = dict(command=command, library_sha256=sha(library), api_version=19,
        source_sha256={str(path.resolve().relative_to(ROSA)):sha(path) for path in (source,native,Path(__file__))},
        native_functions=signatures, generated_sha256=sha(generated), dependencies=dependencies,
        note='ORT>=1.19 C API; no fast-math/FMA contraction; GPU-specific cubin plus forward PTX')
    (args.output/'build.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__': main()

#!/usr/bin/env python3
"""Synchronized encoder/head wall-time assay, not an ORT host-node proxy.

The extra stage barriers perturb scheduling. Use benchmark_fisheye_samples.py
for deployment latency on distinct images without these barriers.
"""
import argparse
import json
from pathlib import Path
import time

from benchmark_fisheye_backends import DeviceMemory, stats, sha
from deployment.runtime import load_npz, postprocess
from deployment.split_runtime import SplitEncoderRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--split-encoder', type=Path, required=True)
    parser.add_argument('--encoder-engine', type=Path)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cuda-graph', action='store_true')
    parser.add_argument('--custom-op-library', type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('Repeats must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    torch.set_num_threads(2)
    feed = load_npz(args.input)
    rows = []
    with DeviceMemory() as memory:
        runner = SplitEncoderRunner(args.model, args.split_encoder,
            args.encoder_engine, args.threads, args.cuda_graph, custom_op_library=args.custom_op_library)
        for _ in range(3):
            runner.run(feed)
        for index in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            if runner.encoder:
                runner.buffers['images'].update_inplace(feed['images'])
            torch.cuda.synchronize()
            after_upload = time.perf_counter()
            if runner.encoder:
                runner.encoder.run_with_iobinding(runner.encoder_binding)
            else:
                runner.trt.run_device({'images':feed['images']})
            torch.cuda.synchronize()
            after_encoder = time.perf_counter()
            runner.head.run_with_iobinding(runner.head_binding)
            torch.cuda.synchronize()
            after_head = time.perf_counter()
            raw = runner.head_binding.copy_outputs_to_cpu()
            after_copy = time.perf_counter()
            postprocess(*raw, score_threshold=.1)
            end = time.perf_counter()
            row = dict(input_upload=after_upload-start, encoder=after_encoder-after_upload,
                head=after_head-after_encoder, output_download=after_copy-after_head,
                postprocess=end-after_copy, synchronized_total=end-start)
            rows.append(row)
            print(index+1, {k:round(v*1000,2) for k,v in row.items()}, flush=True)
    result = dict(model=str(args.model.resolve()), model_sha256=sha(args.model),
        input_sha256=sha(args.input), split_encoder=str(args.split_encoder.resolve()),
        encoder_engine=str(args.encoder_engine.resolve()) if args.encoder_engine else None,
        encoder_precision=runner.encoder_precision, cuda_graph=args.cuda_graph,
        gpu=torch.cuda.get_device_name(0), repeats=args.repeats,
        timings={key:stats([row[key] for row in rows]) for key in rows[0]},
        peak_gpu_device_mib=memory.peak/1024**2, per_iteration_seconds=rows,
        scope='Warm same-input stage wall times with full CUDA synchronization after each stage. '
              'Encoder includes backbone+FPN+embedded normalization; TensorRT encoder also includes image upload. '
              'Head includes six geometry/fusion/refinement stages. This is an attribution assay with extra barriers, '
              'not sustained end-to-end latency; JPEG I/O, calibration changes and initialization excluded. '
              'ORT host-node profile durations are not treated as device-kernel durations.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'per_iteration_seconds'}, indent=2), flush=True)


if __name__ == '__main__':
    main()

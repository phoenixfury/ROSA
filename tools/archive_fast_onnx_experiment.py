#!/usr/bin/env python3
"""Freeze E012's focused source patch and measured results, never model/data files."""
import json
from pathlib import Path
from archive_rosa_experiments import ROSA,cumulative_patch,immutable,check_patch,sha,copy_result

FILES=['deployment/runtime.py','deployment/onnx_math.py','deployment/onnx_static.py',
    'deployment/split_runtime.py','deployment/tensorrt_runtime.py','deployment/e007_fast_first3.json',
    'tools/benchmark_fisheye_backends.py','tools/optimize_fisheye_onnx.py',
    'tools/build_fisheye_tensorrt.py','tools/validate_fisheye_deployment.py',
    'tools/infer_fisheye_onnx.py','tools/prepare_fast_fisheye_onnx.py',
    'tools/benchmark_fisheye_samples.py','tools/archive_fast_onnx_experiment.py',
    'tests/test_fast_deployment.py','tests/test_onnx_math.py','docs/fast_onnx_inference.md']
BASE='635216a9b29b6ea4deefd423117327421009ce85'


def main():
    folder=ROSA/'experiments/E012_fast_onnx';folder.mkdir(parents=True,exist_ok=True)
    patch=folder/'rosa.patch';immutable(patch,cumulative_patch(FILES,BASE));check_patch(BASE,[patch])
    work=ROSA/'work_dirs/onnx_under350_20261105'
    artifacts={}
    sources={
        'first3_validation.json':work/'accepted_first3_5deg_60_real/validation.json',
        'four_bag_diagnostic.json':work/'accepted_encoder_5deg_80_real/validation.json',
        'old601_metrics.json':work/'original_encoder_fp16_old601_metrics.json',
        'selected_warm_benchmark.json':work/'initial_encoder_fp16/benchmark_iobinding.json',
        'four_bag_real_frame_benchmark.json':work/'accepted_encoder_real80/benchmark.json',
        'rejected_trt_fp16_validation.json':work/'split_trt_fp16_16_real/validation.json',
        'rejected_trt_fp16_benchmark.json':work/'split_trt_fp16_benchmark/benchmark.json',
        'slower_trt_tf32_benchmark.json':work/'split_trt_tf32_benchmark/benchmark.json',
        'selection.json':ROSA/'deployment/e007_fast_first3.json'}
    for name,source in sources.items():
        if not source.is_file():raise FileNotFoundError(source)
        artifacts[name]=copy_result(source,folder/name)
    selection=json.loads((folder/'selection.json').read_text())
    report=dict(experiment='E012_fast_onnx',status='accuracy_accepted_first3_only_sustained350_target_not_met',
        patch_kind='focused_current_compatible_snapshot_not_training_attestation',base_commit=BASE,
        source_sha256={name:sha(ROSA/name) for name in FILES},patch_sha256=sha(patch),
        patch_verification='git apply --check passed on clean pinned base',artifacts=artifacts,
        selection=selection,acceptance_bags=['16_09','17_09','30_09'],excluded_acceptance_bag='25_09',
        original_baseline_overwritten=False,
        notes='First3 acceptance uses the user-approved maximum5-degree FP32 difference; cached paired GPU raw outputs '
              'were reanalyzed after fixing NumPy-bool JSON serialization. The excluded25 result remains archived. '
              'No dataset/checkpoint changes and no claim of sustained sub350 or server latency.')
    (folder/'record.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(experiment=report['experiment'],status=report['status'],patch_sha256=sha(patch),
        patch_verified=True,source_files=len(FILES)),indent=2),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Freeze E013 deployment source and lightweight measurements, not models/data."""
import json
from pathlib import Path

from archive_rosa_experiments import ROSA,cumulative_patch,immutable,check_patch,sha,copy_result

BASE='635216a9b29b6ea4deefd423117327421009ce85'
FILES=['deployment/runtime.py','deployment/onnx_math.py','deployment/onnx_static.py',
    'deployment/split_runtime.py','deployment/tensorrt_runtime.py','deployment/epoch6_choices_first3.json',
    'tools/export_fisheye_onnx.py','tools/benchmark_fisheye_backends.py','tools/optimize_fisheye_onnx.py',
    'tools/build_fisheye_tensorrt.py','tools/validate_fisheye_deployment.py','tools/infer_fisheye_onnx.py',
    'tools/prepare_fast_fisheye_onnx.py','tools/benchmark_fisheye_samples.py','tools/prepare_deployment_checks.py',
    'tools/profile_fisheye_components.py','tools/study_epoch6_deployment.py','tools/report_epoch6_deployment.py',
    'tools/archive_epoch6_deployment.py','tests/test_fast_deployment.py','tests/test_onnx_math.py',
    'tests/test_deployment_prefetch.py','tests/test_deployment_reference_cache.py','docs/epoch6_fast_deployment.md']


def main():
    root=ROSA/'work_dirs/epoch6_deployment_20261107'
    output=ROSA/'experiments/E013_epoch6_deployment';output.mkdir(parents=True,exist_ok=True)
    patch=output/'rosa.patch';immutable(patch,cumulative_patch(FILES,BASE));check_patch(BASE,[patch])
    paths=[root/'choices.json',root/'REPORT.md',root/'fp32/pytorch_validation.json',root/'trt_encoder/build_report.json',
           root/'raw_encoder_first3_validation/validation.json',root/'profile_fp32/profile_summary.json']
    paths+=list(root.glob('*_check_20/validation.json'))+list(root.glob('*_bench_*/benchmark.json'))
    paths+=list(root.glob('profile_*.json'))
    if (root/'native_amp_benchmark.json').is_file():paths.append(root/'native_amp_benchmark.json')
    artifacts={str(p.relative_to(root)):copy_result(p,output/p.relative_to(root)) for p in paths}
    record=dict(experiment='E013_epoch6_deployment',status='user_choice_pending_no_default_changed',
        source_snapshot='Focused current-compatible cumulative patch, not checkpoint-training-source attestation',
        base_commit=BASE,patch_sha256=sha(patch),patch_verification='git apply --check passed on clean pinned base',
        source_sha256={name:sha(ROSA/name) for name in FILES},artifacts=artifacts,
        maximum_heading_difference_deg=10,acceptance_bags=['16_09','17_09','30_09'],
        excluded_acceptance_bag='25_09',baseline_overwritten=False,
        note='User relaxed only heading to10deg; all strict query checks remain visible. '
             'Model/engine and dataset files are not included in this archive. No default selected.')
    (output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    (output/'README.md').write_text('# E013: epoch6 deployment alternatives\n\n'
        'See REPORT.md and choices.json. All candidates and numerical warnings remain available for user choice.\n'
        'rosa.patch is a focused source snapshot against the pinned base; no weights or datasets are archived.\n')
    print(json.dumps(dict(experiment=record['experiment'],patch_verified=True,artifacts=len(artifacts)),indent=2))


if __name__=='__main__':main()

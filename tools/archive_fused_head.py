#!/usr/bin/env python3
"""Freeze E014 source patch and lightweight results; never weights/datasets."""
import json
from pathlib import Path

from archive_rosa_experiments import ROSA,immutable,cumulative_patch,check_patch,copy_result,sha
from archive_epoch6_deployment import FILES as PREVIOUS_FILES,BASE

ADDITIONS=['deployment/fused_head.py','deployment/cuda/fused_head.cu',
    'deployment/epoch6_fused_choices_first3.json','docs/fused_head_deployment.md',
    'tools/build_fused_head_ops.py','tools/check_fused_head_ops.py','tools/fuse_fisheye_head.py',
    'tools/study_fused_head.py','tools/report_fused_head.py','tools/archive_fused_head.py',
    'tests/test_fused_head.py']


def main():
    root=ROSA/'work_dirs/epoch6_fused_head_20261108'
    out=ROSA/'experiments/E014_fused_head';out.mkdir(parents=True,exist_ok=True)
    patch=out/'rosa.patch';files=sorted(set(PREVIOUS_FILES+ADDITIONS))
    immutable(patch,cumulative_patch(files,BASE));check_patch(BASE,[patch])
    paths=[root/name for name in ('REPORT.md','choices.json','parameter_integrity.json',
        'kernel_checks.json','kernel_checks_nhwc.json','kernel_checks_release.json')]
    paths+=list(root.glob('*_check_20/validation.json'))+list(root.glob('*_bench_20/benchmark.json'))
    paths+=list(root.glob('profile_*.json'))+list(root.glob('ops*/build.json'))
    paths+=list(root.glob('*.invocation.json'))
    paths+=list(root.glob('*/model.json'))
    artifacts={str(p.relative_to(root)):copy_result(p,out/p.relative_to(root)) for p in paths if p.is_file()}
    metadata=json.loads((root/'choices.json').read_text())
    record=dict(experiment='E014_fused_head',status='USER_CHOICE_PENDING_NO_DEFAULT_CHANGED',
        base_commit=BASE,patch_sha256=sha(patch),patch_verification='git apply --check passed on clean pinned base',
        source_snapshot='Focused cumulative deployment patch, not checkpoint training-source attestation',
        source_sha256={name:sha(ROSA/name) for name in files},
        checkpoint_sha256=metadata['checkpoint_sha256'],artifacts=artifacts,
        acceptance_bags=['16_09','17_09','30_09'],excluded_acceptance_bag='25_09',maximum_heading_difference_deg=10,
        default_changed=False,prior_experiments_modified=False,
        note='Custom ORT CUDA operators require a rebuilt checksum-bound library on the target server. '
             'No weights, ONNX binaries, CUDA binaries or data archived. Earlier trial build/library '
             'hashes are retained; ops_release is the build from final maintained CUDA source.')
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    (out/'README.md').write_text('# E014: fused ONNX head\n\nSee REPORT.md and choices.json. '
        'The immutable focused patch includes compatible E013 tooling against the pinned base. '
        'No default was changed; strict raw score failures remain explicit.\n')
    print(json.dumps(dict(patch_verified=True,artifacts=len(artifacts),experiment=record['experiment']),indent=2))


if __name__=='__main__':main()

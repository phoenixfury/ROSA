#!/usr/bin/env python3
"""Export lightweight experiment records/patches, never weights/data/Git writes.

Historical recipe patches have exact pinned commit bases but are not claimed
as proof of checkpoint training source. Uncommitted temporal work is exported
as explicitly reconstructed cumulative patches against the pinned ROSA HEAD.
Existing patch content is immutable; later invocations update result artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROSA = Path(__file__).resolve().parents[1]
ROOT = ROSA.parents[1]
OUT = ROSA/'experiments'
BASE = 'f305bfc'
CORE = ['rosa/integrations/fisheye3dod/dataset.py',
    'rosa/integrations/fisheye_box_sparse4d.py',
    'rosa/integrations/fisheye3dod/temporal.py',
    'configs/fisheye3dod/rosa_vov99_temporal_12e.py',
    'tests/test_temporal_training.py','tools/train_temporal_fisheye.py','tools/watch_temporal_training.py']
MEMORY = CORE + ['rosa/integrations/fisheye3dod/motion_targets.py',
    'tests/test_motion_targets.py','tests/test_temporal_memory_selection.py',
    'tools/audit_temporal_motion_targets.py','tools/audit_temporal_motion_provenance.py',
    'tools/test_temporal_memory.py','docs/temporal_memory_testing.md']
VELOCITY = MEMORY + ['rosa/integrations/fisheye3dod/velocity_recovery.py',
    'tools/run_velocity_recovery.py','tests/test_velocity_recovery.py','docs/velocity_gate_recovery.md']
JOINT_ADDITIONS = ['rosa/integrations/fisheye3dod/joint_velocity_recovery.py',
    'tools/run_joint_velocity_recovery.py','tests/test_joint_velocity_recovery.py','docs/joint_velocity_recovery.md']
JOINT = VELOCITY + JOINT_ADDITIONS
SINGLEFRAME = ['configs/fisheye3dod/rosa_vov99_singleframe_refresh_24e.py',
    'rosa/integrations/fisheye3dod/singleframe_plateau.py',
    'tools/train_singleframe_refresh.py', 'tests/test_singleframe_plateau.py',
    'docs/singleframe_refresh.md']
SPECS = [
    dict(id='E000_baseline', title='Original baseline and corrected decoding references',
        commit='899a7b7', artifact='retrain_comparison', metrics='retrain_comparison/analysis/comparison.json',
        report='retrain_comparison/REPORT.md', scope='Committed baseline export/inference recipe reference; checkpoint training source is not attested by this patch.'),
    dict(id='E001_balanced_retrain', title='Balanced clean retraining',
        commit='7e00a98', artifact='retrain_comparison', metrics='retrain_comparison/analysis/comparison.json',
        report='retrain_comparison/REPORT.md', scope='Exact committed recipe change; reports include the original/same-decoder comparison and offline NMS variants.'),
    dict(id='E002_baseline_adaptation', title='Baseline epoch-16 adaptation: live BN and backbone learning rate',
        commit='649f7de', artifact='adaptation_analysis', metrics='adaptation_analysis/metrics/comparison.json',
        report='adaptation_analysis/REPORT.md', scope='Exact committed recipe change; the server reported a dirty worktree, so this is not a complete attestation of its source.'),
    dict(id='E003_temporal', title='Chronological temporal query memory and causal motion targets',
        files=CORE, artifact='temporal_plateau_comparison_20261031',
        metrics='temporal_plateau_comparison_20261031/evaluations.json',
        report='temporal_plateau_comparison_20261031/REPORT.md',
        scope='Reconstructed current-compatible temporal runtime, not the original historical source snapshot. Includes later backward-compatible cache knobs; original epoch-4 defaults have an exact prediction canary.'),
    dict(id='E004_memory_motion_audit', title='Frozen memory policies and offline causal motion audit',
        files=MEMORY, artifact='temporal_memory_ablation_20261031',
        metrics='temporal_memory_ablation_20261031/results.json',
        report='temporal_memory_ablation_20261031/REPORT.md',
        scope='Cumulative reconstructed code snapshot. No training or default-policy promotion. Captured run source hashes remain the stronger evidence.'),
    dict(id='E005_velocity_rows', title='Paired velocity-row-only gate recovery pilot',
        files=VELOCITY, artifact='velocity_gate_recovery_20261031',
        metrics='velocity_gate_recovery_20261031/results.json', report='velocity_gate_recovery_20261031/REPORT.md',
        scope='Cumulative current source snapshot; captured E005 source files are checked against its run manifest. Only velocity/gate rows trained.'),
    dict(id='E006_joint_decoder_recovery', title='Paired joint decoder adaptation with gate recovery',
        files=JOINT, artifact='joint_decoder_recovery_20261031',
        metrics='joint_decoder_recovery_20261031/results.json', report='joint_decoder_recovery_20261031/REPORT.md',
        scope='Cumulative current source snapshot plus an exact new-file delta against the E005 patch. Backbone/FPN/BN/priors/Z rows frozen; full decoder adapts with native detection losses.'),
    dict(id='E007_singleframe_refresh', title='Non-temporal refresh from the best adapted single-frame checkpoint',
        files=SINGLEFRAME, artifact='singleframe_refresh_20261031',
        metrics='singleframe_refresh_20261031/results.json', report='singleframe_refresh_20261031/REPORT.md',
        scope='Independent non-temporal recipe on the pinned base; no E003-E006 patches required. Full model/live BN, current combined dataset, class-wise NMS, dual-metric automatic plateau stop.'),
]


def git(*args, cwd=ROSA):
    result = subprocess.run(['git',*map(str,args)],cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    return result.stdout


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def immutable(path,data):
    if path.exists():
        if path.read_bytes()!=data:
            raise RuntimeError(f'Frozen patch differs: {path}; create a new experiment ID instead of overwriting history')
    else:
        path.write_bytes(data)


def addition_patch(name,content):
    body=''.join(difflib.unified_diff([],content.splitlines(keepends=True),
        fromfile='/dev/null',tofile='b/'+name))
    return f'diff --git a/{name} b/{name}\nnew file mode 100644\n{body}'.encode()


def cumulative_patch(files,base):
    chunks=[]
    for name in sorted(set(files)):
        path=ROSA/name
        known=subprocess.run(['git','cat-file','-e',f'{base}:{name}'],cwd=ROSA,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        if known:
            chunks.append(git('diff','--binary',base,'--',name))
        else:
            chunks.append(addition_patch(name,path.read_text()))
    return b''.join(chunks)


def check_patch(base,patches):
    # Generated sandbox only; never touch the user's worktree/index/Git objects.
    archive=git('archive','--format=tar',base)
    with tempfile.TemporaryDirectory(prefix='rosa-experiment-patch-') as temporary:
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            for member in source.getmembers():
                assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            source.extractall(temporary)
        for patch in patches:
            git('apply','--check',patch.resolve(),cwd=temporary)
            git('apply',patch.resolve(),cwd=temporary)


def copy_result(source,target):
    if source.is_file():
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        return dict(source=str(source.relative_to(ROSA)),sha256=sha(target))
    return None


def export(spec,verify):
    directory=OUT/spec['id']; directory.mkdir(parents=True,exist_ok=True)
    if 'commit' in spec:
        commit=git('rev-parse',spec['commit']).decode().strip()
        base=git('rev-parse',f'{commit}^').decode().strip()
        patch=git('diff','--binary',base,commit)
        provenance='exact_committed_recipe_change_not_checkpoint_attestation'
        files=git('diff','--name-only',base,commit).decode().splitlines()
    else:
        commit=None; base=git('rev-parse',BASE).decode().strip()
        files=spec['files']; patch=cumulative_patch(files,base)
        provenance='current_compatible_cumulative_snapshot_not_exact_historical_delta'
    patch_path=directory/'rosa.patch'; immutable(patch_path,patch)
    if verify:
        check_patch(base,[patch_path])
    support_patch = None
    if spec['id'] == 'E007_singleframe_refresh':
        # The launcher also exports its final report. Supply that standalone
        # archive tool separately without rewriting the frozen recipe patch.
        support_patch = directory/'archive_tooling.patch'
        immutable(support_patch, addition_patch('tools/archive_rosa_experiments.py',
                                               Path(__file__).read_text()))
        if verify:
            check_patch(base,[patch_path,support_patch])
    hashes={name:sha(ROSA/name) for name in files if (ROSA/name).is_file()} if commit is None else {}
    artifacts={}
    root=ROSA/'work_dirs'
    for source,target in [(root/spec['metrics'],directory/'metrics.json'),
                          (root/spec['report'],directory/'analysis.md'),
                          (root/spec['artifact']/'plan.json',directory/'plan.json')]:
        record=copy_result(source,target)
        if record: artifacts[target.name]=record
    current=root/spec['artifact']
    extras={
        'E000_baseline':[(root/'temporal_plateau_comparison_20261031/evaluations.json','current_601_references.json')],
        'E002_baseline_adaptation':[(root/'temporal_plateau_comparison_20261031/evaluations.json','current_601_references.json')],
        'E004_memory_motion_audit':[(current/'evaluations.json','evaluations.json'),
            (current/'fixed_budget_controls/evaluations.json','fixed_budget_controls.json'),
            (current/'motion_audit/summary.json','motion_audit.json'),
            (current/'motion_audit/provenance_summary.json','source_aware_motion_audit.json'),
            (current/'motion_audit/REPORT.md','motion_analysis.md'),
            (current/'motion_audit/PROVENANCE_REPORT.md','source_aware_motion_analysis.md'),
            (current/'motion_audit/predicted_velocity_diagnostics.json','predicted_velocity_diagnostics.json'),
            (current/'motion_audit/velocity_gradient_probe.json','velocity_gradient_probe.json'),
            (current/'motion_audit/velocity_gradient_probe_neutral_gate.json','velocity_gradient_probe_neutral_gate.json')],
        'E005_velocity_rows':[(current/'gate_saturation.json','gate_saturation.json'),
            (current/'control/training.json','control_training.json'),(current/'recovery/training.json','recovery_training.json')],
        'E006_joint_decoder_recovery':[(current/'control/training.json','control_training.json'),
            (current/'recovery/training.json','recovery_training.json')],
        'E007_singleframe_refresh':[(current/'plateau.json','plateau.json'),
            (current/'balanced_preflight.json','balanced_preflight.json'),
            (current/'batch_probe/results.json','memory_probe.json')],
    }.get(spec['id'],[])
    for source,name in extras:
        record=copy_result(source,directory/name)
        if record: artifacts[name]=record
    # Preserve useful small diagnostics/relative report links, not data/model dumps.
    small=['learning_curves.png','gate_saturation.json','checkpoint_audit.json',
        'analysis/checkpoint_inventory.json','audit_findings.md','temporal_consistency.json',
        'results.md','results.json','motion_audit/REPORT.md','motion_audit/PROVENANCE_REPORT.md',
        'motion_audit/summary.json','motion_audit/provenance_summary.json']
    if spec['id'] in ('E005_velocity_rows','E006_joint_decoder_recovery'):
        small += [f'{arm}/{name}' for arm in ('control','recovery')
                  for name in ('training.json','steps.jsonl')]
        small += [f'{arm}_eval/evaluation.json' for arm in ('control','recovery')]
        plot_source=ROSA/'work_dirs/velocity_gate_recovery_20261031/plot_recovery.py'
        record=copy_result(plot_source,directory/'plot_recovery.py')
        if record: artifacts['plot_recovery.py']=record
    for name in small:
        source=current/name
        if source.is_file() and source.stat().st_size <= 2*1024*1024:
            record=copy_result(source,directory/name)
            if record: artifacts[name]=record
    matched={}; mismatched={}
    plan_path=current/'plan.json'
    captured_path=plan_path if plan_path.is_file() else current/'evaluations.json'
    if captured_path.is_file():
        captured=json.loads(captured_path.read_text()).get('source_sha256',{})
        for name,expected in captured.items():
            if name.startswith('projects/ROSA/'):
                relative=name.removeprefix('projects/ROSA/') if hasattr(str,'removeprefix') else name[len('projects/ROSA/'):]
                if (ROSA/relative).is_file():
                    actual=sha(ROSA/relative)
                    (matched if actual==expected else mismatched)[relative]=dict(expected=expected,current=actual)
        if mismatched:
            raise RuntimeError(f'Captured source differs for {spec["id"]}: {mismatched}')
    delta_spec={
        'E004_memory_motion_audit':('E003_temporal',CORE,MEMORY),
        'E005_velocity_rows':('E004_memory_motion_audit',MEMORY,VELOCITY),
        'E006_joint_decoder_recovery':('E005_velocity_rows',VELOCITY,JOINT),
    }.get(spec['id'])
    if delta_spec:
        parent_id,previous,now=delta_spec
        delta=directory/f'changes_from_{parent_id[:4]}.patch'
        immutable(delta,b''.join(addition_patch(name,(ROSA/name).read_text())
            for name in now if name not in previous))
        if verify:
            check_patch(base,[OUT/parent_id/'rosa.patch',delta])
    status='complete' if (current/'REPORT.md').is_file() else ('running' if plan_path.is_file() else 'historical')
    record=dict(id=spec['id'],title=spec['title'],status=status,patch_kind=provenance,
        exported_at_utc=datetime.now(timezone.utc).isoformat(),
        patch_base_commit=base,recipe_commit=commit,patch_sha256=sha(patch_path),
        scope=spec['scope'],source_sha256=hashes,captured_run_source_matches=matched,
        artifacts=artifacts,work_dir=str(current.relative_to(ROSA)),
        external_dependencies='See ../dependencies.json and ../external_mmdetection3d.patch',
        verification='git apply --check on clean pinned base' if verify else 'not rerun this invocation')
    if support_patch:
        record['archive_tooling_patch'] = dict(file=support_patch.name, sha256=sha(support_patch))
    write_json(directory/'record.json',record)
    write_json(directory/'changes.json',dict(experiment=spec['id'],change=spec['title'],
        interpretation=spec['scope'],patch_base_commit=base,files=files,
        relative_delta=(f'changes_from_{delta_spec[0][:4]}.patch' if delta_spec else None),
        relative_delta_kind='Exact additions between archived compatible snapshots, not a claim of historical runtime chronology.'))
    command={
        'E003_temporal':'python tools/train_temporal_fisheye.py --help',
        'E004_memory_motion_audit':'python tools/test_temporal_memory.py --epoch 4',
        'E005_velocity_rows':'python tools/run_velocity_recovery.py --output-dir work_dirs/NEW_E005_TRIAL',
        'E006_joint_decoder_recovery':'python tools/run_joint_velocity_recovery.py --output-dir work_dirs/NEW_E006_TRIAL',
        'E007_singleframe_refresh':'python tools/train_singleframe_refresh.py --data-root "$ROSA_DATA_ROOT" --gpus 1 --batch-size 1 --effective-batch 8',
    }.get(spec['id'],'See analysis.md and the committed recipe/config; exact checkpoint retraining is not asserted.')
    (directory/'README.md').write_text(f'# {spec["id"]}: {spec["title"]}\n\n'
        f'Status: {status}.\n\n{spec["scope"]}\n\n'
        f'Patch base: `{base}`. Patch SHA256: `{record["patch_sha256"]}`.\n\n'
        'The cumulative patch is applied independently on its clean pinned base; do not apply all cumulative patches in sequence. '
        'External host dependencies must also be present. See record.json for source/run checks and artifact hashes.\n\n'
        + ('For E007, also apply `archive_tooling.patch` after `rosa.patch`; it supplies '
           'the launcher\'s automatic final-report exporter on the pinned clean base.\n\n'
           if support_patch else '') +
        f'Reproduction entry point (from the ROSA directory with the documented environment):\n\n```bash\n{command}\n```\n\n'
        'Metrics retain their original validation hashes/protocols. Never directly rank the older 376-frame dataset against the current 601-frame split. '
        'Dataset/checkpoint/prediction binaries are intentionally not copied into this archive.\n')
    return record


def export_variants():
    parent=OUT/'E004_memory_motion_audit'
    for source in [ROSA/'work_dirs/temporal_memory_ablation_20261031/evaluations.json',
                   ROSA/'work_dirs/temporal_memory_ablation_20261031/fixed_budget_controls/evaluations.json']:
        data=json.loads(source.read_text())
        for name,record in data['models'].items():
            folder=parent/'variants'/name; folder.mkdir(parents=True,exist_ok=True)
            write_json(folder/'metrics.json',dict(metrics=record['metrics'],cache_options=record['cache_options'],
                validation_sha256=record['validation_sha256'],validation_samples=record['validation_samples'],
                checkpoint_sha256=record['checkpoint_sha256'],operating_points=record.get('operating_points'),
                cache_summary=record.get('cache_summary')))
            content=("# Apply after the E004 cumulative source patch. Runtime policy only.\n"
                "_base_ = ['../fisheye3dod/rosa_vov99_temporal_12e.py']\n"
                f"model = {dict(head=dict(instance_bank=record['cache_options']))!r}\n")
            immutable(folder/'policy.patch',addition_patch(f'configs/experiments/{name}.py',content))
            epoch=record['epoch']; variant=record['variant']
            (folder/'README.md').write_text(f'# {name}\n\n'
                'policy.patch adds a named configuration on top of E004; it is not an independent source patch. '
                'Frozen checkpoints and output decoder are unchanged. Metrics include occupancy controls.\n\n'
                f'Configuration options: `{record["cache_options"]}`.\n\n'
                f'Production runner: `python tools/test_temporal_memory.py --epoch {epoch} --variant {variant}`. '
                'The two budget_* controls use the archived fixed_budget_controls.py wrapper instead.\n')
    copy_result(ROSA/'work_dirs/temporal_memory_ablation_20261031/fixed_budget_controls.py',parent/'fixed_budget_controls.py')


def dependencies(verify):
    paths=['projects/mmdet3d_plugin/datasets/transforms/pack_sparse4d.py',
        'projects/mmdet3d_plugin/models/detection3d/target.py',
        'projects/mmdet3d_plugin/models/sparse4d_head.py']
    patch=git('diff','--binary','HEAD','--',*paths,cwd=ROOT)
    path=OUT/'external_mmdetection3d.patch'; immutable(path,patch)
    result=dict(mmdetection3d_commit=git('rev-parse','HEAD',cwd=ROOT).decode().strip(),
        rosa_current_commit=git('rev-parse','HEAD').decode().strip(),
        rosa_worktree_status=git('status','--short').decode().splitlines(),
        external_patch_sha256=sha(path),host_source_sha256={name:sha(ROOT/name) for name in paths},
        notes=['The ROSA patches alone cannot reconstruct modified host code.',
               'Fisheye3DOD and datasets/weights are separate dependencies; no binaries are copied.',
               'Git commits/index/worktrees are never modified by this exporter.'])
    official=ROOT/'projects/Fisheye3DOD'
    if (official/'.git').exists():
        result['fisheye3dod_commit']=git('rev-parse','HEAD',cwd=official).decode().strip()
        result['fisheye3dod_worktree_status']=git('status','--short',cwd=official).decode().splitlines()
    if verify:
        git('apply','--reverse','--check',path.resolve(),cwd=ROOT)
    write_json(OUT/'dependencies.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-patches',action='store_true')
    parser.add_argument('--only',action='append',choices=[spec['id'] for spec in SPECS],
        help='Refresh only selected records; useful after new experiments change old source files.')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    dependencies(args.verify_patches)
    selected=[spec for spec in SPECS if not args.only or spec['id'] in args.only]
    old=json.loads((OUT/'index.json').read_text())['experiments'] if (OUT/'index.json').is_file() else []
    records={record['id']:record for record in old}
    records.update({spec['id']:export(spec,args.verify_patches) for spec in selected})
    if not args.only or 'E004_memory_motion_audit' in args.only:
        export_variants()
    if args.verify_patches and (not args.only or 'E004_memory_motion_audit' in args.only):
        check_patch(git('rev-parse',BASE).decode().strip(),
            [OUT/'E004_memory_motion_audit/rosa.patch',
             *sorted((OUT/'E004_memory_motion_audit/variants').glob('*/policy.patch'))])
    write_json(OUT/'index.json',dict(schema_version=1,experiments=[records[key] for key in sorted(records)],
        rules='Exact committed recipe patches are not checkpoint attestation. Cumulative reconstructed patches are independent snapshots, not sequential historical diffs.'))
    print('Updated',len(selected),'experiment families; archive contains',len(records),'families at',OUT)


if __name__=='__main__':
    main()

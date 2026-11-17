#!/usr/bin/env python3
"""Fresh native old-601/new-158 validation before any mixed candidate export."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from train_balanced_fisheye import ROOT,ROSA
from infer_fisheye_checkpoints import export_portable_json
sys.path.insert(0,str(ROOT/'tools/custom_data'))
from infer_rosa_synchronized_bag import infer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--targets',type=Path,required=True)
    p.add_argument('--new-inference-only',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();run=args.run.resolve()
    report=json.loads((run/'train/training_report.json').read_text())
    config=ROSA/'configs/fisheye3dod/rosa_vov99_singleframe_refresh_24e.py'
    oldroot=ROOT/'data/Custom/combined_improved_detzero_rosa'
    prepared=ROOT/'data/Custom/BAG_2026_09_25_13_28_18_converted/dataset/fisheye3dod_inference_2hz'
    output=run/'fresh_new';output.mkdir(exist_ok=True)
    if args.new_inference_only:
        # Isolate model registry imports from the metric/audit modules.
        native=SimpleNamespace(config=config,checkpoint=Path(report['checkpoint']),workers=2,
            prepared=prepared,output=output)
        infer(native,args.targets.parent/'val_infos.pkl',output/'predictions.pkl')
        export_portable_json(output/'predictions.pkl',output/'predictions.json',native.checkpoint,.01)
        payload=json.loads((output/'predictions.json').read_text())
        payload['checkpoint_sha256']=report['checkpoint_sha256']
        (output/'predictions.json').write_text(json.dumps(payload,indent=2)+'\n')
        return
    old_path=run/'fresh_old/comparison.json'
    if old_path.exists():
        completed=json.loads(old_path.read_text())['checkpoints'][0]
        if completed['sha256']!=report['checkpoint_sha256'] or completed['prediction_frames']!=601:
            raise ValueError('Existing old validation belongs to another candidate')
    else:
        subprocess.run([sys.executable,str(ROSA/'tools/infer_fisheye_checkpoints.py'),
            '--config',str(config),'--checkpoint',report['checkpoint'],'--data-root',str(oldroot),
            '--output-dir',str(run/'fresh_old'),'--no-lidar-visualization'],check=True)
    old=json.loads((run/'fresh_old/comparison.json').read_text())['checkpoints'][0]['metrics']
    # Independent epoch-1 fresh native reference, same pinned old validation.
    history=json.loads((ROSA/'work_dirs/singleframe_refresh_20261031/results.json').read_text())['policy']['history']
    baseline=next(row['metrics'] for row in history if row['epoch']==1)
    old_pass=old['NDS']>=baseline['fisheye_bev/NDS']-.002 and old['mAP']>=baseline['fisheye_bev/mAP']*.99
    teacher_metrics=report.get('baseline_old',report.get('baseline_old_metrics',{}))
    failed_orientation=[key for key,value in teacher_metrics.items()
                        if key.endswith('_AOE') and old[key]>value+.03]
    old_pass=old_pass and not failed_orientation
    targets=json.loads(args.targets.read_text())
    subset=dict(frames_data=[row for row in targets['frames_data'] if row['split']=='val'])
    if not (output/'predictions.json').exists():
        with (output/'inference.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run',str(run),
                '--targets',str(args.targets.resolve()),'--new-inference-only'],stdout=log,stderr=subprocess.STDOUT,check=True)
    from evaluate_heading_adaptation import angle_pairs,common_angles
    candidate=json.loads((output/'predictions.json').read_text())
    if candidate.get('checkpoint_sha256')!=report['checkpoint_sha256']:
        raise ValueError('Existing new-bag predictions lack matching checkpoint provenance; regenerate them')
    thresholds=json.loads((run/'train/threshold_profile.json').read_text())['score_thresholds']
    for frame in candidate['frames']:
        keep=[i for i,(n,s) in enumerate(zip(frame['names'],frame['scores'])) if s>=thresholds[n]]
        for key in ('boxes_3d','names','scores','labels'):
            frame[key]=[frame[key][i] for i in keep]
    before=json.loads((ROSA/'work_dirs/static_heading_adapt_20261105/native_original/predictions.json').read_text())
    tokens={row['token'] for row in subset['frames_data']}
    before['frames']=[f for f in before['frames'] if f['token'] in tokens]
    heading=common_angles(angle_pairs(before,subset),angle_pairs(candidate,subset))
    new_pass=heading['common_matches']>0 and heading['all']['after']['axis_mean_deg']<heading['all']['before']['axis_mean_deg']-.05
    result=dict(passed=bool(old_pass and new_pass),old_gate_passed=bool(old_pass),new_gate_passed=bool(new_pass),
        failed_old_orientation_gates=failed_orientation,
        old_metrics=old,baseline_old=baseline,new_heading=heading,
        caveat='Fresh native same-scene pseudo-label evaluation; no claim of unseen-scene generalization.')
    (run/'fresh_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()

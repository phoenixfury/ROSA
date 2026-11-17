#!/usr/bin/env python3
"""Queue guarded training behind full replay caching, then validate and export.

No deployment candidate is promoted over the baseline automatically. Status and
logs remain available if training/validation rejects the candidate.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROSA=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--targets',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Reuse completed training and continue validation/export')
    args=p.parse_args()
    run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    def status(phase,**values):
        doc=dict(phase=phase,updated_unix=time.time(),**values)
        temporary=run/'workflow_status.partial.json'
        temporary.write_text(json.dumps(doc,indent=2)+'\n')
        temporary.replace(run/'workflow_status.json')
        print(json.dumps(doc),flush=True)
    def command(name,values):
        values=list(map(str,values))
        status(name,command=values,log=str(run/(name+'.log')))
        with (run/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,*map(str,values)],stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        while True:
            try:
                manifest=json.loads((run/'features/cache_manifest.json').read_text())
                rows=manifest['datasets']['old_train']
                # Pin exact expected complete source count without importing the whole model stack.
                expected=3278
                count=len(rows['records'])
                status('preparing_full_old_replay',cached=count,expected=expected)
                if count==expected:
                    break
                if count>expected:
                    raise ValueError('Unexpected training source size; review the recipe')
            except (FileNotFoundError,json.JSONDecodeError,KeyError):
                pass
            time.sleep(15)
        if not args.resume or not (run/'train/training_report.json').exists():
            command('training',[ROSA/'tools/finetune_mixed_heading.py','--cache',run/'features',
                '--targets',args.targets,'--output',run/'train'])
        training=json.loads((run/'train/training_report.json').read_text())
        if training.get('status')!='complete':
            raise ValueError('Training is incomplete; do not silently reuse a partial run')
        if training['best_pass']==0:
            status('complete_no_acceptable_improvement_keep_parent',training_report=str(run/'train/training_report.json'))
            return
        command('fresh_validation',[ROSA/'tools/validate_guarded_heading.py','--run',run,'--targets',args.targets])
        validation=json.loads((run/'fresh_validation.json').read_text())
        if not validation['passed']:
            status('complete_candidate_rejected_keep_parent',validation=validation)
            return
        deployment=run/'deployment';deployment.mkdir(exist_ok=True)
        parent=Path(training['parent']);checkpoint=Path(training['checkpoint'])
        if not (deployment/'rosa_guarded_fp32.onnx').exists():
            command('export_fp32',[ROSA/'tools/update_heading_onnx.py','--base-onnx',
                ROSA/'work_dirs/onnx_singleframe_e1/rosa_e1.onnx','--parent-checkpoint',parent,
                '--checkpoint',checkpoint,'--output',deployment/'rosa_guarded_fp32.onnx'])
        command('prepare_fast_onnx',[ROSA/'tools/optimize_fisheye_onnx.py','convert','--model',
            deployment/'rosa_guarded_fp32.onnx','--output',deployment,'--precision','backbone-fp16'])
        status('complete_guarded_native_candidate_fast_onnx_requires_parity_and_speed_checks',
            checkpoint=str(checkpoint),deployment=str(deployment),baseline_replaced=False)
    except Exception as exc:
        status('failed_keep_parent',error=repr(exc))
        raise


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Queue parking-heading training after context caching; validate before use."""
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
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--targets',type=Path,required=True)
    args=p.parse_args();run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    def status(phase,**kwargs):
        document=dict(phase=phase,updated_unix=time.time(),**kwargs)
        temporary=run/'workflow_status.partial.json';temporary.write_text(json.dumps(document,indent=2)+'\n')
        temporary.replace(run/'workflow_status.json');print(json.dumps(document),flush=True)
    def command(phase,values):
        status(phase,log=str(run/(phase+'.log')))
        with (run/(phase+'.log')).open('w') as log:
            subprocess.run([sys.executable,*map(str,values)],stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        while True:
            try:
                context=json.loads((run/'context/context_manifest.json').read_text())
                if context.get('status')=='complete':break
                status('preparing_gate_context',completed=context.get('completed'),total=context.get('total'),eta_seconds=context.get('eta_seconds'))
            except (FileNotFoundError,json.JSONDecodeError):pass
            time.sleep(20)
        report_path=run/'train/training_report.json'
        if not report_path.exists():
            command('training',[ROSA/'tools/finetune_gated_heading.py','--cache',args.cache.resolve(),
                '--context',run/'context','--targets',args.targets.resolve(),'--output',run/'train'])
        report=json.loads(report_path.read_text())
        if report.get('status')!='complete':raise ValueError('Training incomplete; do not reuse a partial candidate')
        if report['best_pass']==0:
            status('no_acceptable_heading_improvement_keep_parent');return
        command('fresh_validation',[ROSA/'tools/validate_guarded_heading.py','--run',run,'--targets',args.targets.resolve()])
        validation=json.loads((run/'fresh_validation.json').read_text())
        after=validation['new_heading']['all']['after']
        parking_target=after['axis_mean_deg']<=30 and after['axis_over_45deg_fraction']<=.2
        status('complete_validated_candidate' if validation['passed'] else 'candidate_rejected_keep_parent',
            validation_passed=validation['passed'],parking_target_met=bool(parking_target),
            new_axis_mean=after['axis_mean_deg'],old_mAP=validation['old_metrics']['mAP'],old_NDS=validation['old_metrics']['NDS'],
            checkpoint=report['checkpoint'],baseline_replaced=False,
            note='Parking target <=30 degrees mean, <=20% above45 degrees on same-scene pseudo-label holdout; not a guarantee on unseen parking scenes.')
    except Exception as exc:
        status('failed_keep_parent',error=repr(exc));raise


if __name__=='__main__':main()

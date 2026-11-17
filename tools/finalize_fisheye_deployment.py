#!/usr/bin/env python3
"""Wait for engine build, benchmark, validate old/new frames and full old metrics.

Original baseline is never overwritten. A validated candidate and a speed target
are distinct outcomes; a failed speed/accuracy check is reported, not hidden.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROSA=Path(__file__).resolve().parents[1]
ROOT=ROSA.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--engine-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,help='Separate release checks for each engine candidate')
    args=p.parse_args();run=args.run.resolve();engine_dir=args.engine_dir.resolve()
    model=run/'deployment/rosa_guarded_static.onnx';engine=engine_dir/'rosa.engine'
    output=args.output.resolve() if args.output else engine_dir/'release_checks'
    output.mkdir(parents=True,exist_ok=True)
    def status(phase,**values):
        doc=dict(phase=phase,updated_unix=time.time(),**values)
        temporary=output/'status.partial.json';temporary.write_text(json.dumps(doc,indent=2)+'\n')
        temporary.replace(output/'status.json');print(json.dumps(doc),flush=True)
    def command(name,command,expected):
        if expected.exists():
            return
        command=list(map(str,command));status(name,command=command,log=str(output/(name+'.log')))
        with (output/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,*command],stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        while True:
            try:
                build=json.loads((engine_dir/'build_report.json').read_text())
                if 'engine_sha256' in build and engine.is_file():
                    break
            except (FileNotFoundError,json.JSONDecodeError):
                pass
            status('waiting_for_TensorRT_compilation');time.sleep(30)
        command('benchmark',[ROSA/'tools/optimize_fisheye_onnx.py','benchmark','--model',model,'--engine',engine,
            '--output',output/'benchmark','--repeats','20','--reference-model',run/'deployment/rosa_guarded_fp32.onnx'],
            output/'benchmark/benchmark.json')
        dense=ROSA/'work_dirs/onnx_dense_old_bags_20261105'
        validation=[ROSA/'tools/validate_fisheye_deployment.py','--model',model,'--engine',engine,
            '--reference-model',run/'deployment/rosa_guarded_fp32.onnx',
            '--threshold-profile',run/'train/threshold_profile.json','--output',output/'real_frames']
        for name in ('16_09','17_09','30_09'):
            validation+=['--manifest',name+'='+str(dense/name/'samples.json')]
        validation+=['--manifest','25_09='+str(ROSA/'work_dirs/static_heading_adapt_20261105/onnx/check_samples.json')]
        command('real_frame_parity',validation,output/'real_frames/validation.json')
        parity=json.loads((output/'real_frames/validation.json').read_text())
        if not parity['passed']:
            status('rejected_accuracy_keep_baseline',parity=parity);return
        command('old_601_inference',[ROSA/'tools/infer_fisheye_onnx.py','--model',model,'--engine',engine,
            '--samples',ROSA/'work_dirs/onnx_singleframe_e1/validation_samples.json','--score-threshold','0',
            '--output',output/'old_601_predictions.json'],output/'old_601_predictions.json')
        command('old_601_metrics',[ROSA/'tools/evaluate_deployment_predictions.py',
            '--predictions',output/'old_601_predictions.json',
            '--info',ROOT/'data/Custom/combined_improved_detzero_rosa/rosa/fisheye3dod_infos_val.pkl',
            '--output',output/'old_601_metrics.json'],output/'old_601_metrics.json')
        metrics=json.loads((output/'old_601_metrics.json').read_text())['metrics']
        baseline=json.loads((run/'fresh_validation.json').read_text())['old_metrics']
        failed=[]
        if metrics['NDS']<baseline['NDS']-.002:failed.append('NDS')
        if metrics['mAP']<baseline['mAP']*.99:failed.append('mAP')
        for key,value in baseline.items():
            if key.endswith('_AOE') and metrics[key]>value+.03:failed.append(key)
        benchmark=json.loads((output/'benchmark/benchmark.json').read_text())
        latency=benchmark['latency']['mean_ms']
        report=dict(accuracy_passed=not failed,failed_gates=failed,engine=str(engine),model=str(model),
            threshold_profile=str(run/'train/threshold_profile.json'),checkpoint_sha256=build['checkpoint_sha256'],
            engine_sha256=build['engine_sha256'],onnx_sha256=build['onnx_sha256'],benchmark=benchmark,
            old_601_metrics=metrics,native_old_601_reference=baseline,real_frame_parity=parity,
            target_mean_ms=100.,meets_speed_target=latency<100.,
            speedup_vs_original_onnx=620.8407689/latency,speedup_vs_native_amp=213.2004101/latency,
            original_baseline_overwritten=False,
            note='Hardware-specific engine, four 800x800 cameras, batch 1. Numerical/detection guards do not solve the remaining new-scene heading errors.')
        (output/'release.json').write_text(json.dumps(report,indent=2)+'\n')
        status('validated_candidate' if not failed else 'rejected_old_metrics_keep_baseline',
            accuracy_passed=not failed,meets_speed_target=latency<100.,mean_ms=latency,release=str(output/'release.json'))
    except Exception as exc:
        status('failed_keep_baseline',error=repr(exc));raise


if __name__=='__main__':
    main()

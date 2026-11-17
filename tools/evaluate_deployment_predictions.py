#!/usr/bin/env python3
"""Official old-validation metrics from deployment JSON, without model inference."""
import argparse
import json
from pathlib import Path

from train_balanced_fisheye import np
import mmengine
from evaluate_heading_adaptation import official


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--info',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    payload=json.loads(args.predictions.read_text())
    if payload['box_origin']!='gravity_center' or payload['score_threshold']!=0 or payload['class_score_thresholds'] is not None:
        raise ValueError('Official comparison requires unfiltered gravity-center predictions (--score-threshold 0)')
    infos=mmengine.load(args.info)['data_list']
    if len(infos)!=len(payload['frames']):
        raise ValueError('Prediction and annotation frame counts differ')
    metrics=official(payload,infos)
    report=dict(metrics=metrics,frames=len(infos),checkpoint_sha256=payload['checkpoint_sha256'],
        source_predictions=str(args.predictions.resolve()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()

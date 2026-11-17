#!/usr/bin/env python3
"""Preflight, dense-frame memory probe and server training of four-bag ROSA.

Does not publish results into the older E007 experiment archive. Fresh optimizer,
live BN, full backbone/head, same class-wise NMS and dual-metric plateau hook.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

from train_balanced_fisheye import ROOT,ROSA,Config,batch_plan,runtime_options,merge_options,preflight,auto_batch,training_command
from finetune_heading_rows import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--init-checkpoint',type=Path,required=True)
    p.add_argument('--gpus',type=int,default=2);p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--effective-batch',type=int,default=8);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--master-port',type=int,default=29508);p.add_argument('--work-dir',type=Path)
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--probe-only',action='store_true')
    args=p.parse_args();data=args.data_root.resolve()
    os.environ['ROSA_DATA_ROOT']=str(data);os.environ['ROSA_INIT_CKPT']=str(args.init_checkpoint.resolve())
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    summary=json.loads((data/'summary.json').read_text())
    if {e['name'] for e in summary['sequences']}!={'16_09','17_09','25_09','30_09'}:
        raise ValueError('Expected all four September bags')
    if summary['split'].get('preserved_validation_frames')!=601:
        raise ValueError('Expected the protected old601 holdout; review dataset provenance')
    config_path=ROSA/'configs/fisheye3dod/rosa_vov99_fourbag_85_15_24e.py'
    config=Config.fromfile(config_path);plan=batch_plan(args.gpus,args.batch_size,args.effective_batch)
    options=runtime_options(config,plan,args.workers);merge_options(config,options)
    work=(args.work_dir or Path(config.work_dir)).resolve()
    if not args.preflight_only and (work/'plan.json').exists():raise FileExistsError('Use a new work directory')
    dense=preflight(config,work)
    if args.preflight_only:return
    probe=SimpleNamespace(**vars(args),cfg_options=[],fast=False,batch_candidates=[args.batch_size],
        probe_warmup=8,probe_iterations=24,memory_margin=.02)
    auto_batch(probe,config,config_path,work,dense)
    if args.probe_only:return
    command=training_command(probe,config_path,work,options)
    report=dict(experiment='E011_fourbag_full_model_85_15',command=command,batch_plan=plan,
        initialization=str(args.init_checkpoint.resolve()),initialization_sha256=sha(args.init_checkpoint),
        data_root=str(data),train_sha256=sha(data/'rosa/fisheye3dod_infos_train.pkl'),
        val_sha256=sha(data/'rosa/fisheye3dod_infos_val.pkl'),dataset_summary_sha256=sha(data/'summary.json'),
        source_sha256={str(config_path.relative_to(ROOT)):sha(config_path),str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__))},
        full_model_training=True,temporal_slots=0,stopping_policy=dict(patience=4,min_delta=1e-4,max_epochs=24),
        note='Evaluate the initializer on this 674-frame holdout before comparing new scores; old601 aggregate is not the same benchmark.')
    (work/'plan.json').write_text(json.dumps(report,indent=2)+'\n')
    print('$ '+shlex.join(command),flush=True)
    subprocess.run(command,cwd=ROOT,env=os.environ.copy(),check=True)


if __name__=='__main__':main()

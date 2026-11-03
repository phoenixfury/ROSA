#!/usr/bin/env python3
"""Preflight/stress-test and train single-frame ROSA with dual-metric stopping."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

from train_balanced_fisheye import (
    ROOT, ROSA, Config, auto_batch, batch_plan, merge_options, preflight,
    runtime_options, training_command)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--effective-batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--master-port', type=int, default=29505)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--background', action='store_true',
                        help='Launch durably and write console.log / launch.json')
    args = parser.parse_args()
    os.environ['ROSA_DATA_ROOT'] = str(args.data_root.resolve())
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if args.init_checkpoint:
        os.environ['ROSA_INIT_CKPT'] = str(args.init_checkpoint.resolve())
    config_path = ROSA / 'configs/fisheye3dod/rosa_vov99_singleframe_refresh_24e.py'
    config = Config.fromfile(config_path)
    plan = batch_plan(args.gpus, args.batch_size, args.effective_batch)
    overrides = runtime_options(config, plan, args.workers)
    merge_options(config, overrides)
    work = (args.work_dir or Path(config.work_dir)).resolve()
    if not args.preflight_only and not args.resume and (work / 'plan.json').exists():
        raise FileExistsError('Existing run: use --resume or a new --work-dir')
    if args.resume and args.resume.resolve().parent != work:
        raise ValueError('--resume must be a checkpoint from this work directory')
    if args.workers < 0:
        raise ValueError('workers must be nonnegative')
    if args.background:
        launch_path = work / 'launch.json'
        if launch_path.exists():
            previous = json.loads(launch_path.read_text())
            try:
                os.kill(previous['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('A launcher is already running in this directory')
        work.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()),
                   *[value for value in sys.argv[1:] if value != '--background']]
        environment = os.environ.copy()
        environment['PYTHONUNBUFFERED'] = '1'
        with (work / 'console.log').open('a' if args.resume else 'w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        launch_path.write_text(json.dumps(dict(pid=process.pid, command=command,
            log=str(work / 'console.log')), indent=2) + '\n')
        print(f'Launched PID {process.pid}; log: {work / "console.log"}', flush=True)
        return
    assert config.model.head.get('temporal_enabled', False) is False
    assert config.model.head.instance_bank.type == 'InstanceBank'
    assert config.model.head.instance_bank.num_temp_instances == 0
    assert config.train_dataloader.dataset.dataset.type == 'Fisheye3DODSparse4DDataset'
    assert config.train_dataloader.sampler.type == 'DefaultSampler'
    dense = preflight(config, work)
    if args.preflight_only:
        return
    probe_args = SimpleNamespace(**vars(args), cfg_options=[], fast=False,
        batch_candidates=[args.batch_size], probe_warmup=8, probe_iterations=24,
        memory_margin=.02)
    # Includes real accumulated optimizer steps on the 64 most crowded frames,
    # not just a small forward/backward pass without optimizer state allocation.
    auto_batch(probe_args, config, config_path, work, dense)
    if args.probe_only:
        return
    command = training_command(probe_args, config_path, work, overrides)
    if args.resume:
        position = command.index('--cfg-options')
        command[position:position] = ['--resume', str(args.resume.resolve())]
    sources = [config_path, Path(__file__).resolve(),
               ROSA / 'rosa/integrations/fisheye3dod/singleframe_plateau.py']
    info = dict(experiment='E007_singleframe_refresh', command=command,
        initialization=str(Path(config.load_from).resolve()),
        initialization_sha256=sha(config.load_from),
        data_root=str(args.data_root.resolve()),
        train_sha256=sha(args.data_root / 'rosa/fisheye3dod_infos_train.pkl'),
        validation_sha256=sha(args.data_root / 'rosa/fisheye3dod_infos_val.pkl'),
        source_sha256={str(path.relative_to(ROOT)): sha(path) for path in sources},
        batch_plan=plan, stopping_policy=dict(patience=4, min_delta=1e-4, max_epochs=24),
        temporal=False, full_model_training=True,
        baseline_current_601_reference=dict(mAP=.1021, NDS=.2322),
        caveat='DetZero/Kalman/camera-hold pseudo-labels; initializer may have old-split exposure.')
    target = work / ('resume_plan.json' if args.resume else 'plan.json')
    target.write_text(json.dumps(info, indent=2) + '\n')
    print('$ ' + shlex.join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
    if result.returncode:
        (work / 'failure.json').write_text(json.dumps(dict(returncode=result.returncode,
            message='Training failed; best/periodic checkpoints were not deleted.')) + '\n')
        raise subprocess.CalledProcessError(result.returncode, command)
    # Final metrics + report are written by the stop hook before training exits.
    archive = [sys.executable, str(ROSA / 'tools/archive_rosa_experiments.py'),
               '--only', 'E007_singleframe_refresh', '--verify-patches']
    subprocess.run(archive, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()

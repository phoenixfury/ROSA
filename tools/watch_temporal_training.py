#!/usr/bin/env python3
"""Show the current temporal ROSA ETA, losses and last validation metrics."""
import argparse
import json
from pathlib import Path
import re
import time


def snapshot(work_dir):
    log = work_dir/'launch.log'
    if not log.is_file():
        candidates = sorted(work_dir.glob('*/*.log'))
        if not candidates:
            return f'Waiting for training logs under {work_dir}'
        log = candidates[-1]
    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size-2_000_000))
        text = stream.read().decode(errors='replace')
    train = [line for line in text.splitlines() if 'Epoch(train)' in line]
    validation = [line for line in text.splitlines() if 'Epoch(val)' in line and 'fisheye_bev/mAP:' in line]
    output = [f'ROSA temporal training: {work_dir}']
    if train:
        line = train[-1]
        progress = re.search(r'Epoch\(train\)\s*\[(\d+)\]\[\s*(\d+)/(\d+)\]', line)
        if progress:
            epoch, step, total = map(int, progress.groups())
            preflight = work_dir/'temporal_preflight.json'
            epochs = json.loads(preflight.read_text()).get('epochs') if preflight.is_file() else None
            epoch_text = f'{epoch}/{epochs}' if epochs else str(epoch)
            output.append(f'Epoch {epoch_text} | iteration {step}/{total} ({100*step/total:.1f}%)')
        for label, pattern in (
                ('Remaining ETA', r'eta:\s*(.*?)\s+time:'),
                ('Loss', r'\bloss:\s*([^\s]+)'),
                ('Temporal cache fraction', r'temporal_cache_fraction:\s*([^\s]+)'),
                ('GPU memory (MiB)', r'\bmemory:\s*([^\s]+)'),
                ('Seconds/iteration', r'\btime:\s*([^\s]+)')):
            match = re.search(pattern, line)
            if match:
                output.append(f'{label}: {match.group(1)}')
    else:
        output.append('Initializing; ETA appears after the first 10 iterations.')
    if validation:
        for name in ('mAP', 'NDS', 'mATE', 'mASE', 'mAOE',
                     'Car_AP', 'Van_AP', 'Truck_AP', 'Bus_AP', 'Pedestrian_AP', 'Cyclist_AP'):
            match = re.search(r'fisheye_bev/'+name+r':\s*([^\s]+)', validation[-1])
            if match:
                output.append(f'Last validation {name}: {match.group(1)}')
    else:
        # Later epochs can push the old validation line beyond the tail.
        scalars = sorted(work_dir.glob('*/vis_data/scalars.json'))
        latest = None
        for path in scalars:
            with path.open() as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if 'fisheye_bev/mAP' in row:
                        latest = row
        if latest:
            output.append(f'Last validation mAP: {latest["fisheye_bev/mAP"]:.4f}')
            output.append(f'Last validation NDS: {latest["fisheye_bev/NDS"]:.4f}')
        else:
            output.append('Validation mAP/NDS: pending the first completed epoch.')
    errors = [line for line in text.splitlines() if 'Error:' in line or 'OutOfMemoryError' in line]
    if errors:
        output.append('ERROR: '+errors[-1])
    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work_dir', nargs='?', type=Path, default=(
        Path(__file__).resolve().parents[1]/'work_dirs/rosa_vov99_temporal_12e'))
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=5.)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('interval must be positive')
    while True:
        if args.watch:
            print('\033[2J\033[H', end='')
        print(snapshot(args.work_dir), flush=True)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()

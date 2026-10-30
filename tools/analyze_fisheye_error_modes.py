#!/usr/bin/env python3
"""Read-only localization/orientation diagnostics on saved validation outputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

from compare_retrained_fisheye import CLASSES, load_frames, np, read_infos, select


def summarize(values):
    array = np.asarray(values, dtype=float)
    return dict(n=len(array), mean=float(array.mean()), median=float(np.median(array)),
                p90=float(np.percentile(array, 90))) if len(array) else dict(n=0)


def errors(frames, threshold):
    rows, classes = [], defaultdict(list)
    recall_bins = defaultdict(Counter)
    for f in frames:
        used = set()
        for box in f['gt_boxes']:
            recall_bins[str(min(40, int(np.linalg.norm(box[:2]) // 10) * 10))]['gt'] += 1
        for i in select(f, threshold=threshold):
            distance = np.linalg.norm(f['gt_boxes'][:, :2] - f['boxes'][i, :2], axis=1)
            candidates = [j for j in np.flatnonzero(
                (f['gt_names'] == CLASSES[f['labels'][i]]) & (distance < 2.)) if j not in used]
            if not candidates:
                continue
            j = min(candidates, key=lambda j: distance[j])
            used.add(j)
            gt, pred = f['gt_boxes'][j], f['boxes'][i]
            diff = pred[:3] - gt[:3]
            radial = float(np.dot(diff[:2], gt[:2] / max(np.linalg.norm(gt[:2]), 1e-6)))
            yaw = abs(float(np.arctan2(np.sin(pred[6] - gt[6]), np.cos(pred[6] - gt[6]))))
            row = dict(center=float(distance[j]), radial=abs(radial),
                       tangential=float(np.sqrt(max(0., distance[j]**2 - radial**2))),
                       z=abs(float(diff[2])), yaw_deg=float(np.degrees(yaw)),
                       axis_yaw_deg=float(np.degrees(min(yaw, np.pi-yaw))))
            rows.append(row)
            classes[str(f['gt_names'][j])].append(row)
            recall_bins[str(min(40, int(np.linalg.norm(gt[:2]) // 10) * 10))]['tp'] += 1
    def finish(items):
        return dict(metrics={key: summarize([r[key] for r in items]) for key in
                    ('center', 'radial', 'tangential', 'z', 'yaw_deg', 'axis_yaw_deg')},
                    heading_over_90=sum(r['yaw_deg'] > 90 for r in items),
                    heading_over_150=sum(r['yaw_deg'] > 150 for r in items))
    return dict(total=finish(rows), per_class={k: finish(v) for k,v in classes.items()},
                recall_by_gt_range={k: dict(v, recall=v['tp']/max(v['gt'],1))
                                    for k,v in recall_bins.items()})


def projected_sizes(data_root, infos):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools/custom_data'))
    from render_three_model_fisheye_lidar_comparison import OcamCamera, project_box
    ocam = OcamCamera(filename=str(data_root / 'calib_results.txt'), fov=220.)
    measurements = defaultdict(list)
    missing = Counter()
    for sample in infos:
        for box, name in zip(sample['annos']['gt_boxes'], sample['annos']['gt_names']):
            if np.linalg.norm(box[:2]) >= dict(zip(CLASSES, (48,48,48,48,36,42)))[str(name)]:
                continue
            sizes = []
            for camera in sample['cam_info']['cam_fisheye'].values():
                pixels = project_box(np.asarray(box), camera, ocam)
                valid = (np.isfinite(pixels).all(1) & (pixels[:,0]>=0) & (pixels[:,0]<800) &
                         (pixels[:,1]>=0) & (pixels[:,1]<800))
                if valid.sum() >= 4:
                    width, height = np.ptp(pixels[valid], axis=0)
                    sizes.append((float(width),float(height)))
            if sizes:
                width,height=max(sizes,key=lambda wh:wh[0]*wh[1])
                measurements[str(name)].append((width,height,len(sizes)))
            else:
                missing[str(name)] += 1
    return dict(best_view={name: dict(width=summarize([r[0] for r in rows]),
                height=summarize([r[1] for r in rows]),
                fraction_height_below_16=float(np.mean([r[1]<16 for r in rows])),
                fraction_height_below_32=float(np.mean([r[1]<32 for r in rows])),
                views=summarize([r[2] for r in rows])) for name,rows in measurements.items()},
                no_view_with_four_valid_corners=dict(missing),
                caveat='Projected GT cuboids, not measured visible silhouettes; partial corners are possible')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    comparison=json.loads(args.comparison.read_text())
    infos=read_infos(args.data_root/'rosa/fisheye3dod_infos_val.pkl')
    result=dict(projected_gt_sizes=projected_sizes(args.data_root,infos),models={})
    for name,record in comparison['models'].items():
        print(name,flush=True)
        frames,_=load_frames(Path(record['directory']),infos,with_iou=False)
        result['models'][name]={str(t):errors(frames,t) for t in (.05,.1,.2)}
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()

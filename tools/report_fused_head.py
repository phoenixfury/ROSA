#!/usr/bin/env python3
"""Summarize E014 without hiding failed guards or selecting a deployment default."""
import argparse
import json
from pathlib import Path
import sys

ROSA=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROSA),str(ROSA/'work_dirs/onnx_deps_py38')]
import numpy as np
from benchmark_fisheye_backends import sha


def final_guard(d):
    p=d['retained_fp32_parity']
    return bool(d['baseline_retention']>=.99 and d['candidate_match_fraction']>=.99
        and p['class_mismatches']==0 and d['final_matched_max_box_error_m']<=.1
        and d['final_matched_max_yaw_error_degrees']<=10
        and d['final_matched_yaw_p95_degrees']<=.5
        and d['final_matched_max_score_error']<=.005)


def fusion_delta(baseline,candidate):
    a=json.loads((baseline/'cases.json').read_text())
    b=json.loads((candidate/'cases.json').read_text())
    if a!=b:raise ValueError('Fusion comparison must use identical cases')
    maxima=dict(box_m=0.,heading_deg=0.,score=0.,retained_class_changes=0)
    for source in sorted((baseline/'candidate_raw_outputs').glob('*.npz')):
        with np.load(source) as a,np.load(candidate/'candidate_raw_outputs'/source.name) as b:
            keep=(a['scores']>=.1)|(b['scores']>=.1)
            if not keep.any():continue
            maxima['box_m']=max(maxima['box_m'],float(np.abs(a['boxes'][...,:6]-b['boxes'][...,:6])[keep].max()))
            delta=(a['boxes'][...,6]-b['boxes'][...,6]+np.pi)%(2*np.pi)-np.pi
            maxima['heading_deg']=max(maxima['heading_deg'],float(np.abs(delta[keep]).max()*180/np.pi))
            maxima['score']=max(maxima['score'],float(np.abs(a['scores']-b['scores'])[keep].max()))
            maxima['retained_class_changes']+=int((a['labels'][keep]!=b['labels'][keep]).sum())
    return maxima


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROSA/'work_dirs/epoch6_fused_head_20261108')
    parser.add_argument('--reference-root',type=Path,default=ROSA/'work_dirs/epoch6_deployment_20261107')
    args=parser.parse_args();root=args.root.resolve();reference=args.reference_root.resolve()
    tags=('baseline','sampling','sampling_attention','all','all_nhwc_release')
    records={}
    for tag in tags:
        model=reference/'gpu_encoder_fp16/backbone-fp16.onnx' if tag=='baseline' else root/tag/'model.onnx'
        check_dir=reference/'gpu_fp16_graph_check_20' if tag=='baseline' else root/(tag+'_check_20')
        check=json.loads((check_dir/'validation.json').read_text())
        metadata=json.loads(model.with_suffix('.json').read_text())
        if check['onnx_sha256']!=sha(model) or metadata['onnx_sha256']!=sha(model):raise ValueError('Model checksum mismatch')
        bench=json.loads((root/(tag+'_bench_20')/'benchmark.json').read_text())
        if bench['onnx_sha256']!=sha(model):raise ValueError('Benchmark model differs')
        p=check['retained_fp32_parity']
        flags=['--iobinding','--cuda-graph','--prefetch','--threads','2']
        custom_ops=metadata.get('custom_ops')
        if custom_ops:flags+=['--custom-op-library',custom_ops['library']]
        records[tag]=dict(model=str(model),onnx_sha256=sha(model),custom_ops=custom_ops,runtime_flags=flags,
            strict_guard_passed=check['passed'],heading_limit_passed=bool(p['max_yaw_error_rad']<=np.deg2rad(10)),
            final_output_guard_passed=final_guard(check),
            max_heading_difference_deg=p['max_yaw_error_rad']*180/np.pi,
            retained_heading_p95_deg=check['retained_yaw_p95_degrees'],
            retained_box_max_m=p['max_box_error_m'],retained_score_max=p['max_score_error'],
            retained_class_changes=p['class_mismatches'],final_score_max=check['final_matched_max_score_error'],
            original_detections=check['original_detections'],matched_detections=check['matched_detections'],
            final_detections=check['engine_detections'],benchmark=bench,
            fusion_only_delta=None if tag=='baseline' else fusion_delta(reference/'gpu_fp16_graph_check_20',check_dir))
    document=dict(experiment='E014_fused_head',status='USER_CHOICE_PENDING_NO_DEFAULT_CHANGED',
        checkpoint_sha256=check['checkpoint_sha256'],acceptance_bags=['16_09','17_09','30_09'],excluded_acceptance_bag='25_09',
        heading_limit_deg=10,score_threshold=.1,nms_iou=.5,variants=records,
        prior_choices=str(ROSA/'deployment/epoch6_choices_first3.json'),
        note='Heading limit alone was relaxed. Raw retained score guard remains0.005; failures are not hidden. '
             'Paired numerical comparison, not independent GT mAP/NDS validation. No default selected.')
    (root/'choices.json').write_text(json.dumps(document,indent=2)+'\n')
    compact={k:v for k,v in document.items() if k!='variants'}
    compact['variants']={k:{x:y for x,y in v.items() if x!='benchmark'} for k,v in records.items()}
    (ROSA/'deployment/epoch6_fused_choices_first3.json').write_text(json.dumps(compact,indent=2)+'\n')
    lines=['# E014: fused projection / sampling / point attention','',
        'Analysis, implementation and independent ablations completed. No checkpoint, architecture, '
        'dataset, inference threshold or previous deployment default was changed. All prior E013 alternatives remain available.','',
        '## Real-frame results','',
        'RTX3060 Laptop, 6 GiB; same 60 distinct four-camera 800x800 frames (20 each September16/17/30), '
        'batch1, three warmups, CUDA graphs, I/O binding, CPU lookahead. Mean/p95 exclude cold session setup and rendering. '
        'Peak memory is whole-device NVML, not only model allocations. September25 excluded.','',
        '| Variant | Network mean ms | End-to-end mean / p95 ms | Peak MiB | <350ms | Max heading delta | Final matched | Final guard | Strict raw guard |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for tag,v in records.items():
        b=v['benchmark'];t=b['timings'];e=t['end_to_end']
        lines.append(f"| {tag} | {t['network']['mean_ms']:.1f} | {e['mean_ms']:.1f} / {e['p95_ms']:.1f} | {b['peak_gpu_device_mib']:.0f} | {b['end_to_end_frames_under350_ms']}/60 | {v['max_heading_difference_deg']:.4f}deg | {v['matched_detections']}/{v['original_detections']} | {'PASS' if v['final_output_guard_passed'] else 'FAIL'} | {'PASS' if v['strict_guard_passed'] else 'FAIL'} |")
    lines+=['','`all` uses direct NCHW sampling. `all_nhwc_release` packs four FPN maps once '
        'into channel-last layout for coalesced reads, shared across six stages. Other head math stays FP32. '
        'Point attention is the thirteen-point learned reliability/masked-softmax/value reduction, '
        'not a replacement of ROSA camera/geometry attention with generic FlashAttention.','',
        '## Accuracy interpretation','',
        'All heading differences are directed wrapped differences versus original epoch6 FP32, not absolute '
        'ground-truth heading error. Uniform 0.1 thresholds are untuned and unchanged. Final guard also '
        'requires score<=0.005, box<=0.1m, p95heading<=0.5deg, class agreement and>=99%matching/retention. '
        'The original strict guard additionally tests all retained pre-NMS queries. Its score failure '
        'already existed in the half-precision encoder: E013 traced the maximum to an NMS-suppressed '
        'duplicate. These are final-output-passing experimental choices, not strict FP32-equivalent models.','',
        '| Variant | Fusion-only heading max deg | Fusion-only box max m | Fusion-only score max | Class changes |',
        '|---|---:|---:|---:|---:|']
    for tag,v in records.items():
        if not v['fusion_only_delta']:continue
        d=v['fusion_only_delta']
        lines.append(f"| {tag} | {d['heading_deg']:.6f} | {d['box_m']:.7f} | {d['score']:.7f} | {d['retained_class_changes']} |")
    lines+=['','Fusion-only comparison holds the FP16 encoder fixed and compares identical score-retained '
        'query indices against the existing GPU-optimized portable model. It separates custom-kernel '
        'changes from the inherited encoder-precision differences. Full FP32 and raw-output reports '
        'are preserved; no failed guard was relabeled as passing.','',
        '## Bottlenecks and next choices','']
    for name in ('profile_baseline','profile_all','profile_all_nhwc_release'):
        path=root/(name+'.json')
        if path.is_file():
            d=json.loads(path.read_text());t=d['timings']
            lines.append(f"- {name}: encoder/FPN {t['encoder']['mean_ms']:.1f}ms; six-stage head {t['head']['mean_ms']:.1f}ms (15 repeated-input synchronized measurements).")
    lines+=['','Synchronized split-component timings are attribution assays with different buffer lifetimes, '
        'extra barriers and repeated input; do not substitute their sum for the production distinct-frame '
        'benchmark. The older profile_all label incorrectly said FP32encoder; its source graph was '
        'backbone-fp16. The runtime precision label is now derived from source metadata.','',
        'Sampling fusion supplies most of the improvement: no four full sampled-feature/transpose/weighted '
        'intermediates per stage. Projection removes thousands of validity/geometry graph operations; '
        'short point attention gives a smaller incremental gain. Encoder convolutions/FPN are now the '
        'larger component. CPU JPEG work is overlapped; postprocessing and output transfer are sub-ms '
        'and are not the principal target.','',
        'Potential follow-ups are combining the fused head with the existing TensorRT encoder '
        '(must revalidate its larger score errors), then fusing camera-level softmax/RFL after profiling. '
        'INT8, dropping queries/layers/cameras or reducing image resolution are not part of E014. '
        'No claim is made that generic FlashAttention or a full TensorRT parser can optimize the '
        'custom geometric attention unchanged. No TensorRT head plugin was implemented.','',
        '## Safety, reproducibility, limitations','',
        '- No packages installed. The CUDA build uses the existing compatible conda host compiler. '
        'Library requires ORT C API19+, CUDA, and registration before session creation; no libtorch dependency.',
        '- Generated native projection/point kernels, source hashes, library hashes, fixed tensor shapes '
        'and model hashes are recorded. CUDA compute stream comes from ORT; kernels perform no host '
        'copies, allocation or device synchronization. No fast-math/FMA contraction.',
        '- Kernel checks cover interpolation borders, unsupported-shape rejection, all-invalid '
        'attention, calibrated projection and visibility. Focused CPU regression tests are separate.',
        '- The focused regression suite passed31tests. Final maintained graph-rewrite code reproduced '
        'the validated channel-last ONNX byte-for-byte;506 audited parameter arrays were unchanged. '
        'The release library build record matches the final maintained CUDA source.',
        '- One early NHWC benchmark failed while serializing its report due to a metadata variable '
        'name; it produced no accepted benchmark result. The corrected release was rechecked and '
        'remeasured. Earlier builds and failed invocations remain, not silently overwritten.',
        '- Local RTX3060 timings are not server3080Ti timings. Rebuild/benchmark on the target server. '
        'TensorRT needs a separate IPluginV3 wrapper; the ORT CUDA library is not a TensorRT plugin.',
        '- See docs/fused_head_deployment.md for architecture analysis, official sources, build, '
        'Python/C++ use, replay restrictions and reproduction commands. choices.json intentionally '
        'does not name a selected default. E013 portable models and frozen experiment patch were not changed.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k:dict(mean_ms=v['benchmark']['timings']['end_to_end']['mean_ms'],
        final_guard=v['final_output_guard_passed'],strict_guard=v['strict_guard_passed']) for k,v in records.items()},indent=2))


if __name__=='__main__':main()

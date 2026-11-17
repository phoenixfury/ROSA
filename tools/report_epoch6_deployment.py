#!/usr/bin/env python3
"""Collect deployment alternatives without promoting or deleting any candidate."""
import argparse
import json
from pathlib import Path

from benchmark_fisheye_backends import sha
from study_epoch6_deployment import ROSA,variants


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROSA/'work_dirs/epoch6_deployment_20261107')
    args=parser.parse_args();root=args.root.resolve()
    original=json.loads((root/'fp32/rosa_epoch6.json').read_text())
    records={}
    for tag,(model,flags) in variants(root).items():
        if not model.is_file():continue
        validation=root/f'{tag}_check_20/validation.json'
        if tag in ('gpu_fp16','gpu_fp16_graph_prefetch'):validation=root/'gpu_fp16_graph_check_20/validation.json'
        if tag=='trt_encoder_prefetch':validation=root/'trt_encoder_check_20/validation.json'
        check=json.loads(validation.read_text()) if validation.is_file() else None
        bench=next((root/f'{tag}_bench_{n}/benchmark.json' for n in (20,10)
                    if (root/f'{tag}_bench_{n}/benchmark.json').is_file()),None)
        benchmark=json.loads(bench.read_text()) if bench else None
        metadata=json.loads(model.with_suffix('.json').read_text())
        if check and check['onnx_sha256']!=metadata['onnx_sha256']:raise ValueError('Validation/model mismatch')
        if benchmark and benchmark['onnx_sha256']!=metadata['onnx_sha256']:raise ValueError('Timing/model mismatch')
        row=dict(model=str(model.relative_to(ROSA)),onnx_sha256=metadata['onnx_sha256'],
            model_mib=model.stat().st_size/1024**2,runtime_flags=flags,
            validation_file=str(validation) if check else None,benchmark_file=str(bench) if bench else None,
            benchmark=benchmark,strict_validation=check)
        if tag=='fp32':
            row.update(role='FP32 reference; three real-frame CPU PyTorch parity checks passed',heading_limit_passed=True,
                       strict_guard_passed=True,maximum_heading_difference_deg=0.,maximum_score_difference=0.)
        elif check:
            parity=check['retained_fp32_parity']
            row.update(heading_limit_passed=parity['max_yaw_error_rad']<=10*3.141592653589793/180,
                strict_guard_passed=check['passed'],maximum_heading_difference_deg=parity['max_yaw_error_rad']*180/3.141592653589793,
                maximum_score_difference=parity['max_score_error'],maximum_final_heading_difference_deg=check['final_matched_max_yaw_error_degrees'],
                maximum_final_score_difference=check['final_matched_max_score_error'])
            row['final_output_guard_passed']=bool(check['baseline_retention']>=.99 and check['candidate_match_fraction']>=.99
                and parity['class_mismatches']==0 and check['final_matched_max_box_error_m']<=.1
                and check['final_matched_max_yaw_error_degrees']<=10 and check['final_matched_yaw_p95_degrees']<=.5
                and check['final_matched_max_score_error']<=.005)
        records[tag]=row
    components={}
    for name in ('profile_gpu_fp16','profile_trt_encoder','profile_fp32'):
        path=root/(name+'.json')
        if path.is_file():components[name]=json.loads(path.read_text())
    native_path=root/'native_amp_benchmark.json'
    native=json.loads(native_path.read_text()) if native_path.is_file() else None
    result=dict(status='USER_CHOICE_PENDING_NO_DEFAULT_CHANGED',checkpoint=original['checkpoint'],
        checkpoint_sha256=original['checkpoint_sha256'],maximum_heading_difference_deg=10,
        heading_convention='Directed wrapped yaw difference versus FP32; 180-degree flips do not count as equivalent.',
        acceptance_bags=['16_09','17_09','30_09'],excluded_acceptance_bag='25_09',
        operating_threshold=.1,threshold_profile='Untuned uniform epoch6 profile, not E007 tuned cutoffs',
        reference_native_validation=str(root/'fp32/pytorch_validation.json'),variants=records,
        components=components,native_amp=native,
        note='Heading-pass and complete conservative retained-query guard-pass are deliberately separate. '
             'The latter also requires max box0.1m, max score0.005, no retained-query class changes, '
             'yaw p95<=0.5deg and >=99% final retention/matching. No failed guard is silently relaxed. '
             'All candidates remain available; no checkpoint/default replacement. Full detection AP/NDS not recomputed.')
    (root/'choices.json').write_text(json.dumps(result,indent=2)+'\n')
    compact={k:v for k,v in result.items() if k not in ('components','native_amp')}
    compact['variants']={tag:{k:v for k,v in row.items() if k not in ('benchmark','strict_validation')} for tag,row in records.items()}
    (ROSA/'deployment/epoch6_choices_first3.json').write_text(json.dumps(compact,indent=2)+'\n')
    lines=['# Epoch6 fast deployment alternatives','',
        'No deployment default was changed. All model files and both passing/failing checks are retained for user choice.',
        'Checkpoint: `combined_4bags/best_fisheye_bev_mAP_epoch_6.pth`, SHA256 `'+original['checkpoint_sha256']+'`.',
        'Same four 800x800 BGR cameras, batch 1, 300 queries, six decoder stages, CLIDAR x-forward/y-left/z-up geometric-center boxes. No cameras/queries/layers were dropped.','',
        '## Measured alternatives','',
        'RTX3060 Laptop GPU (6 GiB), ORT1.19.2/CUDA. Distinct first-three-bag images, three warmups. Network includes host/device I/O. End-to-end includes input wait and rotated NMS; prefetch can overlap JPEG decode. Per-row frame counts are explicit; initialization/rendering excluded.','',
        '| Variant | Frames | Network mean ms | End-to-end mean / p95 ms | Peak device MiB | Max heading delta deg | Max retained score delta | Final-output guard | Strict raw guard |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for tag,row in records.items():
        b=row['benchmark'];t=b['timings'] if b else None
        angle=f'{row["maximum_heading_difference_deg"]:.3f}' if 'maximum_heading_difference_deg' in row else 'not checked'
        score=f'{row["maximum_score_difference"]:.5f}' if 'maximum_score_difference' in row else 'not checked'
        guard='PASS' if row.get('strict_guard_passed') else 'FAIL' if 'strict_guard_passed' in row else 'not checked'
        final_guard='PASS' if row.get('final_output_guard_passed') else 'FAIL' if 'final_output_guard_passed' in row else 'reference' if tag=='fp32' else 'not checked'
        lines.append(f'| {tag} | '+(f'{b["samples"]} | {t["network"]["mean_ms"]:.1f} | {t["end_to_end"]["mean_ms"]:.1f} / {t["end_to_end"]["p95_ms"]:.1f} | {b["peak_gpu_device_mib"]:.0f}' if b else '— | — | — | —')+f' | {angle} | {score} | {final_guard} | {guard} |')
    lines+=['','The user-approved heading limit is 10 degrees, not 10% of yaw or 10% AP loss. These are direct FP32 numerical differences, not absolute heading errors against annotations. Checks cover 60 real frames at score 0.1 on September 16/17/30 only; September 25 remains outside acceptance.',
        'The original conservative guard additionally limits retained pre-NMS score differences to0.005. Some half-precision variants pass the heading/box/class/final-retention conditions but fail that extra score guard. They remain explicit experimental choices, not silently marked fully accepted. `choices.json` includes final-output score errors separately.','',
        'The final-output guard is an additional, separately labeled post-NMS diagnostic with the same box/score/heading tolerances and retention requirements; it does not overwrite a failed raw guard. In the plain CUDA-graph FP16 case, the worst raw score difference is query207 of case46 (September30 token1790763686521885373):0.345419->0.351729. Both NMS results suppress that duplicate. All758 final detections match; maximum final score difference is0.003511.','',
        '## What the techniques change','',
        '- `fast_ort`: static simplification, FP16 encoder/FPN, unchanged FP32 detection head, persistent CUDA IO buffers; standard ONNX.',
        '- `gpu_fp32`/`gpu_fp16`: compiler-proven shape folding, GPU atan polynomial and TopK class selection; head stays FP32. CUDA replay reduces launch overhead, but extra GPU math can also make variants slower.',
        '- `_prefetch`: one next-frame CPU JPEG/calibration loader; GPU session calls stay serial. Effective latency uses consumer wait, not double-counted overlapped decode time.',
        '- `protected_graph`: stem and FPN remain FP32. `protected_tail_graph` also protects stage5; `protected_stage4_graph` instead protects the large stage4.',
        '- `trt_encoder`: TensorRT image encoder with FP32-protected stem/FPN, GPU-resident handoff to FP32 ORT head with CUDA replay. No user-defined CUDA/ONNX operator; engine is GPU/TensorRT-version-specific and must be rebuilt on the server.','',
        '## Bottleneck measurements','']
    if components:
        lines+=['Stage assay uses full CUDA synchronization after each component; these barriers perturb scheduling. It is not a replacement for the distinct-frame deployment benchmark.','',
            '| Assay | Encoder + FPN ms | Six-stage head ms | Input upload ms | Output download ms | NMS ms |',
            '|---|---:|---:|---:|---:|---:|']
        for tag,c in components.items():
            t=c['timings'];lines.append('| '+tag+' | '+' | '.join(f'{t[k]["mean_ms"]:.2f}' for k in ('encoder','head','input_upload','output_download','postprocess'))+' |')
        if 'profile_gpu_fp16' in components:
            t=components['profile_gpu_fp16']['timings']
            lines+=['',f'The remaining primary bottleneck is the six-stage FP32 geometry/attention/refinement head: about {t["head"]["mean_ms"]:.0f}ms versus {t["encoder"]["mean_ms"]:.0f}ms for encoder+FPN in this assay. More encoder-only acceleration has diminishing returns. CPU JPEG loading is about13-17ms and can overlap inference; host NMS is below1ms. Input/output transfers are also below1ms in the component assay.',
                'The next substantial improvement would target the head: fuse projection/sampling/attention kernels (potentially an ORT/TensorRT custom operator), or carefully test selective learned-linear mixed precision while keeping projection/state/yaw FP32. Fewer decoder stages/queries or smaller images are model/accuracy changes requiring separate validation, not transparent speed flags. These further head modifications were not applied.']
    else:lines+=['Component assays pending. Do not infer GPU kernel percentages from ORT host-node durations.']
    if native:
        lines+=['',f'Native fused PyTorch AMP same-input diagnostic: {native["network_with_host_io"]["mean_ms"]:.1f}ms network, {native["gpu_device_peak_mib"]:.0f}MiB whole-device peak. This is a separate same-input assay, not the distinct-image table.']
    lines+=['','The raw export contains26,724 nodes,1,776 Shape nodes,274 Range nodes,12 shape-only If branches and31 Atan nodes. Ordinary simplification leaves9,557 nodes; compiler-proven shape specialization leaves7,113 before GPU-math rewrites. Node count alone does not predict latency.',
        'The profiling trace contains host Node events, not CUPTI Kernel events: its apparent scope percentages are not GPU compute shares. NVIDIA clocks/power and cuDNN algorithm/layout choices also affect laptop timings; no3080Ti/server speed is claimed.',
        'Official details: [ORT profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html#gpu-profiling), [CUDA graph restrictions](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview).','',
        '## Accuracy and deployment limits','',
        '- Canonical FP32 export passed three real-frame PyTorch FP32 references, no class changes; maximum box difference below0.1mm. The learned camera-ray/point fusion remains intact.',
        '- No retuning, retraining, resolution reduction, guessed calibration values, distance filtering or AP/NDS claims were introduced. Old E007 thresholds and ONNX remain untouched.',
        '- The uniform0.1 profile is not newly optimized for precision/recall. Temporal slots and velocity export remain disabled.',
        '- A mean below350ms does not guarantee every frame or p95 below350ms. Use the longer60-frame results where present; all other timings are30-frame exploratory measurements.',
        '- FP16/FP32 ONNX variants use standard operators and the ordinary ORT CUDA provider. Host NumPy/OpenCV class-wise rotated NMS remains IoU0.5. C++ can implement the same preprocessing/NMS/buffer binding; TensorRT split prototype currently uses PyTorch for device-buffer allocation, not model inference.','',
        '## Reproduce and choose','',
        'All per-candidate commands, raw paired outputs, checksums and benchmark JSON are retained beside this report. `deployment/epoch6_choices_first3.json` lists model paths and runtime flags; it intentionally names no selected default.',
        'Example for the fast plain-ORT CUDA-replay/prefetch alternative (read its guard status above first):','',
        '```bash',
        'conda run --no-capture-output -n openmmlab python projects/ROSA/tools/infer_fisheye_onnx.py \\',
        '  --model projects/ROSA/work_dirs/epoch6_deployment_20261107/gpu_encoder_fp16/backbone-fp16.onnx \\',
        '  --iobinding --cuda-graph --prefetch --threads 2 --warmup 3 --score-threshold 0.1 \\',
        '  --samples projects/ROSA/work_dirs/epoch6_deployment_20261107/fp32/validation_samples.json \\',
        '  --max-frames 10 --output projects/ROSA/work_dirs/epoch6_fast_test.json',
        '```','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(variants=len(records),report=str(root/'REPORT.md')),indent=2))


if __name__=='__main__':main()

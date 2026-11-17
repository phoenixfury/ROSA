#!/usr/bin/env python3
"""Short same-input inference/memory measurement, one isolated backend per process.

No bag-wide inference, model updates or exports. GPU memory is sampled via NVML
on the whole device (including context/background allocations); Torch allocator
counters are separate and must not be mistaken for ORT memory measurements.
"""
import argparse
from contextlib import nullcontext
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

ROSA=Path(__file__).resolve().parents[1]
ROOT=ROSA.parents[1]
sys.path[:0]=[str(ROOT),str(ROSA),str(ROOT/'projects/Fisheye3DOD')]
deps=ROSA/'work_dirs/onnx_deps_py38'
if sys.version_info[:2]==(3,8) and deps.is_dir():
    sys.path.insert(0,str(deps))

import numpy as np
from deployment.runtime import INPUT_NAMES, create_session, load_npz, postprocess


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):
            digest.update(chunk)
    return digest.hexdigest()


class DeviceMemory:
    class Info(ctypes.Structure):
        _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]

    def __init__(self):
        self.library=ctypes.CDLL('libnvidia-ml.so.1')
        if self.library.nvmlInit_v2()!=0:
            raise RuntimeError('NVML initialization failed')
        self.handle=ctypes.c_void_p()
        if self.library.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle))!=0:
            raise RuntimeError('NVML device lookup failed')
        self.baseline=self.read()
        self.peak=self.baseline
        self.done=threading.Event()
        self.thread=threading.Thread(target=self.monitor,daemon=True)

    def read(self):
        info=self.Info()
        if self.library.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(info))!=0:
            raise RuntimeError('NVML memory lookup failed')
        return info.used

    def monitor(self):
        while not self.done.is_set():
            self.peak=max(self.peak,self.read())
            self.done.wait(.02)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self,*exc):
        self.done.set()
        self.thread.join()
        self.peak=max(self.peak,self.read())


def stats(values):
    values=np.asarray(values)
    return dict(mean_ms=float(values.mean()*1000),median_ms=float(np.median(values)*1000),
        p95_ms=float(np.percentile(values,95)*1000),fps_from_mean=float(1/values.mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('torch-fp32','torch-amp','onnx-fp32'),required=True)
    p.add_argument('--input',type=Path,default=ROSA/'work_dirs/onnx_singleframe_e1/example_0_input.npz')
    p.add_argument('--checkpoint',type=Path,default=ROSA/'work_dirs/singleframe_refresh_20261031/best_fisheye_bev_mAP_epoch_1.pth')
    p.add_argument('--model',type=Path,default=ROSA/'work_dirs/onnx_singleframe_e1/rosa_e1.onnx')
    p.add_argument('--config',type=Path,default=ROSA/'configs/fisheye3dod/rosa_vov99_singleframe_refresh_24e.py')
    p.add_argument('--threshold-profile',type=Path,
                   default=ROSA/'work_dirs/threshold_tuning_e1_20261103/balanced_profile.json')
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.repeats<1 or args.warmup<0:
        p.error('Invalid iteration count')
    if args.output.exists():
        raise FileExistsError(args.output)
    feed=load_npz(args.input)
    threshold_path=args.threshold_profile
    profile=json.loads(threshold_path.read_text())
    if profile.get('checkpoint_sha256')!=sha(args.checkpoint) or profile.get('nms_iou')!=.5:
        raise ValueError('Threshold profile must match the checkpoint and NMS0.5 contract')
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    measurements=[]
    totals=[]
    with DeviceMemory() as memory:
        start=time.perf_counter()
        if args.backend=='onnx-fp32':
            import onnxruntime as ort
            metadata=json.loads(args.model.with_suffix('.json').read_text())
            if sha(args.model)!=metadata['onnx_sha256'] or metadata['checkpoint_sha256']!=sha(args.checkpoint):
                raise ValueError('Model/checkpoint metadata mismatch')
            session=create_session(args.model,'cuda',2)
            version=ort.__version__
            def forward():
                return session.run(None,feed)
            parameter_count=None
            file_path=args.model
        else:
            from train_balanced_fisheye import Config, MODELS, init_default_scope
            from export_fisheye_onnx import ExportWrapper
            from mmengine.runner import load_checkpoint
            config=Config.fromfile(args.config)
            init_default_scope(config.default_scope)
            model=MODELS.build(config.model).eval()
            load_checkpoint(model,str(args.checkpoint),map_location='cpu',strict=True)
            model=model.cuda()
            model.head.instance_bank.reset()
            wrapper=ExportWrapper(model).eval()
            parameter_count=sum(p.numel() for p in model.parameters())
            version=torch.__version__
            file_path=args.checkpoint
            def forward():
                inputs=[torch.from_numpy(feed[name]).cuda() for name in INPUT_NAMES]
                context=torch.autocast('cuda',dtype=torch.float16) if args.backend=='torch-amp' else nullcontext()
                with torch.no_grad(),context:
                    outputs=wrapper(*inputs)
                return [value.float().cpu().numpy() if value.is_floating_point() else value.cpu().numpy() for value in outputs]
        load_seconds=time.perf_counter()-start
        for _ in range(args.warmup):
            forward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for i in range(args.repeats):
            torch.cuda.synchronize()
            start=time.perf_counter()
            raw=forward()
            torch.cuda.synchronize()
            network=time.perf_counter()-start
            prediction=postprocess(*raw,score_threshold=0.,nms_iou=.5,
                class_score_thresholds=profile['score_thresholds'])
            total=time.perf_counter()-start
            measurements.append(network)
            totals.append(total)
            print(args.backend,i+1,'network_ms',round(network*1000,2),'boxes',len(prediction['scores']),flush=True)
        steady_gpu=memory.read()
        allocator=dict(max_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                       max_reserved_mib=torch.cuda.max_memory_reserved()/1024**2) if args.backend.startswith('torch') else None
    rss_pages=int(Path('/proc/self/statm').read_text().split()[1])
    result=dict(backend=args.backend,framework_version=version,torch_version=torch.__version__,
        cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        gpu_total_mib=torch.cuda.get_device_properties(0).total_memory/1024**2,
        input=str(args.input.resolve()),input_sha256=sha(args.input),
        image_shape=list(feed['images'].shape),image_dtype=str(feed['images'].dtype),
        checkpoint_sha256=sha(args.checkpoint),model_file=str(file_path.resolve()),
        threshold_profile=str(threshold_path.resolve()),threshold_profile_sha256=sha(threshold_path),
        model_file_sha256=sha(file_path),model_file_mib=file_path.stat().st_size/1024**2,
        trainable_model_parameter_count=parameter_count,load_seconds=load_seconds,
        warmup=args.warmup,repeats=args.repeats,network_with_host_io=stats(measurements),
        network_plus_postprocess=stats(totals),seconds_per_frame=measurements,
        gpu_device_baseline_mib=memory.baseline/1024**2,gpu_device_peak_mib=memory.peak/1024**2,
        gpu_device_steady_mib=steady_gpu/1024**2,torch_allocator=allocator,
        host_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        host_steady_rss_mib=rss_pages*os.sysconf('SC_PAGE_SIZE')/1024**2,
        note='Isolated process; batch 1, all four 800x800 cameras. Same saved input, thresholds and CPU NMS. '
             'Network timing includes image normalization, CPU↔GPU copies and raw output conversion; excludes JPEG decode and rendering. '
             'GPU peak is whole-device NVML sampling including context/background, not a Torch allocator count. '
             'Load time excludes imports; host peak includes model/session construction. Short benchmark, not server performance.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Export the best single-frame ROSA checkpoint to a tensor-only ONNX graph."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

from train_balanced_fisheye import ROOT, ROSA, Config, MODELS, init_default_scope, np, torch

LOCAL_DEPS = ROSA / 'work_dirs/onnx_deps_py38'
if sys.version_info[:2] == (3, 8) and LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

from deployment.runtime import CAMERAS, COORDINATES, INPUT_NAMES, OUTPUT_NAMES, image_input
from deployment.torch_export import portable_cuda_ops
from rosa.integrations.fisheye3dod.calibration import convert_lidar2cam, convert_ocamcamera


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        head = model.head
        if getattr(head, 'temporal_enabled', False) or head.instance_bank.num_temp_instances != 0:
            raise ValueError('Only single-frame checkpoints are supported')
        if not head.decoder.single_label or not head.decoder.quality_before_topk:
            raise ValueError('Exporter requires single-label, quality-first decoder')

    def forward(self, images, lidar2cam, ocam_intrinsic, ocam_pol, image_wh, ocam_lens_fov_deg):
        normalized = self.model.data_preprocessor._process_img(images)
        inputs = dict(img=normalized, lidar2cam=lidar2cam, ocam_intrinsic=ocam_intrinsic,
                      ocam_pol=ocam_pol, image_wh=image_wh, ocam_lens_fov_deg=ocam_lens_fov_deg)
        raw = self.model._forward(inputs)
        head = self.model.head
        prediction = raw['prediction'][head.inference_output_decoder_index]
        logits = raw['classification'][head.inference_classification_decoder_index]
        quality = raw['quality'][head.inference_quality_decoder_index]
        boxes = torch.cat([prediction[..., :3], prediction[..., 3:6].exp(),
            torch.atan2(prediction[..., 6:7], prediction[..., 7:8])], dim=-1)
        scores, labels = (logits.sigmoid() * quality[..., 0:1].sigmoid()).max(dim=-1)
        return boxes, scores, labels


def calibration(info):
    cameras = [info['cam_info']['cam_fisheye'][name] for name in CAMERAS]
    ocam = [convert_ocamcamera(camera['cam2img']) for camera in cameras]
    return dict(lidar2cam=np.stack([convert_lidar2cam(camera['lidar2cam']) for camera in cameras])[None],
        ocam_intrinsic=np.stack([item.intrinsic for item in ocam])[None],
        ocam_pol=np.stack([item.polynomial for item in ocam])[None],
        image_wh=np.full((1, 4, 2), 800., dtype=np.float32),
        ocam_lens_fov_deg=np.array([[item.lens_fov_deg for item in ocam]], dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROSA / 'work_dirs/singleframe_refresh_20261031/best_fisheye_bev_mAP_epoch_1.pth')
    parser.add_argument('--config', type=Path, default=ROSA / 'configs/fisheye3dod/rosa_vov99_singleframe_refresh_24e.py')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/Custom/combined_improved_detzero_rosa')
    parser.add_argument('--output-dir', type=Path, default=ROSA / 'work_dirs/onnx_singleframe_e1')
    parser.add_argument('--example-indices', type=int, nargs='+', default=[0, 200, 400])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--force', action='store_true', help='Explicitly overwrite an existing generated ONNX model')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Prepare calibration/sample manifest for an existing ONNX model without exporting again')
    parser.add_argument('--references-only', action='store_true',
                        help='Regenerate real-frame PyTorch references for an existing, matching ONNX model')
    args = parser.parse_args()
    import mmengine
    import onnx
    from mmengine.runner import load_checkpoint

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / 'rosa_e1.onnx'
    if destination.exists() and not args.force and not args.prepare_only and not args.references_only:
        raise FileExistsError(f'{destination}: use a new output directory or --force')
    if args.references_only:
        previous = json.loads(destination.with_suffix('.json').read_text())
        if previous['checkpoint_sha256'] != sha(args.checkpoint) or previous['onnx_sha256'] != sha(destination):
            raise ValueError('Existing model/checkpoint does not match; re-export instead')
    os.environ['ROSA_DATA_ROOT'] = str(args.data_root.resolve())
    torch.set_num_threads(args.threads)
    config = Config.fromfile(args.config)
    init_default_scope(config.default_scope)
    original_fused = config.model.head.v4_decoder_cfg.get('use_fused_daf', False)
    # Keep the trained camera-RFL architecture. Only replace its CUDA readers
    # inside the scoped export context; never disable learned camera RFL.
    config.model.head.v4_decoder_cfg.streaming_camera_rfl_cuda = False
    info_file = args.data_root / 'rosa/fisheye3dod_infos_val.pkl'
    infos = mmengine.load(info_file)['data_list']
    if any(index < 0 or index >= len(infos) for index in args.example_indices):
        raise ValueError('Example indices outside validation manifest')
    calib_dir = args.output_dir / 'calibrations'
    calib_dir.mkdir(exist_ok=True)
    records = []
    for index, info in enumerate(infos):
        values = calibration(info)
        fingerprint = hashlib.sha256(b''.join(values[key].tobytes() for key in INPUT_NAMES[1:])).hexdigest()
        path = calib_dir / (fingerprint + '.npz')
        if not path.exists():
            np.savez(path, **values)
        if index == 0:
            np.savez(args.output_dir / 'calibration.npz', **values)
        records.append(dict(token=str(info['token']), index=index,
            sequence=info.get('metainfo', {}).get('sequence_id'),
            image_paths=[str((args.data_root / info['cam_info']['cam_fisheye'][name]['cam_path']).resolve())
                         for name in CAMERAS],
            image_relative_paths=[(args.data_root / info['cam_info']['cam_fisheye'][name]['cam_path']).resolve()
                                  .relative_to(args.data_root.resolve()).as_posix() for name in CAMERAS],
            calibration=str(path.relative_to(args.output_dir))))
    write_json(args.output_dir / 'validation_samples.json', dict(samples=records,
               validation_sha256=sha(info_file), camera_order=CAMERAS,
               data_root=str(args.data_root.resolve())))
    if args.prepare_only:
        print('Prepared sample/calibration manifest; model/checkpoint unchanged.', flush=True)
        return
    model = MODELS.build(config.model).cpu().eval()
    load_checkpoint(model, str(args.checkpoint.resolve()), map_location='cpu', strict=True)
    model.head.instance_bank.reset()
    wrapper = ExportWrapper(model).eval()
    # The legacy exporter supports this static B=1, 4x800x800 contract. Disable
    # MHA's fused native fastpath only locally; export standard attention ops.
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    # PyTorch 2.4 lacks a legacy symbolic for this simple unit conversion.
    # Emit an ordinary ONNX Mul, not a custom runtime operator.
    def deg2rad_symbolic(graph, angle):
        return graph.op('Mul', angle, graph.op('Constant',
            value_t=torch.tensor(np.pi / 180., dtype=torch.float32)))
    def sinc_symbolic(graph, value):
        zero = graph.op('Constant', value_t=torch.tensor(0., dtype=torch.float32))
        one = graph.op('Constant', value_t=torch.tensor(1., dtype=torch.float32))
        argument = graph.op('Mul', value, graph.op('Constant',
            value_t=torch.tensor(np.pi, dtype=torch.float32)))
        at_zero = graph.op('Equal', value, zero)
        denominator = graph.op('Where', at_zero, one, argument)
        return graph.op('Where', at_zero, one, graph.op('Div', graph.op('Sin', argument), denominator))
    torch.onnx.register_custom_op_symbolic('aten::deg2rad', deg2rad_symbolic, 17)
    torch.onnx.register_custom_op_symbolic('aten::sinc', sinc_symbolic, 17)
    references = []
    try:
        with torch.no_grad(), portable_cuda_ops():
            for index in args.example_indices:
                record = records[index]
                feed = calibration(infos[index])
                feed['images'] = image_input(record['image_paths'])
                tensors = tuple(torch.from_numpy(feed[name]) for name in INPUT_NAMES)
                output = wrapper(*tensors)
                input_path = args.output_dir / f'example_{index}_input.npz'
                reference_path = args.output_dir / f'example_{index}_reference.npz'
                np.savez(input_path, **feed)
                np.savez(reference_path, **{name: value.detach().numpy()
                                            for name, value in zip(OUTPUT_NAMES, output)})
                references.append(dict(index=index, token=record['token'],
                    input=str(input_path), reference=str(reference_path)))
                print('Prepared PyTorch FP32 reference', index, flush=True)
                if index == args.example_indices[0]:
                    export_tensors = tensors
            if not args.references_only:
                print('Exporting ONNX on CPU; training weights are unchanged.', flush=True)
                torch.onnx.export(wrapper, export_tensors, str(destination), input_names=list(INPUT_NAMES),
                    output_names=list(OUTPUT_NAMES), opset_version=17, do_constant_folding=True,
                    training=torch.onnx.TrainingMode.EVAL)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)
        torch.onnx.unregister_custom_op_symbolic('aten::deg2rad', 17)
        torch.onnx.unregister_custom_op_symbolic('aten::sinc', 17)
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph)
    if any(node.domain not in ('', 'ai.onnx') for node in graph.graph.node):
        raise RuntimeError('Graph contains nonstandard/custom operators')
    metadata = dict(format_version=1, checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=sha(args.checkpoint), config=str(args.config.resolve()),
        validation_sha256=sha(info_file), onnx_sha256=sha(destination), opset=17,
        precision='FP32; compare with FP32 PyTorch, not bitwise with AMP evaluation',
        original_use_fused_daf=original_fused, export_use_fused_daf=original_fused,
        custom_op_policy='Scoped standard-ONNX bilinear readers; full learned camera RFL/attention retained; strict checkpoint loading',
        coordinates=COORDINATES, box_origin='gravity_center', camera_order=CAMERAS,
        inputs=[dict(name=item.name, shape=[dim.dim_value for dim in item.type.tensor_type.shape.dim])
                for item in graph.graph.input], outputs=list(OUTPUT_NAMES),
        image_contract='uint8 BGR [1,4,3,800,800]; RGB conversion/normalization embedded in graph',
        calibration_contract='Dynamic tensors in converted ROSA OCam [down,right,back] camera basis; output stays CLIDAR FLU',
        postprocessing=dict(single_label=True, quality_before_topk=True, nms_iou=.5,
                            nms='class-wise rotated BEV on host; not included in ONNX', max_distance=None),
        references=references, checked=True, node_count=len(graph.graph.node),
        export_source_sha256={str(path.relative_to(ROSA)): sha(path) for path in
            (Path(__file__).resolve(), ROSA / 'deployment/torch_export.py')},
        operator_types=sorted(set(node.op_type for node in graph.graph.node)))
    write_json(destination.with_suffix('.json'), metadata)
    print('Exported and ONNX-checked:', destination, flush=True)


if __name__ == '__main__':
    main()

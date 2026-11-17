#!/usr/bin/env python3
"""Cache frozen final-layer features for a heading-only checkpoint fine-tune.

Only the existing final trunk yaw row and final signed-face yaw row will be
optimized later. No image/position/classification network is trained here.
"""
import argparse
import gc
import hashlib
import json
import logging
from pathlib import Path
import time

from train_balanced_fisheye import ROOT, ROSA, Config, MODELS, init_default_scope, np, torch
import mmengine
from mmengine.runner import Runner, load_checkpoint
from mmengine.logging import MMLogger


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROSA/'configs/fisheye3dod/rosa_vov99_singleframe_refresh_24e.py')
    p.add_argument('--dataset', nargs=3, action='append', required=True, metavar=('NAME', 'ROOT', 'INFO'))
    p.add_argument('--replay-count', type=int, default=96)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reuse-manifest', type=Path, help='Reuse already frozen new/validation features from the same parent')
    p.add_argument('--resume', action='store_true', help='Resume this output only with matching source/checkpoint pins')
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    MMLogger.get_current_instance().setLevel(logging.WARNING)
    config = Config.fromfile(args.config)
    init_default_scope(config.default_scope)
    model = MODELS.build(config.model)
    model.init_weights()
    load_checkpoint(model, str(args.checkpoint), map_location='cpu', strict=True)
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    last = model.head.fisheye_box_decoder.layers[-1].refinement
    if (last.yaw_update_mode != 'rotation' or not last.separate_face_yaw_residual or
            last.face_yaw_gate is None or last.trunk[-1].weight.shape != (10, 256)):
        raise ValueError('Checkpoint does not have the expected separate final yaw refinement')
    active = {}

    def capture(module, values):
        feature, _, position, signed, _ = values
        ref = (feature+position if module.state_conditioned else feature).float()
        active['hidden'] = module.trunk[:-1](ref).detach().float().cpu().numpy().astype(np.float16)[0]
        active['faces'] = signed.flatten(-2).detach().float().cpu().numpy().astype(np.float16)[0]
        active['yaw_gate'] = module.update_gate(ref).sigmoid()[..., 2].detach().float().cpu().numpy()[0]
        active['face_gate'] = module.face_yaw_gate(ref).sigmoid()[..., 0].detach().float().cpu().numpy()[0]

    handle = last.register_forward_pre_hook(capture)
    metadata = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
        config=str(args.config.resolve()), cached_precision='native CUDA AMP; hidden/face features stored FP16',
        trainable_rows=['head.fisheye_box_decoder.layers.5.refinement.trunk.3.weight[6]',
                        'head.fisheye_box_decoder.layers.5.refinement.trunk.3.bias[6]',
                        'head.fisheye_box_decoder.layers.5.refinement.face_refinement.weight[3]'], datasets={})
    metadata['config_sha256'] = sha(args.config)
    if args.reuse_manifest:
        reused = json.loads(args.reuse_manifest.read_text())
        if reused['checkpoint_sha256'] != metadata['checkpoint_sha256'] or reused['config'] != metadata['config']:
            raise ValueError('Reused features belong to a different checkpoint/config')
        metadata['datasets'] = reused['datasets']
        metadata['reused_manifest_sha256'] = sha(args.reuse_manifest)
    manifest_path = args.output/'cache_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not args.resume or any(previous.get(k) != metadata.get(k) for k in ('checkpoint_sha256','config_sha256')):
            raise ValueError('Existing cache requires --resume and matching checkpoint/config')
        metadata['datasets'].update(previous['datasets'])
    manifest_path.write_text(json.dumps(metadata, indent=2)+'\n')
    for name, root_string, info_string in args.dataset:
        root, info_path = Path(root_string).resolve(), Path(info_string).resolve()
        document = mmengine.load(info_path)
        samples = document['data_list']
        indices = np.arange(len(samples))
        if name == 'replay':
            indices = np.unique(np.linspace(0, len(samples)-1, min(len(samples), args.replay_count)).astype(int))
            document['data_list'] = [samples[i] for i in indices]
        directory = args.output/name
        directory.mkdir(exist_ok=True)
        previous_source = metadata['datasets'].get(name)
        if previous_source and previous_source['source_info_sha256'] != sha(info_path):
            raise ValueError('Dataset annotation changed during caching')
        selected_info = directory/'selected_info.pkl'
        mmengine.dump(document, selected_info)
        cfg = config.test_dataloader.copy()
        cfg.batch_size = 1
        cfg.num_workers = 2
        cfg.persistent_workers = False
        cfg.dataset = dict(cfg.dataset, data_root=str(root), ann_file=str(selected_info),
                           test_mode=True, load_eval_anns=False)
        loader = Runner.build_dataloader(cfg)
        records = []
        metadata['datasets'][name] = dict(root=str(root), source_info=str(info_path),
            source_info_sha256=sha(info_path), selected_info=str(selected_info.resolve()), records=records)
        manifest_path.write_text(json.dumps(metadata, indent=2)+'\n')
        started = time.monotonic()
        for index, batch in enumerate(loader):
            path = directory/f'{index:05d}.npz'
            token = str(document['data_list'][index]['token'])
            if args.resume and path.exists():
                with np.load(path, allow_pickle=False) as existing:
                    if str(existing['token']) != token or existing['hidden'].shape != (300,256):
                        raise ValueError('Invalid existing cached token/features')
                records.append(dict(index=index, source_index=int(indices[index]), token=token,path=str(path.resolve())))
                continue
            # Recompute rather than reuse stale features from changed input/config.
            with torch.no_grad():
                data = model.data_preprocessor(batch, training=False)
                with torch.autocast('cuda', dtype=torch.float16):
                    raw = model._forward(data['inputs'], data.get('data_samples'))
                state = raw['prediction'][-1][0].float()
                box = torch.cat((state[:, :3], state[:, 3:6].exp(),
                                 torch.atan2(state[:, 6:7], state[:, 7:8])), dim=-1)
                scores, labels = (raw['classification'][-1].float().sigmoid() *
                                  raw['quality'][-1][..., :1].float().sigmoid()).max(dim=-1)
            np.savez_compressed(path, **active, boxes=box.cpu().numpy(),
                scores=scores[0].cpu().numpy(), labels=labels[0].cpu().numpy(), token=np.asarray(token))
            records.append(dict(index=index, source_index=int(indices[index]), token=token,
                                path=str(path.resolve())))
            if (index+1) % 50 == 0 or index+1 == len(loader):
                elapsed = time.monotonic()-started
                eta = elapsed/(index+1)*(len(loader)-index-1)
                manifest_path.write_text(json.dumps(metadata, indent=2)+'\n')
                print(f'{name}: cached {index+1}/{len(loader)} | ETA {eta/60:.1f} min', flush=True)
        metadata['datasets'][name] = dict(root=str(root), source_info=str(info_path),
            source_info_sha256=sha(info_path), selected_info=str(selected_info.resolve()), records=records)
        (args.output/'cache_manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
        del loader
        gc.collect()
    handle.remove()


if __name__ == '__main__':
    main()

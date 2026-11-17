#!/usr/bin/env python3
"""Cache missing final yaw-gate context, without modifying the original cache.

Frozen parent checkpoint; native CUDA AMP. Sidecars contain only the gate's
256-dimensional visual/state input. Train/validation membership is unchanged.
"""
import argparse
import gc
import json
import logging
from pathlib import Path
import time

from train_balanced_fisheye import Config, MODELS, init_default_scope, np, torch
from mmengine.runner import Runner, load_checkpoint
from mmengine.logging import MMLogger
from finetune_heading_rows import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();source=args.cache/'cache_manifest.json'
    manifest=json.loads(source.read_text());out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    pins=dict(checkpoint_sha256=manifest['checkpoint_sha256'],source_cache_sha256=sha(source),
              config_sha256=sha(Path(manifest['config'])))
    if pins['checkpoint_sha256']!=sha(Path(manifest['checkpoint'])):
        raise ValueError('Parent checkpoint changed')
    status_path=out/'context_manifest.json'
    if status_path.exists():
        previous=json.loads(status_path.read_text())
        if any(previous.get(k)!=v for k,v in pins.items()):
            raise ValueError('Context cache provenance differs; use another output')
    status=dict(**pins,status='caching',datasets={})
    MMLogger.get_current_instance().setLevel(logging.WARNING)
    config=Config.fromfile(manifest['config']);init_default_scope(config.default_scope)
    model=MODELS.build(config.model);model.init_weights()
    load_checkpoint(model,manifest['checkpoint'],map_location='cpu',strict=True)
    model.cuda().eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    captured={}
    def capture(module,values):
        feature,_,position,_,_=values
        ref=feature+position if module.state_conditioned else feature
        captured['context']=ref.detach().float().cpu().numpy().astype(np.float16)[0]
    handle=model.head.fisheye_box_decoder.layers[-1].refinement.register_forward_pre_hook(capture)
    start=time.monotonic();total=sum(len(manifest['datasets'][k]['records']) for k in ('new','old_train','old_val'));done=0
    for name in ('new','old_train','old_val'):
        dataset=manifest['datasets'][name]
        if sha(Path(dataset['source_info']))!=dataset['source_info_sha256']:
            raise ValueError('Dataset annotations changed')
        cfg=config.test_dataloader.copy();cfg.batch_size=1;cfg.num_workers=2;cfg.persistent_workers=False
        cfg.dataset=dict(cfg.dataset,data_root=dataset['root'],ann_file=dataset['selected_info'],
                         test_mode=True,load_eval_anns=False)
        loader=Runner.build_dataloader(cfg)
        if len(loader)!=len(dataset['records']):raise ValueError('Incomplete source cache')
        directory=out/name;directory.mkdir(exist_ok=True);records=[];status['datasets'][name]=dict(records=records)
        for index,(batch,record) in enumerate(zip(loader,dataset['records'])):
            path=directory/('%05d.npz'%index)
            if path.exists():
                with np.load(path,allow_pickle=False) as row:
                    if str(row['token'])!=record['token'] or row['context'].shape!=(300,256):
                        raise ValueError('Invalid sidecar')
            else:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                    data=model.data_preprocessor(batch,training=False)
                    model._forward(data['inputs'],data.get('data_samples'))
                # Temporary file + replace makes interrupted resumes safe.
                temporary=path.with_suffix('.partial.npz')
                np.savez_compressed(temporary,context=captured['context'],token=np.asarray(record['token']))
                temporary.replace(path)
            records.append(dict(token=record['token'],path=str(path)));done+=1
            if done%25==0 or done==total:
                elapsed=time.monotonic()-start
                status.update(completed=done,total=total,eta_seconds=elapsed/max(done,1)*(total-done))
                status_path.write_text(json.dumps(status,indent=2)+'\n')
                print('%s %d/%d | total %d/%d | ETA %.1f min'%(name,index+1,len(loader),done,total,status['eta_seconds']/60),flush=True)
        del loader;gc.collect()
    handle.remove();status.update(status='complete',completed=total,total=total)
    status_path.write_text(json.dumps(status,indent=2)+'\n')


if __name__=='__main__':main()

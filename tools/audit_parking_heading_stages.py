#!/usr/bin/env python3
"""Localize parking yaw error across decoder stages using fixed final matches.

Read-only diagnostic; final query/GT assignments stay fixed across stages so
changing assignment quality cannot masquerade as heading improvement.
"""
import argparse
import json
import logging
from pathlib import Path

from train_balanced_fisheye import Config,MODELS,init_default_scope,np,torch
from mmengine.runner import Runner,load_checkpoint
from mmengine.logging import MMLogger
from finetune_heading_rows import sha,match_queries,angular_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--targets',type=Path,required=True)
    p.add_argument('--frames',type=int,default=24);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();manifest=json.loads((args.cache/'cache_manifest.json').read_text())
    if sha(Path(manifest['checkpoint']))!=manifest['checkpoint_sha256']:raise ValueError('Parent changed')
    targets=json.loads(args.targets.read_text())['frames_data'];dataset=manifest['datasets']['new']
    eligible=[i for i,r in enumerate(dataset['records']) if targets[r['source_index']]['targets']]
    indices={eligible[i] for i in np.unique(np.linspace(0,len(eligible)-1,min(args.frames,len(eligible))).astype(int))}
    config=Config.fromfile(manifest['config']);init_default_scope(config.default_scope)
    MMLogger.get_current_instance().setLevel(logging.WARNING)
    model=MODELS.build(config.model);model.init_weights();load_checkpoint(model,manifest['checkpoint'],map_location='cpu',strict=True)
    model.cuda().eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    cfg=config.test_dataloader.copy();cfg.batch_size=1;cfg.num_workers=2;cfg.persistent_workers=False
    cfg.dataset=dict(cfg.dataset,data_root=dataset['root'],ann_file=dataset['selected_info'],test_mode=True,load_eval_anns=False)
    rows=[];angles=None;truth=[]
    for i,batch in enumerate(Runner.build_dataloader(cfg)):
        if i not in indices:continue
        record=dataset['records'][i];gt=targets[record['source_index']]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            data=model.data_preprocessor(batch,training=False);raw=model._forward(data['inputs'],data.get('data_samples'))
        with np.load(record['path'],allow_pickle=False) as cached:
            iq,ig=match_queries(cached,gt['targets'])
        if not len(iq):continue
        yaw=[torch.atan2(value[0,:,6].float(),value[0,:,7].float()).cpu().numpy()[iq] for value in raw['prediction']]
        if angles is None:angles=[[] for _ in yaw]
        target=np.asarray([gt['targets'][j]['box'][6] for j in ig]);truth.extend(target)
        for k,value in enumerate(yaw):angles[k].extend(value)
        rows.append(dict(frame=record['source_index'],split=gt['split'],matches=len(iq),
                         stage_axis_mean=[angular_metrics(value,target)['axis_mean_deg'] for value in yaw]))
    if not truth:raise ValueError('No diagnostic matches')
    report=dict(checkpoint_sha256=manifest['checkpoint_sha256'],sampled_frames=len(rows),matched_objects=len(truth),
        fixed_final_query_assignment=True,stages=[dict(stage=i,**angular_metrics(np.asarray(value),np.asarray(truth))) for i,value in enumerate(angles)],
        frames=rows,scope='Diagnostic on selected same-scene pseudo-label frames, not a held-out final quality benchmark.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='frames'},indent=2),flush=True)


if __name__=='__main__':main()

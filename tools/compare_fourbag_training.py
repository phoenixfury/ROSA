#!/usr/bin/env python3
"""Read-only checkpoint audit and fresh native comparison on fixed data splits.

Only analysis artifacts are written. Checkpoints, source annotations, model
implementation and deployment defaults are never modified.
"""
import argparse
from collections import Counter,defaultdict
from contextlib import redirect_stdout
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROSA=Path(__file__).resolve().parents[1]
ROOT=ROSA.parents[1]
OFFICIAL=ROOT/'projects/Fisheye3DOD'
sys.path[:0]=[str(ROOT),str(ROSA),str(OFFICIAL),str(ROOT/'tools/custom_data')]
os.environ['FISHEYE3DOD_OFFICIAL_ROOT']=str(OFFICIAL)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from run import _load_checkpoint_without_executing_code,Config,np,torch
import mmengine


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024**2),b''):h.update(part)
    return h.hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')
    temporary.replace(path)


def state_digest(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def checkpoint_audit(path,reference):
    checkpoint=_load_checkpoint_without_executing_code(path);state=checkpoint['state_dict'];meta=checkpoint.get('meta',{})
    mismatches=[key for key,value in state.items() if key not in reference or reference[key].shape!=value.shape]
    missing=sorted(set(reference)-set(state))
    changed=defaultdict(lambda:dict(tensors=0,squared_delta=0.,squared_reference=0.))
    bn_changes=0
    for key,value in state.items():
        if key not in reference or reference[key].shape!=value.shape or not value.is_floating_point():continue
        group='backbone' if key.startswith('img_backbone.') else 'neck' if key.startswith('img_neck.') else 'head'
        a=value.float();b=reference[key].float();row=changed[group]
        row['tensors']+=1;row['squared_delta']+=float(torch.sum((a-b).double().square()))
        row['squared_reference']+=float(torch.sum(b.double().square()))
        if ('running_mean' in key or 'running_var' in key) and not torch.equal(value,reference[key]):bn_changes+=1
    for row in changed.values():row['relative_l2_change']=(row['squared_delta']/max(row['squared_reference'],1e-30))**.5
    sources=meta.get('training_implementation_sha256',{})
    source_comparison={}
    for name,digest in sources.items():
        local=ROOT/name
        if not local.is_file():local=ROSA/name
        source_comparison[name]=dict(training_sha256=digest,local_sha256=sha(local) if local.is_file() else None)
    result=dict(path=str(path),file_sha256=sha(path),state_sha256=state_digest(state),
        epoch=meta.get('epoch'),iteration=meta.get('iter'),keys=len(state),tensor_elements=sum(v.numel() for v in state.values()),
        shape_mismatches=mismatches,missing_keys=missing,changes_vs_previous_best=dict(changed),
        changed_bn_running_stat_tensors=bn_changes,training_contract=meta.get('training_contract'),
        training_implementation=source_comparison,source_revision=meta.get('source_revision'),
        experiment_name=meta.get('experiment_name'))
    del checkpoint,state;gc.collect()
    return result


def prepare(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    data=args.data_root
    val=mmengine.load(data/'rosa/fisheye3dod_infos_val.pkl')
    train=mmengine.load(data/'rosa/fisheye3dod_infos_train.pkl')['data_list']
    all_rows=mmengine.load(data/'rosa/fisheye3dod_infos_all.pkl')['data_list']
    old=mmengine.load(ROOT/'data/Custom/combined_improved_detzero_rosa/rosa/fisheye3dod_infos_val.pkl')['data_list']
    sequence=lambda row:row['metainfo']['sequence_id']
    by_token={str(row['token']):row for row in all_rows}
    if len(by_token)!=len(all_rows):raise ValueError('Duplicate dataset tokens')
    vtokens={str(row['token']) for row in val['data_list']}
    ttokens={str(row['token']) for row in train}
    if vtokens&ttokens:raise ValueError('Train/validation leakage')
    oldtokens={str(row['token']) for row in old}
    if not oldtokens<=vtokens:raise ValueError('Original holdout is not preserved')
    new=[row for row in all_rows if sequence(row)=='25_09']
    union={str(row['token']):copy.deepcopy(row) for row in [*val['data_list'],*new]}
    rows=sorted(union.values(),key=lambda row:(int(row['metainfo']['acquisition_timestamp_ns']),str(row['token'])))
    info=dict(metainfo=val.get('metainfo',{}),data_list=rows)
    mmengine.dump(info,out/'evaluation_union.pkl')
    old_equal=all(np.array_equal(row['annos']['gt_boxes'],by_token[str(row['token'])]['annos']['gt_boxes']) and
        np.array_equal(row['annos']['gt_names'],by_token[str(row['token'])]['annos']['gt_names']) for row in old)
    subsets=dict(old_validation_601=[str(r['token']) for r in old],
        combined_validation_674=[str(r['token']) for r in val['data_list']],
        new_bag_holdout_73=[str(r['token']) for r in val['data_list'] if sequence(r)=='25_09'],
        new_bag_full_diagnostic_591=[str(r['token']) for r in new])
    counts={name:dict(frames=len(tokens),class_counts=dict(Counter(str(n) for t in tokens for n in by_token[t]['annos']['gt_names'])))
            for name,tokens in subsets.items()}
    models=dict(new_best6=dict(checkpoint=str(args.training/'best_fisheye_bev_mAP_epoch_6.pth'),
                   config=str(args.training/'rosa_vov99_baseline_adapt_8e.py')),
        previous_best=dict(checkpoint=str(ROSA/'work_dirs/singleframe_refresh_20261031/best_fisheye_bev_mAP_epoch_1.pth'),
                   config=str(ROSA/'configs/fisheye3dod/rosa_vov99_singleframe_refresh_24e.py')),
        new_epoch8=dict(checkpoint=str(args.training/'epoch_8.pth'),config=str(args.training/'rosa_vov99_baseline_adapt_8e.py')))
    reference=_load_checkpoint_without_executing_code(Path(models['previous_best']['checkpoint']))['state_dict']
    audit={}
    for path in [Path(models['previous_best']['checkpoint']),*sorted(args.training.glob('*.pth'))]:
        print('Audit:',path.name,flush=True);audit[str(path)]=checkpoint_audit(path,reference)
    del reference;gc.collect()
    for row in models.values():
        row['checkpoint_sha256']=audit[row['checkpoint']]['file_sha256']
        row['state_sha256']=audit[row['checkpoint']]['state_sha256']
        if audit[row['checkpoint']]['shape_mismatches'] or audit[row['checkpoint']]['missing_keys']:
            raise ValueError('Architecture differs; do not silently load a checkpoint')
    scalars=[json.loads(line) for line in (args.training/'20261105_223146/vis_data/scalars.json').read_text().splitlines() if line.strip()]
    history=[dict(epoch=int(row['step']),**{k.split('/',1)[1]:v for k,v in row.items() if k.startswith('fisheye_bev/')})
             for row in scalars if 'fisheye_bev/mAP' in row]
    train_history=[dict(epoch=e,mean_loss=float(np.mean([r['loss'] for r in scalars if r.get('epoch')==e and 'loss' in r])))
                   for e in sorted({r['epoch'] for r in scalars if 'loss' in r})]
    plan=dict(data_root=str(data),union=str(out/'evaluation_union.pkl'),union_sha256=sha(out/'evaluation_union.pkl'),
        union_frames=len(rows),subsets=subsets,counts=counts,models=models,training_history=history,train_loss=train_history,
        old_annotation_boxes_names_unchanged=old_equal,train_val_overlap=0,
        full_new_bag_split_counts=dict(train=sum(str(r['token']) in ttokens for r in new),
            val=sum(str(r['token']) in vtokens for r in new),purge=sum(str(r['token']) not in ttokens|vtokens for r in new)),
        protocol='Fresh native AMP PyTorch; final decoder stage; quality-before-top-k;300 queries; class-wise rotated NMS0.5; '
            'unfiltered official AP/NDS; common score0.1 for operating/heading diagnostics. Full new bag includes training frames.')
    dump(out/'checkpoint_audit.json',audit);dump(out/'plan.json',plan)
    print('Prepared',len(rows),'unique frames',counts,flush=True)


def infer_one(args):
    plan=json.loads((args.output/'plan.json').read_text());model=plan['models'][args.model];directory=args.output/args.model
    directory.mkdir(exist_ok=True)
    checkpoint=Path(model['checkpoint'])
    if sha(checkpoint)!=model['checkpoint_sha256'] or sha(Path(plan['union']))!=plan['union_sha256']:
        raise ValueError('Input checksum changed')
    config=Config.fromfile(model['config'])
    config.model.head.instance_bank.anchor=str(Path(plan['data_root'])/'rosa/anchors/kmeans300_seed3407.npy')
    if config.model.head.instance_bank.num_temp_instances!=0:raise ValueError('This comparison requires single-frame models')
    config.test_dataloader.batch_size=1;config.test_dataloader.num_workers=2
    config.test_dataloader.dataset.update(ann_file=plan['union'],data_root=plan['data_root'],test_mode=True,load_eval_anns=False)
    config.load_from=str(checkpoint);config.resume=False;config.launcher='none';config.work_dir=str(directory/'work')
    config.train_dataloader=None;config.train_cfg=None;config.optim_wrapper=None;config.param_scheduler=None
    config.val_dataloader=None;config.val_cfg=None;config.val_evaluator=None;config.custom_hooks=[]
    config.test_evaluator=dict(type='DumpResults',out_file_path=str(directory/'predictions.pkl'))
    config.default_hooks.checkpoint=None;config.default_hooks.logger.interval=50
    if 'init_cfg' in config.model.img_backbone:config.model.img_backbone.init_cfg=None
    config.dump(str(directory/'resolved_inference_config.py'))
    from mmengine.runner import Runner
    from infer_fisheye_checkpoints import export_portable_json
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    runner=Runner.from_cfg(config);runner.load_or_resume()
    safe=_load_checkpoint_without_executing_code(checkpoint)
    runner.model.load_state_dict(safe['state_dict'],strict=True)
    del safe;gc.collect()
    runner.model.head.instance_bank.reset()
    runner.test()
    del runner;gc.collect();torch.cuda.empty_cache()
    export_portable_json(directory/'predictions.pkl',directory/'predictions.json',checkpoint,0.)
    payload=json.loads((directory/'predictions.json').read_text());payload['checkpoint_sha256']=model['checkpoint_sha256']
    payload['nms_iou']=.5;payload['class_score_thresholds']=None
    dump(directory/'predictions.json',payload)
    dump(directory/'inference_complete.json',dict(checkpoint_sha256=model['checkpoint_sha256'],
        union_sha256=plan['union_sha256'],frames=len(payload['frames']),predictions_sha256=sha(directory/'predictions.json')))


def metric_report(args):
    from evaluate_heading_adaptation import official
    from audit_fisheye_false_positives import CLASSES,RANGES
    from scipy.optimize import linear_sum_assignment
    plan=json.loads((args.output/'plan.json').read_text())
    if sha(Path(plan['union']))!=plan['union_sha256']:raise ValueError('Evaluation annotations changed')
    rows=mmengine.load(plan['union'])['data_list'];infos={str(row['token']):row for row in rows}
    targets=json.loads((ROSA/'work_dirs/static_heading_adapt_20261105/targets/heading_targets.json').read_text())
    ego={str(row['token']):(row['ego_speed_mps']>.15 or abs(row['ego_yaw_rate_rps'])>.01) for row in targets['frames_data']}
    models={};pairsets={}
    for tag,model in plan['models'].items():
        path=args.output/tag/'predictions.json'
        if not (args.output/tag/'inference_complete.json').is_file():continue
        done=json.loads((args.output/tag/'inference_complete.json').read_text())
        if done['checkpoint_sha256']!=model['checkpoint_sha256'] or done['union_sha256']!=plan['union_sha256'] or done['predictions_sha256']!=sha(path):
            raise ValueError('Completed inference provenance differs')
        document=json.loads(path.read_text())
        if document['checkpoint_sha256']!=model['checkpoint_sha256']:raise ValueError('Prediction/checkpoint mismatch')
        frames={str(row['token']):row for row in document['frames']}
        if set(frames)!=set(infos):raise ValueError('Prediction token coverage differs')
        entry={};pairsets[tag]={}
        existing=args.output/tag/'metrics.json'
        cached=json.loads(existing.read_text()) if existing.is_file() else {}
        for name,tokens in plan['subsets'].items():
            payload=dict(frames=[frames[t] for t in tokens]);selected=[infos[t] for t in tokens]
            metrics=cached.get(name,{}).get('metrics')
            if metrics is None:
                with (args.output/tag/(name+'_metrics.log')).open('w') as log,redirect_stdout(log):metrics=official(payload,selected)
            tp=fp=fn=0;pairset={};class_counts={c:Counter() for c in CLASSES};confusion=Counter()
            for token in tokens:
                frame=frames[token];info=infos[token];gt=np.asarray(info['annos']['gt_boxes']).reshape(-1,7)
                gn=np.asarray(info['annos']['gt_names'],dtype=str)
                pred=np.asarray(frame['boxes_3d'])[:,:7] if frame['boxes_3d'] else np.empty((0,7))
                pn=np.asarray(frame['names'],dtype=str);keep=np.asarray(frame['scores'])>=.1;pred,pn=pred[keep],pn[keep]
                valid_gt=np.asarray([np.linalg.norm(box[:2])<RANGES[str(n)] for box,n in zip(gt,gn)],dtype=bool)
                gt_ids=np.asarray(info['annos']['gt_ids'])[valid_gt]
                gt,gn=gt[valid_gt],gn[valid_gt]
                valid_pred=np.asarray([np.linalg.norm(box[:2])<RANGES[str(n)] for box,n in zip(pred,pn)],dtype=bool)
                pred,pn=pred[valid_pred],pn[valid_pred]
                matches=0
                for cls in CLASSES:
                    class_counts[cls]['gt']+=int(np.sum(gn==cls));class_counts[cls]['predictions']+=int(np.sum(pn==cls))
                if len(gt) and len(pred):
                    cost=np.linalg.norm(gt[:,None,:2]-pred[None,:,:2],axis=-1);cost[gn[:,None]!=pn[None,:]]=1e6
                    cost[cost>=2.]=1e6;ia,ib=linear_sum_assignment(cost);matches=int((cost[ia,ib]<2.).sum())
                    for a,b in zip(ia,ib):
                        if cost[a,b]<2.:class_counts[str(gn[a])]['tp']+=1
                tp+=matches;fp+=len(pred)-matches;fn+=len(gt)-matches
                gi=np.flatnonzero(np.isin(gn,('Car','Van','Truck','Bus')))
                pi=np.flatnonzero(np.isin(pn,('Car','Van','Truck','Bus')))
                if len(gi) and len(pi):
                    cost=np.linalg.norm(gt[gi,None,:2]-pred[None,pi,:2],axis=-1);cost[cost>=1.5]=1e6
                    ia,ib=linear_sum_assignment(cost)
                    for a,b in zip(ia,ib):
                        if cost[a,b]>=1.5:continue
                        d=float(pred[pi[b],6]-gt[gi[a],6]);axis=float(np.degrees(abs((d+np.pi/2)%np.pi-np.pi/2)))
                        directed=float(np.degrees(abs((d+np.pi)%(2*np.pi)-np.pi)))
                        confusion[str(gn[gi[a]])+' -> '+str(pn[pi[b]])]+=1
                        pairset[token+':'+str(gi[a])]=dict(axis_deg=axis,directed_deg=directed,gt_name=str(gn[gi[a]]),
                            source_frame=int(info['metainfo'].get('source_frame_id',-1)),
                            track=str(info['metainfo']['sequence_id'])+':'+str(gt_ids[gi[a]]),
                            axis_residual_deg=float(np.degrees((d+np.pi/2)%np.pi-np.pi/2)),moving_ego=ego.get(token),
                            target_axis_deg=float(np.degrees(abs((gt[gi[a],6]+np.pi/2)%np.pi-np.pi/2))))
            precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
            entry[name]=dict(frames=len(tokens),metrics=metrics,operating_score_0_1=dict(tp=tp,fp=fp,fn=fn,
                precision=precision,recall=recall,f1=2*precision*recall/max(precision+recall,1e-12)),
                per_class_operating={c:dict(v,fp=v['predictions']-v['tp'],fn=v['gt']-v['tp']) for c,v in class_counts.items()},
                vehicle_class_confusion=dict(confusion),vehicle_heading=angle_summary(pairset.values()),
                vehicle_heading_ego_stopped=angle_summary(v for v in pairset.values() if v['moving_ego'] is False),
                vehicle_heading_ego_moving=angle_summary(v for v in pairset.values() if v['moving_ego'] is True),
                heading_residual_stability=heading_stability(pairset.values()))
            pairsets[tag][name]=pairset
        models[tag]=entry
        dump(args.output/tag/'metrics.json',entry)
        dump(args.output/tag/'heading_pairs.json',pairsets[tag])
    common={}
    if 'previous_best' in models:
        for name in plan['subsets']:
            sets=[set(pairsets[tag][name]) for tag in models];keys=set.intersection(*sets)
            common[name]=dict(common_gt_matches=len(keys),models={tag:angle_summary(pairsets[tag][name][k] for k in keys) for tag in models})
            common[name]['ego_motion']={label:{tag:angle_summary(pairsets[tag][name][k] for k in keys
                if pairsets['previous_best'][name][k]['moving_ego'] is moving) for tag in models}
                for label,moving in [('stopped',False),('moving',True)]}
            common[name]['residual_stability']={tag:heading_stability(pairsets[tag][name][k] for k in keys) for tag in models}
            common[name]['orientation_bins']={label:dict(models={tag:angle_summary(pairsets[tag][name][k] for k in keys
                if lower<=pairsets['previous_best'][name][k]['target_axis_deg']<upper) for tag in models})
                for label,lower,upper in [('longitudinal_0_30',0,30),('oblique_30_60',30,60),('perpendicular_60_90',60,90.0001)]}
    pairwise={}
    if 'previous_best' in models:
        for tag in models:
            if tag=='previous_best':continue
            pairwise[tag]={}
            for name in plan['subsets']:
                keys=set(pairsets[tag][name])&set(pairsets['previous_best'][name])
                pairwise[tag][name]=dict(common_gt_matches=len(keys),
                    baseline=angle_summary(pairsets['previous_best'][name][k] for k in keys),
                    candidate=angle_summary(pairsets[tag][name][k] for k in keys),
                    ego_motion={label:dict(
                        baseline=angle_summary(pairsets['previous_best'][name][k] for k in keys if pairsets['previous_best'][name][k]['moving_ego'] is moving),
                        candidate=angle_summary(pairsets[tag][name][k] for k in keys if pairsets['previous_best'][name][k]['moving_ego'] is moving))
                        for label,moving in [('stopped',False),('moving',True)]})
    report=dict(models=models,common_vehicle_heading=common,pairwise_vehicle_heading=pairwise,counts=plan['counts'],
        protocol=plan['protocol'],full_new_bag_split_counts=plan['full_new_bag_split_counts'],
        caveats=['Pseudo-labels, not a fresh manual benchmark.','Validation has no Van positives.',
                 'The full591 new-bag result includes training frames; only73 are held out.',
                 'Common vehicle-family heading matches use center<1.5m; operating precision/recall use class-aware center<2m, score0.1 and official class ranges.',
                 'Heading residual stability is the frame-to-frame change in axial prediction-minus-GT error, consecutive source frames only.'])
    report['validation_unique_tracks']={}
    for name,tokens in plan['subsets'].items():
        unique=defaultdict(set)
        for token in tokens:
            info=infos[token]
            for cls,identity in zip(info['annos']['gt_names'],info['annos']['gt_ids']):
                unique[str(cls)].add((info['metainfo']['sequence_id'],str(identity)))
        report['validation_unique_tracks'][name]={cls:len(ids) for cls,ids in unique.items()}
    dump(args.output/'comparison.json',report)
    if len(models)==len(plan['models']):write_report(args,plan,report)
    print('METRICS',json.dumps({tag:{name:{k:r['metrics'][k] for k in ('mAP','NDS','mAOE')} for name,r in entry.items()}
          for tag,entry in models.items()}),flush=True)


def angle_summary(records):
    values=list(records)
    if not values:return dict(matches=0)
    axis=np.asarray([r['axis_deg'] for r in values]);directed=np.asarray([r['directed_deg'] for r in values])
    return dict(matches=len(values),axis_mean_deg=float(axis.mean()),axis_median_deg=float(np.median(axis)),
        axis_p95_deg=float(np.percentile(axis,95)),axis_over45_fraction=float(np.mean(axis>45)),
        directed_mean_deg=float(directed.mean()),directed_over90_fraction=float(np.mean(directed>90)))


def heading_stability(records):
    tracks=defaultdict(list)
    for row in records:tracks[row['track']].append(row)
    differences=[]
    for values in tracks.values():
        values=sorted(values,key=lambda row:row['source_frame'])
        for a,b in zip(values,values[1:]):
            if b['source_frame']-a['source_frame']==1:
                differences.append(abs((b['axis_residual_deg']-a['axis_residual_deg']+90)%180-90))
    if not differences:return dict(consecutive_pairs=0)
    return dict(consecutive_pairs=len(differences),mean_abs_change_deg=float(np.mean(differences)),
        p95_abs_change_deg=float(np.percentile(differences,95)))


def initializer_audit(args,plan):
    """Separate parameter drift from BN buffers, using the actual initializer."""
    path=ROSA/'work_dirs/server/rosa_vov99_baseline_adapt_3080ti_2gpu/best_fisheye_bev_NDS_epoch_4.pth'
    candidate=Path(plan['models']['new_best6']['checkpoint'])
    provenance=dict(initializer=str(path),initializer_sha256=sha(path),candidate=str(candidate),
                    candidate_sha256=sha(candidate))
    destination=args.output/'initializer_audit.json'
    if destination.is_file():
        cached=json.loads(destination.read_text())
        if all(cached.get(k)==v for k,v in provenance.items()):return cached
    initial=_load_checkpoint_without_executing_code(path)['state_dict']
    state=_load_checkpoint_without_executing_code(candidate)['state_dict']
    groups=defaultdict(lambda:dict(tensors=0,elements=0,squared_delta=0.,squared_reference=0.))
    counters=Counter()
    for key,value in state.items():
        if key.endswith('num_batches_tracked'):
            counters[int(value)-int(initial[key])]+=1
        if not key.endswith(('.weight','.bias')):continue
        group='backbone' if key.startswith('img_backbone.') else 'neck' if key.startswith('img_neck.') else 'head'
        row=groups[group];a=value.double();b=initial[key].double()
        row['tensors']+=1;row['elements']+=value.numel()
        row['squared_delta']+=float(torch.sum((a-b).square()))
        row['squared_reference']+=float(torch.sum(b.square()))
    for row in groups.values():row['relative_l2_change']=(row['squared_delta']/max(row['squared_reference'],1e-30))**.5
    result=dict(provenance,parameter_drift=dict(groups),bn_batch_counter_deltas=dict(counters))
    dump(destination,result)
    return result


def write_report(args,plan,report):
    models=report['models'];tags=list(models)
    initialization=initializer_audit(args,plan)
    labels={'previous_best':'Previous best E007 epoch1','new_best6':'Four-bag epoch6','new_epoch8':'Four-bag epoch8'}
    lines=['# ROSA four-bag checkpoint analysis — 2026-11-07','',
        'Fresh native AMP inference, identical samples, strict state loading, unchanged source annotations and deployment defaults.','',
        '## Headline comparison','',
        '| Model | Combined674 mAP | Combined674 NDS | Old601 mAP | Old601 NDS | New73 Car AP |',
        '|---|---:|---:|---:|---:|---:|']
    for tag in tags:
        m=models[tag];a=m['combined_validation_674']['metrics'];b=m['old_validation_601']['metrics'];c=m['new_bag_holdout_73']['metrics']
        lines.append(f'| {labels[tag]} | {a["mAP"]:.5f} | {a["NDS"]:.5f} | {b["mAP"]:.5f} | {b["NDS"]:.5f} | {c["Car_AP"]:.5f} |')
    best=max(tags,key=lambda tag:models[tag]['combined_validation_674']['metrics']['mAP'])
    lines+=['',f'The highest combined-validation mAP is **{labels[best]}**. NDS and old-domain results must also be considered; a macro AP increase is not uniform improvement.','',
        '## Parking/vehicle headings on the new73 held-out frames','',
        'Vehicle-family center matching <1.5m, score>=0.1, official class ranges. The following are pairwise common GT matches versus the previous best, so different detection recall does not silently change the comparison.','',
        '| Candidate | Common matches | Previous axis MAE deg | Candidate axis MAE deg | Previous directed MAE deg | Candidate directed MAE deg |',
        '|---|---:|---:|---:|---:|---:|']
    for tag,rows in report['pairwise_vehicle_heading'].items():
        row=rows['new_bag_holdout_73'];a=row['baseline'];b=row['candidate']
        lines.append(f'| {labels[tag]} | {row["common_gt_matches"]} | {a["axis_mean_deg"]:.2f} | {b["axis_mean_deg"]:.2f} | {a["directed_mean_deg"]:.2f} | {b["directed_mean_deg"]:.2f} |')
    lines+=['','Axial error treats a 180-degree front/back flip as the same box axis; directed error does not. Thus improved parking alignment does not prove all front/back errors are solved. These are errors against pseudo-labels, not the earlier 5-degree ONNX-versus-PyTorch numerical tolerance.','',
        '### Ego stopped versus moving','',
        '| Candidate | Ego group | Common matches | Previous axis MAE deg | Candidate axis MAE deg |',
        '|---|---|---:|---:|---:|']
    for tag,rows in report['pairwise_vehicle_heading'].items():
        for mode,row in rows['new_bag_holdout_73']['ego_motion'].items():
            a=row['baseline'];b=row['candidate']
            if a['matches']:lines.append(f'| {labels[tag]} | {mode} | {a["matches"]} | {a["axis_mean_deg"]:.2f} | {b["axis_mean_deg"]:.2f} |')
    lines+=['','Ego groups use the existing odometry/motion-audit metadata (>0.15m/s or yaw rate>0.01rad/s). No odometry was modified.','',
        '### Heading stability on common detections','',
        '| Model | Split | Axis MAE deg | Consecutive pairs | Mean residual change deg |',
        '|---|---|---:|---:|---:|']
    for name in ('old_validation_601','new_bag_holdout_73'):
        common=report['common_vehicle_heading'][name]
        for tag in tags:
            angles=common['models'][tag];stability=common['residual_stability'][tag]
            lines.append(f'| {labels[tag]} | {name} | {angles["axis_mean_deg"]:.2f} | {stability["consecutive_pairs"]} | {stability["mean_abs_change_deg"]:.2f} |')
    lines+=['','All three models share the same GT matches in this table. Residual change is frame-to-frame axial prediction-minus-GT error, not raw yaw change: legitimate ego/vehicle rotation is not automatically counted as jitter. GT noise and association ambiguity still affect it.','',
        '## Old-domain class tradeoffs (601 unchanged validation frames)','',
        '| Model | Car AP | Truck AP | Bus AP | Pedestrian AP | Cyclist AP | mATE m |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for tag in tags:
        m=models[tag]['old_validation_601']['metrics']
        lines.append('| '+labels[tag]+' | '+' | '.join(f'{m[k]:.5f}' for k in ('Car_AP','Truck_AP','Bus_AP','Pedestrian_AP','Cyclist_AP','mATE'))+' |')
    lines+=['','Fisheye3DOD NDS uses 3*mAP plus translation, scale and orientation scores, divided by 6. A worse translation error can offset better macro AP/orientation. Inspect Car/Truck/Bus individually instead of choosing solely by macro mAP.','',
        '## Fixed operating-point precision/recall','',
        'Score0.1 for all models, class-aware BEV center match<2m and the same evaluator ranges. These are diagnostic counts relative to pseudo-labels, not independent human-confirmed false positives. The old checkpoint-specific class thresholds were not retuned or silently rebound to the new weights.','',
        '| Model | Split | Precision | Recall | F1 | FP | FN |','|---|---|---:|---:|---:|---:|---:|']
    for tag in tags:
        for name in ('old_validation_601','new_bag_holdout_73'):
            m=models[tag][name]['operating_score_0_1']
            lines.append(f'| {labels[tag]} | {name} | {m["precision"]:.4f} | {m["recall"]:.4f} | {m["f1"]:.4f} | {m["fp"]} | {m["fn"]} |')
    lines+=['','Epoch8 trades some recall for fewer false positives at this fixed score and slightly better F1; it still has worse combined AP/NDS. A new per-class threshold sweep could change that operating-point tradeoff and was not performed here.',
        'Per-class operating counts, vehicle subtype confusion, orientation-bin results and consecutive-frame heading-residual stability are in `comparison.json`.','',
        '## Training history','',
        '| Epoch | Server674 mAP | Server674 NDS | Mean logged train loss |','|---:|---:|---:|---:|']
    losses={r['epoch']:r['mean_loss'] for r in plan['train_loss']}
    for r in plan['training_history']:
        lines.append(f'| {r["epoch"]} | {r["mAP"]:.5f} | {r["NDS"]:.5f} | {losses[r["epoch"]]:.4f} |')
    lines+=['','Both server best metrics occur at epoch6. Epochs7–8 lower training loss but fail to improve those validation peaks. This is consistent with overfitting/objective-versus-validation tradeoffs, not evidence that the larger epoch8 file is a better model. Exact causal attribution needs an ablation.','',
        '## What was actually trained','',
        '- Same single-frame VoVNet99 + FPN + six-stage ROSA box decoder,300 queries, six classes, quality-before-top-k and class-wise rotated NMS0.5. Temporal instance slots remain0.',
        '- Checkpoint audit: all1,250 state tensor keys/shapes match the previous best. Epoch6 mAP-best, NDS-best and regular epoch6 contain identical model tensors; evaluate/export only one. The full regular checkpoints also contain optimizer state.',
        '- The uploaded run used the baseline-adaptation8-epoch recipe at LR1e-4, effective batch8, with all backbone stages/BN statistics unfrozen.',
        '- Important lineage difference: initialization was the older `server/rosa_vov99_baseline_adapt_3080ti_2gpu/best_fisheye_bev_NDS_epoch_4.pth`, **not** the previous best `singleframe_refresh_20261031/best_fisheye_bev_mAP_epoch_1.pth`. This is an alternative adaptation branch, not a continuation from E007.',
        '- The prepared E011 recipe instead specified E007 initialization, LR5e-5 and up to24 epochs with plateau stopping. The uploaded run is therefore not that exact recipe.',
        '- The saved human-readable contract still says16_09+30_09, but the actual resolved paths,3787 training frames,674 validation frames and class counts show all four bags were used.',
        '- Local decoder and VoVNet implementations match the saved training hashes. The two differing root head/target files differ from their matching committed training sources only in loss-side NaN masking; prediction does not use those paths. All models run strict loading in the same local inference implementation.','',
        '### Parameter and BatchNorm changes','',
        'Relative L2 changes of weight/bias tensors versus the actual initialization checkpoint (not versus E007): '+
            ', '.join(f'{name} {row["relative_l2_change"]:.1%}' for name,row in initialization['parameter_drift'].items())+'.',
        f'BatchNorm counter increments across layers: `{initialization["bn_batch_counter_deltas"]}`. The 6,330 increment matches six epochs of 1,055 training iterations, consistent with one running-stat update per iteration, not double updates from checkpoint recomputation. Backbone adaptation is substantial, but weight drift alone does not prove the cause of old-domain regression.','',
        '## Dataset limits and interpretation','',
        f'- The full new591-frame bag contains {plan["full_new_bag_split_counts"]["train"]} training, {plan["full_new_bag_split_counts"]["purge"]} purge and {plan["full_new_bag_split_counts"]["val"]} held-out frames. Its full-bag scores are diagnostic and include training examples; do not describe them as unseen-data performance.',
        '- New73 held-out GT:2,328 Car boxes and8 Bus boxes; the other classes have no positives there. Six-class macro mAP/NDS is depressed by missing classes/default TP errors; Car AP and matched heading errors are more informative for this bag.',
        '- Validation has no Van positives. Combined validation has only13 Cyclist boxes from5 tracks; large Cyclist AP changes have weak statistical support.',
        '- Train/validation token overlap is0; the original601 GT boxes and names are byte-equivalent. However, this is same-scene chronological pseudo-label validation, and static objects can occur on both sides of the split. There is no new independent manually labeled test set.',
        '- Same-shape weights and a reproduced validation score do not establish generalization to unseen recordings.','',
        '## Recommended next decisions','',
        '1. Prefer the best combined-validation checkpoint over epoch8 when a balanced old/new result is needed; keep E007 as the old-domain regression reference.',
        '2. If old Car localization and Truck/Bus retention are priorities, use old-bag-balanced replay and the intended smaller LR/previous-best initialization in the next controlled run. Do not infer that more epochs alone will fix the tradeoff.',
        '3. Do not claim Van improvements without validation positives. Review rare-class labels and quantify uncertainty by track/recording rather than treating adjacent frames as independent samples.',
        '4. Re-export and retune the chosen new checkpoint before deployment; the old ONNX file and threshold profile still belong to E007. No model/export/default was replaced by this analysis.','',
        '## Reproduce','',
        'From the mmdetection3d repository root:',
        '```bash','conda run --no-capture-output -n openmmlab python projects/ROSA/tools/compare_fourbag_training.py --stage run','```',
        '', 'The utility writes isolated inference configs/logs, strict-load checkpoints, portable predictions, a de-duplicated1,192-frame evaluation union, raw metrics and this report. To recompute diagnostics from completed predictions use `--stage report`. Source data/checkpoints remain unchanged.','']
    (args.output/'REPORT.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('prepare','run','infer-one','report'),default='run')
    p.add_argument('--model',choices=('new_best6','previous_best','new_epoch8'))
    p.add_argument('--training',type=Path,default=ROSA/'work_dirs/combined_4bags')
    p.add_argument('--data-root',type=Path,default=ROOT/'data/Custom/combined_improved_detzero_4bags_20261105')
    p.add_argument('--output',type=Path,default=ROSA/'work_dirs/combined_4bags_analysis_20261107')
    args=p.parse_args();args.output=args.output.resolve();args.training=args.training.resolve();args.data_root=args.data_root.resolve()
    if args.stage=='prepare':prepare(args);return
    if args.stage=='infer-one':
        if not args.model:p.error('--model is required')
        import fcntl
        directory=args.output/args.model;directory.mkdir(parents=True,exist_ok=True)
        with (directory/'inference.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            complete=directory/'inference_complete.json'
            if complete.is_file():
                plan=json.loads((args.output/'plan.json').read_text());done=json.loads(complete.read_text())
                if done['checkpoint_sha256']!=plan['models'][args.model]['checkpoint_sha256'] or done['union_sha256']!=plan['union_sha256']:
                    raise ValueError('Stale inference cache')
                print('Reusing completed inference:',args.model,flush=True)
            else:infer_one(args)
        return
    if args.stage=='report':metric_report(args);return
    if not (args.output/'plan.json').is_file():prepare(args)
    plan=json.loads((args.output/'plan.json').read_text())
    for i,tag in enumerate(plan['models']):
        complete=args.output/tag/'inference_complete.json'
        if complete.exists():
            done=json.loads(complete.read_text())
            if done['checkpoint_sha256']!=plan['models'][tag]['checkpoint_sha256'] or done['union_sha256']!=plan['union_sha256']:
                raise ValueError('Stale existing inference result')
        else:
            directory=args.output/tag;directory.mkdir(exist_ok=True)
            dump(args.output/'status.json',dict(phase='inference',model=tag,model_number=i+1,models=len(plan['models']),
                frames=plan['union_frames'],log=str(directory/'inference.log')))
            with (directory/'inference.log').open('w') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--stage','infer-one','--model',tag,
                    '--output',str(args.output)],stdout=log,stderr=subprocess.STDOUT,check=True)
        metric_report(args)
    dump(args.output/'status.json',dict(phase='complete',models=list(plan['models']),report=str(args.output/'comparison.json')))


if __name__=='__main__':main()

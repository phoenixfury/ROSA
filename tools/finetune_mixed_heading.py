#!/usr/bin/env python3
"""Guarded old/new heading replay, retaining the exact original architecture.

75% old GT / 25% new heading examples, all old training frames eligible,
old-train teacher distillation, old-validation official mAP/NDS and yaw gates.
Cached passes are not full-backbone training epochs. No validation loss.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from train_balanced_fisheye import np, torch
import mmengine
from scipy.optimize import linear_sum_assignment
from finetune_heading_rows import sha, design, collect_supervision, angular_metrics
from evaluate_heading_adaptation import cached_predictions, official

CLASSES=('Car','Van','Truck','Bus','Pedestrian','Cyclist')


def stratified_weights(groups, cap=5.):
    counts=Counter(groups)
    values=np.asarray([1./np.sqrt(counts[group]) for group in groups],dtype=np.float64)
    values=np.minimum(values,cap*values.mean())
    return (values/values.sum()).astype(np.float32)


def collect_old(dataset, feature_function=None):
    infos=mmengine.load(dataset['selected_info'])['data_list']
    if len(infos)!=len(dataset['records']):
        raise ValueError('Incomplete old cache; all training frames must finish first')
    x,base,target,groups,sequences,classes=[],[],[],[],[],[]
    teacher,teacher_groups=[],[]
    for record,info in zip(dataset['records'],infos):
        if str(info['token'])!=record['token']:
            raise ValueError('Old GT/cache token mismatch')
        sequence=str(info.get('metainfo',{}).get('sequence_id','unknown'))
        with np.load(record['path'],allow_pickle=False) as row:
            features=lambda indices: (design(row,indices) if feature_function is None
                                     else feature_function(row,indices,record))
            tq=np.flatnonzero(row['scores']>=.1)
            if len(tq):
                teacher.append(features(tq))
                teacher_groups.extend('%s/%d'%(sequence,int(row['labels'][j])) for j in tq)
            ann=info['annos']
            gt=np.asarray(ann['gt_boxes'])
            names=np.asarray(ann['gt_names'])
            gi=[i for i,(b,n) in enumerate(zip(gt,names)) if str(n) in CLASSES[:4]
                and np.linalg.norm(b[:2])<48 and b[3]>=1.2*b[4]]
            qi=np.flatnonzero((row['scores']>=.05)&(row['labels']<4))
            if not gi or not len(qi):
                continue
            labels=np.array([CLASSES.index(str(names[i])) for i in gi])
            cost=np.linalg.norm(row['boxes'][qi,None,:2]-gt[None,gi,:2],axis=-1)
            cost[row['labels'][qi,None]!=labels[None]]=1e6
            cost[cost>=2]=1e6
            iq,ig=linear_sum_assignment(cost)
            keep=cost[iq,ig]<2
            iq,ig=qi[iq[keep]],np.asarray(gi)[ig[keep]]
            if not len(iq):
                continue
            x.append(features(iq));base.extend(row['boxes'][iq,6]);target.extend(gt[ig,6])
            sequences.extend([sequence]*len(iq));classes.extend(names[ig].tolist())
            groups.extend('%s/%s/%d'%(sequence,str(names[j]),int((gt[j,6]%np.pi)/(np.pi/12))) for j in ig)
    if not x or not teacher:
        raise ValueError('No old supervision/teacher')
    return dict(x=np.concatenate(x),base=np.asarray(base,dtype=np.float32),target=np.asarray(target,dtype=np.float32),
        sequences=np.asarray(sequences),classes=np.asarray(classes),sampling=stratified_weights(groups),
        teacher=np.concatenate(teacher),teacher_sampling=stratified_weights(teacher_groups))


def preservation_gate(baseline, candidate, baseline_angles, candidate_angles,
                      nds_tolerance=.002, map_relative_tolerance=.01, angle_tolerance=1.):
    failed=[]
    if candidate['NDS']<baseline['NDS']-nds_tolerance:
        failed.append('old_NDS')
    if candidate['mAP']<baseline['mAP']*(1-map_relative_tolerance):
        failed.append('old_mAP')
    for group,value in baseline_angles.items():
        if candidate_angles[group]>value+angle_tolerance:
            failed.append('old_axis_'+group)
    # NDS averaging must not hide damage to a minority class.
    for key,value in baseline.items():
        if key.endswith('_AOE') and key in candidate and candidate[key]>value+.03:
            failed.append('old_'+key)
    return not failed,failed


def grouped_angles(data, delta):
    predicted=data['base']+data['x']@delta
    error=np.degrees(np.abs((predicted-data['target']+np.pi/2)%np.pi-np.pi/2))
    result={'all':float(error.mean())}
    for field in ('sequences','classes'):
        for group in np.unique(data[field]):
            mask=data[field]==group
            if mask.sum()>=20:
                result[str(group)]=float(error[mask].mean())
    return result


def save_checkpoint(manifest,delta,out,report):
    parent=Path(manifest['checkpoint'])
    if sha(parent)!=manifest['checkpoint_sha256']:
        raise ValueError('Parent checkpoint changed')
    checkpoint=torch.load(parent,map_location='cpu',weights_only=False)
    state=checkpoint['state_dict']
    prefix='head.fisheye_box_decoder.layers.5.refinement.'
    changes=[(prefix+'trunk.3.weight',6,delta[:256]),(prefix+'trunk.3.bias',6,delta[256]),
             (prefix+'face_refinement.weight',3,delta[257:])]
    for key,row,value in changes:
        original=state[key].clone()
        state[key][row]+=torch.as_tensor(value,dtype=state[key].dtype)
        mask=torch.ones(len(original),dtype=torch.bool);mask[row]=False
        if not torch.equal(state[key][mask],original[mask]):
            raise RuntimeError('Non-yaw row changed')
    for key in ('optimizer','param_schedulers','message_hub'):
        checkpoint.pop(key,None)
    checkpoint.setdefault('meta',{})['heading_adaptation']=dict(experiment='E009_mixed_rehearsal',
        parent_sha256=manifest['checkpoint_sha256'],only_changed_keys=[key for key,_,_ in changes],
        optimized_scalar_coefficients=1025,selected_pass=report['best_pass'],temporal_slots=0)
    destination=out/'best_guarded_heading.pth'
    torch.save(checkpoint,destination)
    np.savez(out/'heading_delta.npz',delta=delta)
    report.update(checkpoint=str(destination),checkpoint_sha256=sha(destination))
    profile=json.loads((Path(__file__).resolve().parents[1]/'work_dirs/threshold_tuning_e1_20261103/balanced_profile.json').read_text())
    profile.update(checkpoint_sha256=report['checkpoint_sha256'],note='Inherited E007 thresholds; not retuned.')
    (out/'threshold_profile.json').write_text(json.dumps(profile,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--targets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--passes',type=int,default=40)
    p.add_argument('--steps-per-pass',type=int,default=32)
    p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--distill-weight',type=float,default=1.)
    args=p.parse_args()
    torch.manual_seed(3407);np.random.seed(3407)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'training_report.json').exists():
        raise FileExistsError('Use a new output directory')
    manifest=json.loads((args.cache/'cache_manifest.json').read_text())
    targets=json.loads(args.targets.read_text())
    oldset=manifest['datasets']['old_train'];valset=manifest['datasets']['old_val']
    for dataset in (oldset,valset):
        if sha(Path(dataset['source_info']))!=dataset['source_info_sha256']:
            raise ValueError('Old source annotations changed')
    if set(r['token'] for r in oldset['records'])&set(r['token'] for r in valset['records']):
        raise ValueError('Old train/validation overlap')
    print('Loading all old GT and frozen teacher features',flush=True)
    old=collect_old(oldset);guard=collect_old(valset)
    new=collect_supervision(manifest['datasets']['new']['records'],targets['frames_data'],'train')
    val=collect_supervision(manifest['datasets']['new']['records'],targets['frames_data'],'val')
    oldinfos=mmengine.load(valset['selected_info'])['data_list']
    zero=np.zeros(1025,dtype=np.float32)
    print('Computing full old-validation baseline',flush=True)
    baseline_metrics=official(cached_predictions(valset['records'],zero),oldinfos)
    baseline_angles=grouped_angles(guard,zero)
    tensors={}
    for name,data,keys in [('old',old,('x','base','target','sampling','teacher','teacher_sampling')),
                           ('new',new,('x','base','target','directed','weight'))]:
        tensors[name]={key:torch.as_tensor(data[key],device='cuda') for key in keys}
    delta=torch.nn.Parameter(torch.zeros(1025,device='cuda'))
    optimizer=torch.optim.Adam([delta],lr=args.lr)
    report=dict(experiment='E009_mixed_rehearsal',parent=manifest['checkpoint'],parent_sha256=manifest['checkpoint_sha256'],
        cache_sha256=sha(args.cache/'cache_manifest.json'),targets_sha256=sha(args.targets),
        old_training_frames=len(oldset['records']),old_supervised_queries=len(old['x']),old_teacher_queries=len(old['teacher']),
        new_training_queries=len(new['x']),new_moving_ego_frames=targets['split_summary']['train']['moving_ego_frames'],
        training_mix=dict(old=192,new=64),learning_rate=args.lr,distill_weight=args.distill_weight,
        baseline_old_metrics=baseline_metrics,baseline_old_axis=baseline_angles,
        baseline_new=angular_metrics(val['base'],val['target']),history=[],best_pass=0,
        selection='New heading improvement only if old NDS/mAP/per-sequence and per-class axis and teacher drift gates pass',
        caveat='Cached frozen native AMP features, 1025 yaw coefficients only; new targets are partial pseudo-labels.')
    (out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    best=zero.copy();best_metric=report['baseline_new']['axis_mean_deg'];wait=0
    for epoch in range(1,args.passes+1):
        started=time.monotonic();losses=[]
        for step in range(args.steps_per_pass):
            ot,nt=tensors['old'],tensors['new']
            oi=torch.multinomial(ot['sampling'],192,replacement=True)
            ni=torch.randint(len(nt['x']),(64,),device='cuda')
            ti=torch.multinomial(ot['teacher_sampling'],192,replacement=True)
            od=ot['base'][oi]+ot['x'][oi]@delta-ot['target'][oi]
            nd=nt['base'][ni]+nt['x'][ni]@delta-nt['target'][ni]
            old_loss=.5*(1-torch.cos(2*od)).mean()
            new_angles=torch.where(nt['directed'][ni],1-torch.cos(nd),.5*(1-torch.cos(2*nd)))
            new_loss=(new_angles*nt['weight'][ni]).sum()/nt['weight'][ni].sum()
            distill=(1-torch.cos(ot['teacher'][ti]@delta)).mean()
            loss=.75*old_loss+.25*new_loss+args.distill_weight*distill+.01*delta.square().sum()
            progress=(epoch-1)*args.steps_per_pass+step+1
            warmup=min(1.,progress/(3*args.steps_per_pass))
            optimizer.param_groups[0]['lr']=args.lr*warmup
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([delta],1.);optimizer.step()
            losses.append(float(loss.detach()))
        vector=delta.detach().cpu().numpy()
        new_metrics=angular_metrics(val['base']+val['x']@vector,val['target'])
        axes=grouped_angles(guard,vector)
        old_metrics=official(cached_predictions(valset['records'],vector),oldinfos)
        accepted,failures=preservation_gate(baseline_metrics,old_metrics,baseline_angles,axes)
        drift=np.degrees(np.abs((guard['teacher']@vector+np.pi)%(2*np.pi)-np.pi))
        if drift.mean()>3 or np.percentile(drift,95)>10:
            accepted=False;failures.append('teacher_drift')
        improved=accepted and new_metrics['axis_mean_deg']<best_metric-.05
        row=dict(pass_number=epoch,mean_loss=float(np.mean(losses)),new_validation=new_metrics,
            old_metrics=old_metrics,old_axis=axes,guard_passed=accepted,failed_gates=failures,
            old_teacher_mean_drift_deg=float(drift.mean()),old_teacher_p95_drift_deg=float(np.percentile(drift,95)),
            seconds=time.monotonic()-started)
        report['history'].append(row)
        if improved:
            best=vector.copy();best_metric=new_metrics['axis_mean_deg'];report['best_pass']=epoch;wait=0
            report['selected_old_metrics']=old_metrics;report['selected_new_metrics']=new_metrics
            save_checkpoint(manifest,best,out,report)
        else:
            wait+=1
        report['completed_passes']=epoch
        report['status']='training'
        (out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(pass_number=epoch,new_axis_mean=new_metrics['axis_mean_deg'],old_mAP=old_metrics['mAP'],
            old_NDS=old_metrics['NDS'],guard_passed=accepted,failed_gates=failures,best_pass=report['best_pass'],
            patience_used=wait,seconds=row['seconds'])),flush=True)
        if wait>=args.patience:
            break
    report.update(status='complete',stop_reason='guarded plateau' if wait>=args.patience else 'pass cap',
        best_new_axis_mean=best_metric,promotion_status='REQUIRES_FRESH_NATIVE_VALIDATION' if report['best_pass'] else 'NO_ACCEPTABLE_IMPROVEMENT_KEEP_PARENT')
    if not report['best_pass']:
        save_checkpoint(manifest,zero,out,report)
    (out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Finished',report['promotion_status'],flush=True)


if __name__=='__main__':
    main()

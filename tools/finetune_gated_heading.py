#!/usr/bin/env python3
"""Train existing final yaw rows AND their context gates, with old-data guards.

No new architecture: exactly 1,539 scalar coefficients in seven existing tensors.
Wrapped angular Huber loss avoids the cosine axial loss's zero gradient at 90°.
All positional/class/size weights, backbone, BN and earlier stages stay frozen.
"""
import argparse
import json
from pathlib import Path
import time

from train_balanced_fisheye import np,torch
import mmengine
from finetune_heading_rows import sha,collect_supervision,angular_metrics
from finetune_mixed_heading import collect_old,preservation_gate
from evaluate_heading_adaptation import cached_predictions,official

PREFIX='head.fisheye_box_decoder.layers.5.refinement.'


def wrapped_huber(difference,directed=None,beta=.2):
    axial=.5*torch.atan2(torch.sin(2*difference),torch.cos(2*difference))
    if directed is not None:
        full=torch.atan2(torch.sin(difference),torch.cos(difference))
        axial=torch.where(directed,full,axial)
    absolute=axial.abs()
    return torch.where(absolute<beta,.5*absolute.square()/beta,absolute-.5*beta)


class GatedYaw(torch.nn.Module):
    def __init__(self,state):
        super().__init__()
        self.register_buffer('trunk',state[PREFIX+'trunk.3.weight'][6].float().clone())
        self.register_buffer('bias',state[PREFIX+'trunk.3.bias'][6].float().clone())
        self.register_buffer('face',state[PREFIX+'face_refinement.weight'][3].float().clone())
        self.delta=torch.nn.Parameter(torch.zeros(1539))

    def forward(self,features):
        h,f,r=features[:,:256],features[:,256:1024],features[:,1024:1280]
        yg,fg=features[:,1280].clamp(1e-6,1-1e-6),features[:,1281].clamp(1e-6,1-1e-6)
        d=self.delta
        a=h@self.trunk+self.bias;b=f@self.face
        # Use the recorded parent gates, preserving a zero residual exactly.
        gy=torch.sigmoid(torch.logit(yg)+r@d[1025:1281]+d[1281])
        gf=torch.sigmoid(torch.logit(fg)+r@d[1282:1538]+d[1538])
        return (a+h@d[:256]+d[256])*gy-a*yg+(b+f@d[257:1025])*gf-b*fg


class ContextFeatures:
    def __init__(self,document):
        if document.get('status')!='complete':raise ValueError('Context caching incomplete')
        self.paths={}
        for dataset in document['datasets'].values():
            for row in dataset['records']:
                if row['token'] in self.paths:raise ValueError('Duplicate context token')
                self.paths[row['token']]=row['path']

    def __call__(self,row,indices,record):
        with np.load(self.paths[record['token']],allow_pickle=False) as context:
            if str(context['token'])!=record['token']:raise ValueError('Context token differs')
            return np.concatenate((row['hidden'][indices],row['faces'][indices],context['context'][indices],
                row['yaw_gate'][indices,None],row['face_gate'][indices,None]),axis=1).astype(np.float32)


def changed_rows(delta):
    return [('trunk.3.weight',6,delta[:256]),('trunk.3.bias',6,delta[256]),
            ('face_refinement.weight',3,delta[257:1025]),('update_gate.weight',2,delta[1025:1281]),
            ('update_gate.bias',2,delta[1281]),('face_yaw_gate.weight',0,delta[1282:1538]),
            ('face_yaw_gate.bias',0,delta[1538])]


def save_checkpoint(manifest,model,out,report):
    if sha(Path(manifest['checkpoint']))!=manifest['checkpoint_sha256']:raise ValueError('Parent changed')
    checkpoint=torch.load(manifest['checkpoint'],map_location='cpu',weights_only=False)
    delta=model.delta.detach().cpu();allowed={}
    for suffix,row,value in changed_rows(delta):
        key=PREFIX+suffix;before=checkpoint['state_dict'][key].clone()
        checkpoint['state_dict'][key][row]+=value.to(before.dtype)
        allowed[key]=row
        mask=torch.ones(len(before),dtype=torch.bool);mask[row]=False
        if not torch.equal(checkpoint['state_dict'][key][mask],before[mask]):raise RuntimeError('Non-yaw row changed')
    for key in ('optimizer','param_schedulers','message_hub'):checkpoint.pop(key,None)
    checkpoint.setdefault('meta',{})['heading_adaptation']=dict(experiment='E010_gated_heading',
        parent_sha256=manifest['checkpoint_sha256'],only_changed_rows=allowed,optimized_scalar_coefficients=1539,
        selected_pass=report['best_pass'],temporal_slots=0)
    destination=out/'best_gated_heading.pth';torch.save(checkpoint,destination)
    np.savez(out/'gated_delta.npz',delta=delta.numpy())
    report.update(checkpoint=str(destination),checkpoint_sha256=sha(destination))
    profile_path=Path(__file__).resolve().parents[1]/'work_dirs/threshold_tuning_e1_20261103/balanced_profile.json'
    profile=json.loads(profile_path.read_text());profile.update(checkpoint_sha256=report['checkpoint_sha256'],note='Inherited E007 thresholds, not retuned')
    (out/'threshold_profile.json').write_text(json.dumps(profile,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--context',type=Path,required=True)
    p.add_argument('--targets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--passes',type=int,default=80);p.add_argument('--steps-per-pass',type=int,default=64)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--patience',type=int,default=8)
    p.add_argument('--distill-weight',type=float,default=1.)
    args=p.parse_args();torch.manual_seed(3407);np.random.seed(3407)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'training_report.json').exists():raise FileExistsError('Use a new output directory')
    manifest=json.loads((args.cache/'cache_manifest.json').read_text())
    context=json.loads((args.context/'context_manifest.json').read_text())
    targets=json.loads(args.targets.read_text())
    if context['source_cache_sha256']!=sha(args.cache/'cache_manifest.json') or context['checkpoint_sha256']!=manifest['checkpoint_sha256']:
        raise ValueError('Context/feature cache provenance mismatch')
    oldset,valset=manifest['datasets']['old_train'],manifest['datasets']['old_val']
    if set(r['token'] for r in oldset['records'])&set(r['token'] for r in valset['records']):raise ValueError('Train/val overlap')
    for dataset in (oldset,valset):
        if sha(Path(dataset['source_info']))!=dataset['source_info_sha256']:raise ValueError('Annotations changed')
    features=ContextFeatures(context)
    print('Loading full old/new context and supervision',flush=True)
    old=collect_old(oldset,features);guard=collect_old(valset,features)
    new=collect_supervision(manifest['datasets']['new']['records'],targets['frames_data'],'train',features)
    val=collect_supervision(manifest['datasets']['new']['records'],targets['frames_data'],'val',features)
    parent=torch.load(manifest['checkpoint'],map_location='cpu',weights_only=False)
    model=GatedYaw(parent['state_dict']).cuda();del parent
    optimizer=torch.optim.Adam([model.delta],lr=args.lr)
    tensors={name:{k:torch.as_tensor(v,device='cuda') for k,v in data.items() if isinstance(v,np.ndarray) and v.dtype.kind not in 'UOS'}
        for name,data in [('old',old),('new',new),('val',val),('guard',guard)]}
    def correction(x):
        with torch.no_grad():
            return np.concatenate([model(torch.as_tensor(chunk,device='cuda')).cpu().numpy() for chunk in np.array_split(x,max(1,len(x)//2048))])
    def axis_groups(data,delta):
        error=np.degrees(np.abs((data['base']+delta-data['target']+np.pi/2)%np.pi-np.pi/2))
        groups=dict(all=float(error.mean()))
        for key in ('sequences','classes'):
            for value in np.unique(data[key]):
                mask=data[key]==value
                if mask.sum()>=20:groups[str(value)]=float(error[mask].mean())
        return groups
    oldinfos=mmengine.load(valset['selected_info'])['data_list']
    baseline=official(cached_predictions(valset['records'],np.zeros(1025,dtype=np.float32)),oldinfos)
    axes=axis_groups(guard,np.zeros(len(guard['base'])))
    initial=angular_metrics(val['base'],val['target'])
    report=dict(experiment='E010_gated_heading',parent=manifest['checkpoint'],parent_sha256=manifest['checkpoint_sha256'],
        cache_sha256=sha(args.cache/'cache_manifest.json'),context_sha256=sha(args.context/'context_manifest.json'),
        targets_sha256=sha(args.targets),optimized_scalar_coefficients=1539,old_training_frames=len(oldset['records']),
        old_supervised_queries=len(old['x']),old_teacher_queries=len(old['teacher']),new_training_queries=len(new['x']),
        moving_ego_frames_in_training=targets['split_summary']['train']['moving_ego_frames'],
        method='Existing yaw rows and both context gates; wrapped angular Huber; 75% old / 25% new; old TRAIN teacher distillation',
        learning_rate=args.lr,distill_weight=args.distill_weight,baseline_old=baseline,baseline_new=initial,
        best_pass=0,history=[],status='training',limitations=targets.get('caveats',[]))
    best=initial['axis_mean_deg'];wait=0
    for epoch in range(1,args.passes+1):
        started=time.monotonic();losses=[]
        for step in range(args.steps_per_pass):
            ot,nt=tensors['old'],tensors['new']
            oi=torch.multinomial(ot['sampling'],192,True);ni=torch.randint(len(nt['x']),(64,),device='cuda')
            ti=torch.multinomial(ot['teacher_sampling'],192,True)
            od=ot['base'][oi]+model(ot['x'][oi])-ot['target'][oi]
            nd=nt['base'][ni]+model(nt['x'][ni])-nt['target'][ni]
            old_loss=wrapped_huber(od).mean()
            new_loss=(wrapped_huber(nd,nt['directed'][ni])*nt['weight'][ni]).sum()/nt['weight'][ni].sum()
            teacher=(1-torch.cos(model(ot['teacher'][ti]))).mean()
            loss=.75*old_loss+.25*new_loss+args.distill_weight*teacher+.01*model.delta.square().sum()
            optimizer.param_groups[0]['lr']=args.lr*min(1.,((epoch-1)*args.steps_per_pass+step+1)/(3*args.steps_per_pass))
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([model.delta],1.);optimizer.step()
            losses.append(float(loss.detach()))
        new_metrics=angular_metrics(val['base']+correction(val['x']),val['target'])
        old_angles=axis_groups(guard,correction(guard['x']))
        residual=lambda row,indices,record:correction(features(row,indices,record))
        metrics=official(cached_predictions(valset['records'],None,residual_function=residual),oldinfos)
        passed,failed=preservation_gate(baseline,metrics,axes,old_angles)
        drift=np.degrees(np.abs((correction(guard['teacher'])+np.pi)%(2*np.pi)-np.pi))
        if drift.mean()>3 or np.percentile(drift,95)>10:passed=False;failed.append('teacher_drift')
        row=dict(pass_number=epoch,loss=float(np.mean(losses)),new_validation=new_metrics,old_metrics=metrics,
            old_axis=old_angles,guard_passed=passed,failed_gates=failed,teacher_mean_deg=float(drift.mean()),
            teacher_p95_deg=float(np.percentile(drift,95)),seconds=time.monotonic()-started)
        report['history'].append(row)
        if passed and new_metrics['axis_mean_deg']<best-.05:
            best=new_metrics['axis_mean_deg'];report.update(best_pass=epoch,selected_old_metrics=metrics,selected_new_metrics=new_metrics)
            save_checkpoint(manifest,model,out,report);wait=0
        else:wait+=1
        report.update(completed_passes=epoch,patience_used=wait)
        (out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(pass_number=epoch,new_axis_mean=new_metrics['axis_mean_deg'],old_mAP=metrics['mAP'],
            old_NDS=metrics['NDS'],guard_passed=passed,failed_gates=failed,best_pass=report['best_pass'],seconds=row['seconds'])),flush=True)
        if wait>=args.patience:break
    report.update(status='complete',stop_reason='guarded plateau' if wait>=args.patience else 'pass cap',
        promotion_status='REQUIRES_FRESH_NATIVE_VALIDATION' if report['best_pass'] else 'NO_ACCEPTABLE_IMPROVEMENT_KEEP_PARENT')
    (out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Synthetic border/mask/calibration parity checks for the opt-in CUDA library."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROSA=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROSA),str(ROSA/'work_dirs/onnx_deps_py38')]
import numpy as np
import onnx
from onnx import helper,TensorProto
import torch
import torch.nn.functional as F
from deployment.fused_head import custom,DOMAIN
from deployment.runtime import create_session
from benchmark_fisheye_backends import sha


def run(kind,inputs,outputs,library,**attributes):
    ishapes=[list(x.shape) for x in inputs]
    oshapes=[list(x.shape) for x in outputs]
    def vi(name,x):return helper.make_tensor_value_info(name,
        TensorProto.BOOL if x.dtype==np.bool_ else TensorProto.FLOAT,list(x.shape))
    names=['x'+str(i) for i in range(len(inputs))]
    out_names=['y'+str(i) for i in range(len(outputs))]
    op=custom('test',kind,names,out_names,ishapes,oshapes)
    op.attribute.extend(helper.make_attribute(k,v) for k,v in attributes.items())
    graph=helper.make_graph([op],
        kind,[vi(n,x) for n,x in zip(names,inputs)],[vi(n,x) for n,x in zip(out_names,outputs)])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17),helper.make_opsetid(DOMAIN,1)],ir_version=8)
    with tempfile.TemporaryDirectory(prefix='rosa_fused_test_') as folder:
        path=Path(folder)/'test.onnx';onnx.save(model,str(path))
        path.with_suffix('.json').write_text(json.dumps(dict(custom_ops=dict(library_sha256=sha(library)))))
        session=create_session(path,custom_op_library=library)
        result=session.run(None,dict(zip(names,inputs)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True,help='Real calibration NPZ')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rng=np.random.RandomState(20261108)
    rows={}
    # Include UV0/1, half-pixel fringes, fully outside and random locations.
    n=31
    features=[rng.normal(size=(4,256,h,w)).astype(np.float32) for h,w in ((17,19),(9,11),(5,7),(3,4))]
    grid=rng.uniform(-1.4,1.4,size=(4,n,1,2)).astype(np.float32)
    for i,g in enumerate(((-1,-1),(1,1),(-1,1),(1,-1),(0,0),(-1.01,0),(1.01,0))):grid[:,i,0]=g
    weights=rng.uniform(0,1,size=(1,n,1,4,4,8)).astype(np.float32)
    weights/=weights.sum(axis=4,keepdims=True)
    reference=np.zeros((1,n,4,256),np.float32)
    for l,f in enumerate(features):
        sampled=F.grid_sample(torch.from_numpy(f),torch.from_numpy(grid),align_corners=False).numpy()
        sampled=sampled.reshape(4,256,n).transpose(2,0,1).reshape(1,n,4,8,32)
        reference+=(sampled*weights[:,:,0,:,l,:,None]).reshape(1,n,4,256)
    actual=run('SampleLevels',features+[grid,weights],[reference],args.library)[0]
    np.testing.assert_allclose(actual,reference,rtol=2e-5,atol=5e-6)
    rows['sampling']=dict(max_abs=float(np.abs(actual-reference).max()),border_cases=7)
    # Channel-last packing is once per image and shared by all six stages.
    nhwc=[np.ascontiguousarray(f.transpose(0,2,3,1)) for f in features]
    actual=run('SampleLevels',nhwc+[grid,weights],[reference],args.library,layout=1)[0]
    np.testing.assert_allclose(actual,reference,rtol=2e-5,atol=5e-6)
    rows['sampling_nhwc']=dict(max_abs=float(np.abs(actual-reference).max()),border_cases=7)
    try:
        run('SampleLevels',[np.ascontiguousarray(features[0][:,:255])]+features[1:]+[grid,weights],
            [reference],args.library)
    except Exception as error:
        if 'Sampling requires' not in str(error):raise
    else:raise AssertionError('An unsupported feature width was accepted')
    rows['sampling']['unsupported_shape_rejected']=True

    q=rng.normal(size=(1,5,8,8)).astype(np.float32)
    k=rng.normal(size=(1,5,13,8,8)).astype(np.float32)
    reliability=rng.normal(size=(1,5,13,8)).astype(np.float32)
    values=rng.normal(size=(1,5,13,8,32)).astype(np.float32)
    valid=rng.uniform(size=(1,5,13))>.4;valid[:,0]=False;valid[:,1]=True
    logits=np.einsum('bqhd,bqphd->bqph',q,k)/np.sqrt(np.float32(8))+reliability
    logits=np.where(valid[...,None],logits,-1e30)
    prob=np.exp(logits-logits.max(axis=2,keepdims=True))*valid[...,None]
    prob/=np.maximum(prob.sum(axis=2,keepdims=True),1e-8)
    evidence=np.einsum('bqph,bqphd->bqhd',prob,values)
    actual=run('PointAttention',[q,k,reliability,values,valid],[prob,evidence],args.library)
    for a,b in zip(actual,(prob,evidence)):np.testing.assert_allclose(a,b,rtol=2e-5,atol=3e-6)
    assert not actual[0][:,0].any() and not actual[1][:,0].any()
    rows['attention']=dict(max_abs=max(float(np.abs(a-b).max()) for a,b in zip(actual,(prob,evidence))),all_invalid_zero=True)

    from rosa.fisheye_projection import NativeFisheyeProjector
    from rosa.fisheye_evidence import angular_fpn_geometry
    from rosa.ray_state import normalize_direction
    with np.load(args.input) as source:
        metadata={name:torch.from_numpy(source[name]) for name in ('lidar2cam','ocam_intrinsic','ocam_pol','image_wh','ocam_lens_fov_deg')}
    points=torch.from_numpy(rng.normal(size=(1,260,3)).astype(np.float32)*20)
    extent=torch.from_numpy(rng.uniform(-4,-1,size=(1,260,2)).astype(np.float32))
    strides=torch.tensor([4.,8.,16.,32.])
    with torch.no_grad():
        projection=NativeFisheyeProjector()(normalize_direction(points),metadata,points=points)
        g=projection.geometry
        descriptor,compatibility=angular_fpn_geometry(g.tangent,g.jacobian,g.lens_margin,g.overlap,g.valid,strides,extent)
    reference=[projection.uv.numpy(),g.rays.numpy(),g.valid.numpy(),descriptor.numpy(),compatibility.numpy()]
    inputs=[points.numpy(),metadata['lidar2cam'].numpy(),metadata['image_wh'].numpy(),metadata['ocam_intrinsic'].numpy(),
        metadata['ocam_pol'].numpy(),torch.deg2rad(metadata['ocam_lens_fov_deg']*.5).numpy(),extent.numpy(),strides.numpy()]
    actual=run('ProjectGeometry',inputs,reference,args.library)
    assert np.array_equal(actual[2],reference[2]),'Projection visibility changed'
    differences=[float(np.abs(a-b).max()) for i,(a,b) in enumerate(zip(actual,reference)) if i!=2]
    for i,(a,b) in enumerate(zip(actual,reference)):
        if i!=2:np.testing.assert_allclose(a,b,rtol=2e-3,atol=3e-3)
    rows['projection']=dict(uv_ray_descriptor_compatibility_max_abs=differences,visibility_identical=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=True,library_sha256=sha(args.library),checks=rows),indent=2)+'\n')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()

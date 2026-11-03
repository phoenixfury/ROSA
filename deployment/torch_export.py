"""Export-local equivalents of CUDA feature readers; learned RFL is retained."""
from contextlib import contextmanager, ExitStack
import os
from unittest.mock import patch

import torch
import torch.nn.functional as F


def tensor_feature_maps_format(features, inverse=False):
    if inverse:
        raise ValueError('Inverse packing is not used by this single-frame export')
    return tuple(features), None, None


def tensor_aggregation_per_camera(features, spatial_shape, starts, locations, weights):
    """Equivalent bilinear reading and grouped level/keypoint reduction.

    features: L x [B,C,D,H,W], locations: [B,N,K,C,2] in [0,1],
    weights: [B,N,K,C,L,heads]. The caller retains all ray-conditioned camera
    fusion, learned geometry attention, semantic point attention and refinement.
    """
    batch, queries, keypoints, cameras = locations.shape[:4]
    channels, heads = features[0].shape[2], weights.shape[-1]
    grid = locations.permute(0, 3, 1, 2, 4).reshape(batch*cameras, queries, keypoints, 2)*2-1
    per_camera = None
    for level, feature in enumerate(features):
        height, width = feature.shape[-2:]
        sampled = F.grid_sample(feature.float().reshape(batch*cameras, channels, height, width),
                                grid.float(), mode='bilinear', padding_mode='zeros', align_corners=False)
        sampled = sampled.reshape(batch, cameras, channels, queries, keypoints).permute(0, 3, 4, 1, 2)
        sampled = sampled.reshape(batch, queries, keypoints, cameras, heads, channels//heads)
        weighted = (sampled * weights[..., level, :, None].float()).sum(dim=2)
        per_camera = weighted if per_camera is None else per_camera + weighted
    per_camera = per_camera.reshape(batch, queries, cameras, channels)
    return per_camera.sum(dim=2), per_camera


def tensor_aggregation(*args):
    return tensor_aggregation_per_camera(*args)[0]


@contextmanager
def portable_cuda_ops():
    import projects.mmdet3d_plugin.ops as host
    flags = ('ROSA_DISABLE_FUSED_PROJECTION', 'ROSA_DISABLE_FUSED_STATE_KEYPOINTS',
             'ROSA_DISABLE_FUSED_STATE_UPDATE', 'ROSA_DISABLE_FUSED_POINT_ATTENTION')
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {name: '1' for name in flags}))
        stack.enter_context(patch.object(host, 'feature_maps_format', tensor_feature_maps_format))
        stack.enter_context(patch.object(host, 'deformable_aggregation_function', tensor_aggregation))
        stack.enter_context(patch.object(host, 'deformable_aggregation_per_camera_function', tensor_aggregation_per_camera))
        yield

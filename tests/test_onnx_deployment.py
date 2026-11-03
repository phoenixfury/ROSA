"""Box-coordinate/NMS/parity tests; no neural network or GPU needed."""
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

ROSA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROSA))
from deployment.runtime import compare_outputs, convert_box_origin, postprocess, rotated_nms


def test_bottom_to_center_preserves_axes_dimensions_yaw_and_caller():
    boxes = np.array([[4., 2., -1., 4., 2., 1.5, np.pi/2]], dtype=np.float32)
    result = convert_box_origin(boxes, 'bottom_center', 'gravity_center')
    assert result[0, 2] == pytest.approx(-.25)
    np.testing.assert_array_equal(result[0, [0, 1, 3, 4, 5, 6]], boxes[0, [0, 1, 3, 4, 5, 6]])
    assert boxes[0, 2] == -1.
    np.testing.assert_array_equal(convert_box_origin(result, 'gravity_center', 'bottom_center'), boxes)


def test_unknown_origin_rejected():
    with pytest.raises(ValueError):
        convert_box_origin(np.zeros((0, 7)), source='unknown')


def test_nms_only_suppresses_same_class_and_keeps_highest_score():
    boxes = np.array([[4, 0, 0, 4, 2, 2, 0]]*3, dtype=np.float32)
    result = postprocess(boxes, [.8, .9, .7], [0, 0, 3])
    assert result['labels'] == [0, 3]
    assert result['scores'] == [.9, .7]


def test_optional_distance_is_xy_and_disabled_by_default():
    boxes = np.array([[40, 40, 100, 4, 2, 2, 0], [61, 0, 0, 4, 2, 2, 0]], dtype=np.float32)
    assert len(postprocess(boxes, [.9, .8], [0, 0])['scores']) == 2
    result = postprocess(boxes, [.9, .8], [0, 0], max_distance=60)
    assert len(result['scores']) == 1
    assert result['boxes_3d'][0][2] == 100


def test_empty_outputs_are_valid():
    assert postprocess(np.empty((0, 7)), [], [])['boxes_3d'] == []


def test_class_score_profile_overrides_scalar_without_changing_nms():
    from deployment.runtime import CLASSES
    boxes = np.array([[4, 0, 0, 4, 2, 2, 0]]*3, dtype=np.float32)
    limits = dict.fromkeys(CLASSES, .1)
    limits['Car'] = .3
    result = postprocess(boxes, [.2, .4, .2], [0, 0, 3], class_score_thresholds=limits)
    assert result['labels'] == [0, 3]
    assert result['scores'] == [.4, .2]
    with pytest.raises(ValueError):
        postprocess(boxes, [.2]*3, [0]*3, class_score_thresholds={'Car': .1})
    limits['Car'] = float('nan')
    with pytest.raises(ValueError):
        postprocess(boxes, [.2]*3, [0]*3, class_score_thresholds=limits)


def test_amp_score_threshold_is_not_rounded_down_to_half_precision():
    box = np.array([[4, 0, 0, 4, 2, 2, 0]], dtype=np.float32)
    result = postprocess(box, np.array([.1], dtype=np.float16), [0], score_threshold=.1)
    assert result['scores'] == []  # float16(.1) is actually 0.0999755859375.


@pytest.mark.parametrize('box', [[0, 0, float('nan'), 1, 1, 1, 0], [0, 0, 0, 1, -1, 1, 0]])
def test_invalid_box_is_not_silently_exported(box):
    with pytest.raises(ValueError):
        postprocess([box], [.8], [0])


def test_yaw_parity_wraps_at_pi():
    reference = dict(boxes=np.array([[[4., 0., 1., 4., 2., 2., np.pi-1e-6]]]),
                     scores=np.array([[.7]]), labels=np.array([[0]]))
    actual = [value.copy() for value in reference.values()]
    actual[0][..., 6] = -np.pi+1e-6
    assert compare_outputs(actual, reference)['passed']
    actual[2][0, 0] = 3
    assert not compare_outputs(actual, reference)['passed']


def test_rotated_nms_matches_native_mmcv():
    torch = pytest.importorskip('torch')
    from mmcv.ops import nms_rotated
    random = np.random.RandomState(3407)
    boxes = np.concatenate([random.uniform(-3, 3, (100, 3)),
        random.uniform(1, 5, (100, 3)), random.uniform(-np.pi, np.pi, (100, 1))], axis=1).astype(np.float32)
    scores = random.uniform(0, 1, 100).astype(np.float32)
    labels = random.randint(0, 3, 100)
    native = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        _, keep = nms_rotated(torch.from_numpy(boxes[members][:, [0, 1, 3, 4, 6]].copy()),
                             torch.from_numpy(scores[members]), .5, clockwise=False)
        native.extend(members[keep.numpy()].tolist())
    native = np.array(native)
    native = native[np.argsort(-scores[native])]
    np.testing.assert_array_equal(rotated_nms(boxes, scores, labels), native)


def test_cpu_runtime_import_does_not_import_torch_or_mmdetection():
    command = [sys.executable, '-c', 'import sys; '
        f'sys.path.insert(0,{str(ROSA)!r}); '
        'import deployment.runtime; '
        'assert "torch" not in sys.modules; assert "mmdet3d" not in sys.modules']
    subprocess.run(command, check=True)


def test_tensor_reader_retains_camera_and_head_channels():
    torch = pytest.importorskip('torch')
    from deployment.torch_export import tensor_aggregation_per_camera
    feature = torch.ones(1, 2, 4, 2, 2)
    feature[:, 1] *= 2
    locations = torch.full((1, 3, 1, 2, 2), .5)
    weights = torch.ones(1, 3, 1, 2, 1, 2)
    fused, cameras = tensor_aggregation_per_camera([feature], None, None, locations, weights)
    torch.testing.assert_close(cameras[:, :, 0], torch.ones(1, 3, 4))
    torch.testing.assert_close(cameras[:, :, 1], torch.full((1, 3, 4), 2.))
    torch.testing.assert_close(fused, torch.full((1, 3, 4), 3.))


@pytest.mark.skipif(os.environ.get('ROSA_DEPLOY_CUDA_TESTS') != '1', reason='Explicit NVIDIA test opt-in')
def test_tensor_reader_matches_native_cuda():
    sys.path.insert(0, str(ROSA / 'tools'))
    from train_balanced_fisheye import torch
    from deployment.torch_export import tensor_aggregation_per_camera
    from projects.mmdet3d_plugin.ops import feature_maps_format, deformable_aggregation_per_camera_function
    torch.manual_seed(3407)
    features = [torch.randn(1, 4, 32, 12, 16, device='cuda'), torch.randn(1, 4, 32, 6, 8, device='cuda')]
    locations = torch.rand(1, 19, 1, 4, 2, device='cuda')
    weights = torch.rand(1, 19, 1, 4, 2, 8, device='cuda')
    native = deformable_aggregation_per_camera_function(*feature_maps_format(features), locations, weights)
    portable = tensor_aggregation_per_camera(features, None, None, locations, weights)
    for actual, reference in zip(portable, native):
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)

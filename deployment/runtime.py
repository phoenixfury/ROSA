"""NumPy/OpenCV postprocessing for single-frame ROSA ONNX outputs.

The graph returns geometric-center boxes in CLIDAR FLU. Rotated BEV NMS is
explicit host postprocessing, not silently dropped or replaced with 2D NMS.
No OpenMMLab imports are required; PyTorch is optional CUDA-library preload only.
"""
import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ('Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist')
CAMERAS = ('fisheye_camera_front', 'fisheye_camera_left',
           'fisheye_camera_right', 'fisheye_camera_rear')
INPUT_NAMES = ('images', 'lidar2cam', 'ocam_intrinsic', 'ocam_pol', 'image_wh', 'ocam_lens_fov_deg')
OUTPUT_NAMES = ('boxes', 'scores', 'labels')
COORDINATES = dict(frame='CLIDAR', axes=dict(x='forward', y='left', z='up'),
                   handedness='right', box_order=['x', 'y', 'z', 'length', 'width', 'height', 'yaw'],
                   length_units='meters', yaw_units='radians',
                   yaw_convention='0 along +x; positive about +z toward +y')


def convert_box_origin(boxes, source='gravity_center', target='gravity_center'):
    allowed = ('gravity_center', 'bottom_center')
    if source not in allowed or target not in allowed:
        raise ValueError('Box origin must explicitly be gravity_center or bottom_center')
    value = np.asarray(boxes, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] < 7:
        raise ValueError('Boxes must be [N,7+] in x,y,z,length,width,height,yaw order')
    if not np.isfinite(value).all() or np.any(value[:, 3:6] <= 0):
        raise ValueError('Non-finite boxes or nonpositive dimensions')
    result = value.copy()
    if source != target:
        result[:, 2] += (0.5 if source == 'bottom_center' else -0.5) * result[:, 5]
    return result


def _polygon(box):
    local = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]], dtype=np.float32)
    # Reproduce this detector's MMCV nms_rotated(clockwise=False) footprint
    # exactly (verified against its CPU and CUDA kernels). Returned physical
    # box yaw is NOT changed; this sign is only the native NMS convention.
    cosine, sine = np.cos(-box[6]), np.sin(-box[6])
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    return np.ascontiguousarray((local * box[3:5]) @ rotation.T + box[:2])


def rotated_nms(boxes, scores, labels, iou_threshold=.5):
    """Class-wise rotated NMS matching the native decoder's angle convention."""
    if not 0 < iou_threshold <= 1:
        raise ValueError('NMS IoU threshold must be in (0,1]')
    boxes = np.asarray(boxes, dtype=np.float32)
    scores, labels = np.asarray(scores), np.asarray(labels)
    order = np.argsort(-scores, kind='stable')
    polygons = [_polygon(box) for box in boxes]
    areas = boxes[:, 3] * boxes[:, 4]
    radius = np.linalg.norm(boxes[:, 3:5], axis=-1) / 2
    keep = []
    for index in order:
        suppressed = False
        for other in keep:
            if labels[index] != labels[other]:
                continue
            if np.linalg.norm(boxes[index, :2] - boxes[other, :2]) > radius[index] + radius[other]:
                continue
            intersection = max(0., float(cv2.intersectConvexConvex(polygons[index], polygons[other])[0]))
            union = float(areas[index] + areas[other]) - intersection
            if intersection / max(union, 1e-9) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(int(index))
    return np.asarray(keep, dtype=np.int64)


def postprocess(boxes, scores, labels, score_threshold=.1, nms_iou=.5,
                max_distance=None, origin='gravity_center', class_score_thresholds=None):
    boxes = np.asarray(boxes).reshape(-1, 7)
    scores, labels = np.asarray(scores).reshape(-1), np.asarray(labels).reshape(-1)
    if len(boxes) != len(scores) or len(boxes) != len(labels):
        raise ValueError('Output box/score/label counts differ')
    if not 0 <= score_threshold <= 1 or (max_distance is not None and max_distance <= 0):
        raise ValueError('Invalid score threshold or distance limit')
    if not np.isfinite(scores).all() or np.any((labels < 0) | (labels >= len(CLASSES))):
        raise ValueError('Invalid model scores/classes')
    convert_box_origin(boxes)  # Validate, never silently discard malformed model outputs.
    limits = score_threshold
    if class_score_thresholds is not None:
        if set(class_score_thresholds) != set(CLASSES):
            raise ValueError('Class score thresholds must specify exactly the six model classes')
        limits = np.asarray([class_score_thresholds[name] for name in CLASSES], dtype=np.float64)
        if not np.isfinite(limits).all() or np.any((limits < 0) | (limits > 1)):
            raise ValueError('Invalid class score thresholds')
        limits = limits[labels.astype(np.int64)]
    # Do not round decimal cutoffs to float16 when consuming native AMP scores.
    keep = scores.astype(np.float64) >= limits
    if max_distance is not None:
        keep &= np.linalg.norm(boxes[:, :2], axis=-1) <= max_distance
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep].astype(np.int64)
    selected = rotated_nms(boxes, scores, labels, nms_iou)
    boxes = convert_box_origin(boxes[selected], target=origin)
    labels, scores = labels[selected], scores[selected]
    return dict(boxes_3d=boxes.tolist(), scores=scores.tolist(), labels=labels.tolist(),
                names=[CLASSES[int(label)] for label in labels])


def create_session(path, provider='cuda', threads=2, profile=False):
    if provider not in ('cuda', 'cpu'):
        raise ValueError('provider must be cuda or cpu')
    if provider == 'cuda':
        # ORT's documented compatibility path: importing torch preloads the
        # matching CUDA/cuDNN libraries. Not required if system libs suffice.
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    if provider == 'cuda' and 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('CUDA provider is unavailable; install onnxruntime-gpu or explicitly use --provider cpu')
    options = ort.SessionOptions()
    # Suppress repetitive generic ScatterND warnings for the exporter's
    # unique slice-assignment indices; provider errors still remain visible.
    options.log_severity_level = 3
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.enable_profiling = profile
    providers = ['CPUExecutionProvider']
    if provider == 'cuda':
        providers.insert(0, ('CUDAExecutionProvider', dict(
            device_id=0, gpu_mem_limit=4 * 1024**3, arena_extend_strategy='kSameAsRequested',
            cudnn_conv_algo_search='HEURISTIC', cudnn_conv_use_max_workspace='0', use_tf32='0')))
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    session.disable_fallback()
    if provider == 'cuda' and session.get_providers()[0] != 'CUDAExecutionProvider':
        raise RuntimeError('CUDA initialization failed; refusing a silent CPU-only run')
    return session


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def image_input(paths):
    if len(paths) != 4:
        raise ValueError('Exactly four images required in front,left,right,rear order')
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        if image.shape != (800, 800, 3):
            raise ValueError(f'{path}: need canonical 800x800 BGR images; do not resize arbitrary fisheye images')
        images.append(image.transpose(2, 0, 1))
    return np.ascontiguousarray(np.stack(images)[None], dtype=np.uint8)


def compare_outputs(actual, reference, box_atol=.02, score_atol=.002, yaw_atol=.002):
    boxes, scores, labels = [np.asarray(value) for value in actual]
    difference = boxes - reference['boxes']
    difference[..., 6] = (difference[..., 6] + np.pi) % (2*np.pi) - np.pi
    result = dict(max_box_error_m=float(np.abs(difference[..., :6]).max()),
                  max_yaw_error_rad=float(np.abs(difference[..., 6]).max()),
                  max_score_error=float(np.abs(scores-reference['scores']).max()),
                  class_mismatches=int(np.sum(labels != reference['labels'])))
    result['passed'] = (result['max_box_error_m'] <= box_atol and
        result['max_yaw_error_rad'] <= yaw_atol and result['max_score_error'] <= score_atol and
        result['class_mismatches'] == 0)
    return result

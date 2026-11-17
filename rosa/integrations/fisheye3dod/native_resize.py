"""Native OpenCV-fisheye images, resized in memory without lens remapping."""
import json
from pathlib import Path

import cv2
import mmcv
import numpy as np
from numpy.polynomial import Polynomial
from mmdet3d.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

from .calibration import convert_lidar2cam
from .constants import CAMERAS

CAMERA_KEYS = dict(zip(CAMERAS, ('ARC_F', 'ARC_SL', 'ARC_SR', 'ARC_R')))


def resized_opencv_as_ocam(intrinsic, output_size=(800, 800)):
    """Exact polynomial change of variables, NOT a fitted/approximated lens.

    ROSA rays are [down,right,back], so native incidence t = theta_ocam+pi/2.
    OpenCV rho = t+k1*t^3+k2*t^5+k3*t^7+k4*t^9. OCam's affine
    handles unequal fx/fy and unequal resize scales. Pixel centers follow
    cv2.resize's half-pixel convention. Coefficients are ascending (Horner).
    """
    sx, sy = output_size[0] / intrinsic['width'], output_size[1] / intrinsic['height']
    k = np.asarray(intrinsic['camera_matrix'], dtype=np.float64)
    if k[0, 1] != 0 or k[1, 0] != 0:
        raise ValueError('This adapter requires zero-skew OpenCV calibration')
    distortion = np.asarray(intrinsic['distortion_coefficients']).reshape(-1)
    if distortion.size != 4:
        raise ValueError('Expected four OpenCV-fisheye distortion coefficients')
    t = Polynomial([np.pi / 2, 1.])
    radial = t
    for index, coefficient in enumerate(distortion):
        radial = radial + coefficient * t ** (3 + 2 * index)
    fx, fy = k[0, 0] * sx, k[1, 1] * sy
    cx, cy = (k[0, 2] + .5) * sx - .5, (k[1, 2] + .5) * sy - .5
    matrix = np.array([[fy / fx, 0., cx], [0., 1., cy], [0., 0., 1.]], np.float32)
    polynomial = np.zeros(10, np.float32)
    polynomial[:len(radial.coef)] = radial.coef * fx
    return matrix, polynomial


@TRANSFORMS.register_module()
class LoadNativeResizedFisheyeForSparse4D(BaseTransform):
    """Read native JPEGs, cv2.resize to 800x800, update projection in memory."""

    def __init__(self, calibration, output_size=(800, 800), lens_fov_deg=220.):
        self.intrinsics = json.loads(Path(calibration).read_text())['intrinsics']
        self.output_size = tuple(output_size)
        self.fov = float(lens_fov_deg)
        if self.output_size != (800, 800):
            raise ValueError('This experiment fixes network input to 800x800')
        self.projections = {
            camera: resized_opencv_as_ocam(self.intrinsics[key], self.output_size)
            for camera, key in CAMERA_KEYS.items()}

    def transform(self, results):
        images, originals, paths, transforms, matrices, polynomials = [], [], [], [], [], []
        for camera in CAMERAS:
            info = results['cam_info']['cam_fisheye'][camera]
            native = self.intrinsics[CAMERA_KEYS[camera]]
            image = mmcv.imread(str(info['cam_path']), flag='color')
            if image is None:
                raise FileNotFoundError(str(info['cam_path']))
            if image.shape[:2] != (native['height'], native['width']):
                raise ValueError('Native dimensions disagree with camera calibration')
            if not np.allclose(np.asarray(info['cam2img'])[:3, :3],
                               native['camera_matrix'], rtol=0, atol=1e-4):
                raise ValueError('Annotations must contain the native OpenCV intrinsics')
            originals.append(image.shape)
            images.append(cv2.resize(image, self.output_size, interpolation=cv2.INTER_LINEAR).astype(np.float32))
            matrix, polynomial = self.projections[camera]
            matrices.append(matrix)
            polynomials.append(polynomial)
            transforms.append(convert_lidar2cam(info['lidar2cam']))
            paths.append(str(info['cam_path']))
        lidar2cam = np.stack(transforms)
        results.update(img=images, filename=paths, cam_names=list(CAMERAS),
                       cam_model=['FISHEYE_OCAMCALIB'] * 4,
                       cam2img=np.stack(matrices), fisheye_params=np.stack(polynomials),
                       lidar2cam=lidar2cam, cam2ego=np.linalg.inv(lidar2cam).astype(np.float32),
                       lidar2img=np.tile(np.eye(4, dtype=np.float32), (4, 1, 1)),
                       img_shape=[x.shape for x in images], ori_shape=originals,
                       pad_shape=[x.shape for x in images], ocam_lens_fov_deg=self.fov)
        return results

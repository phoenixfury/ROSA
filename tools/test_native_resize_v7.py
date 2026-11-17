"""Focused CPU checks for runtime native-image resize and exact projection."""
import copy
import json
import pickle
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

# Same compatibility shim as ROSA/tools/run.py for the installed old sklearn.
for key, value in [('float', float), ('int', int), ('bool', bool)]:
    if key not in np.__dict__:
        setattr(np, key, value)

from rosa.geometry import ocam_project_points
from rosa.integrations.fisheye3dod.native_resize import (
    CAMERA_KEYS, LoadNativeResizedFisheyeForSparse4D, resized_opencv_as_ocam)
from rosa.integrations.fisheye3dod.calibration import OFFICIAL_TO_ROSA_OCAM

REPO = Path(__file__).resolve().parents[3]
CALIB = REPO / 'projects/raybev_lite/configs/calibration_baselink.json'


class NativeResizeTests(unittest.TestCase):
    def test_native_projection_matches_scaled_pixels(self):
        calibration = json.loads(CALIB.read_text())['intrinsics']
        random = np.random.RandomState(3407)
        incidence = random.uniform(.01, np.deg2rad(109), 2000)
        azimuth = random.uniform(-np.pi, np.pi, 2000)
        rays = np.stack([np.sin(incidence)*np.cos(azimuth),
                         np.sin(incidence)*np.sin(azimuth), np.cos(incidence)], -1)
        for native in calibration.values():
            matrix, polynomial = resized_opencv_as_ocam(native)
            meta = dict(lidar2cam=torch.tensor(OFFICIAL_TO_ROSA_OCAM)[None,None],
                        ocam_intrinsic=torch.tensor(matrix)[None,None],
                        ocam_pol=torch.tensor(polynomial)[None,None], ocam_lens_fov_deg=220.)
            predicted = ocam_project_points(torch.tensor(rays,dtype=torch.float32)[None,:,None],
                meta, torch.tensor([[[800.,800.]]]),contract='full_fov')[0,0,:,0].numpy()*800
            distorted = incidence.copy()
            for index,k in enumerate(native['distortion_coefficients']):
                distorted += k * incidence ** (3+index*2)
            k = np.asarray(native['camera_matrix'])
            u = k[0,0]*distorted*np.cos(azimuth)+k[0,2]
            v = k[1,1]*distorted*np.sin(azimuth)+k[1,2]
            expected = (np.stack([u,v],-1)+.5)*[800/native['width'],800/native['height']]-.5
            valid = ((expected>=0)&(expected<800)).all(-1)
            np.testing.assert_allclose(predicted[valid],expected[valid],atol=.01,rtol=0)
            # OpenCV parity on its forward-camera hemisphere too.
            front = rays[:,2] > 0
            cv_pixels,_ = cv2.fisheye.projectPoints(rays[front,None],np.zeros(3),np.zeros(3),
                k,np.asarray(native['distortion_coefficients']))
            np.testing.assert_allclose(cv_pixels[:,0],np.stack([u,v],-1)[front],atol=1e-5)

    def test_actual_native_images_only_resized(self):
        root = REPO / 'data/Custom/combined_bags_camera_ray_dedup_v2'
        if not root.is_dir():
            root = REPO / 'data/combined_bags_camera_ray_dedup_v2'
        if not root.is_dir():
            self.skipTest('Native dataset unavailable')
        path = REPO / 'projects/raybev_lite/datasets/combined_bags_camera_ray_manual_review_v7/fisheye3dod_infos_val.pkl'
        with path.open('rb') as stream:
            row = copy.deepcopy(pickle.load(stream)['data_list'][0])
        ann_before = pickle.dumps(row['annos'])
        for info in row['cam_info']['cam_fisheye'].values():
            info['cam_path'] = str(root / info['cam_path'])
        result = LoadNativeResizedFisheyeForSparse4D(str(CALIB)).transform(row)
        self.assertEqual(pickle.dumps(row['annos']), ann_before)
        self.assertEqual(result['fisheye_params'].shape, (4,10))
        for index,camera in enumerate(CAMERA_KEYS):
            native = cv2.imread(row['cam_info']['cam_fisheye'][camera]['cam_path'])
            expected = cv2.resize(native,(800,800),interpolation=cv2.INTER_LINEAR)
            np.testing.assert_array_equal(result['img'][index],expected)
            self.assertEqual(result['ori_shape'][index],native.shape)


if __name__ == '__main__':
    unittest.main()

"""Pure-NumPy checks: no CUDA, MMEngine registry, or training needed."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1]/'rosa/integrations/fisheye3dod/motion_targets.py'
SPEC = importlib.util.spec_from_file_location('rosa_motion_targets_test_module', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
MotionTargetConfig = MODULE.MotionTargetConfig
fit_world_xy = MODULE.fit_world_xy
robust_track_velocities = MODULE.robust_track_velocities
AUDIT_SPEC = importlib.util.spec_from_file_location(
    'rosa_offline_audit_test_module', PATH.parents[3]/'tools/audit_temporal_motion_targets.py')
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
sys.modules[AUDIT_SPEC.name] = AUDIT
AUDIT_SPEC.loader.exec_module(AUDIT)
PROVENANCE_SPEC = importlib.util.spec_from_file_location(
    'rosa_motion_provenance_test_module', PATH.parents[3]/'tools/audit_temporal_motion_provenance.py')
PROVENANCE = importlib.util.module_from_spec(PROVENANCE_SPEC)
sys.modules[PROVENANCE_SPEC.name] = PROVENANCE
PROVENANCE_SPEC.loader.exec_module(PROVENANCE)


def sample(index, world, yaw=0., translation=0., scene='train', track_id=1):
    pose = np.eye(4)
    pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    pose[0, 3] = translation
    point = pose[:3, :3].T@(np.r_[world, 0.]-pose[:3, 3])
    return dict(token=f'{scene}_{index}', metainfo=dict(scene_name=scene,
        acquisition_timestamp_ns=1_700_000_000_000_000_000+int(index*500_000_000),
        T_world_CLIDAR=pose), annos=dict(gt_ids=[track_id],
        gt_boxes=[[*point, 4., 2., 1.5, 0.]]))


class MotionTargetTests(unittest.TestCase):
    def test_global_static_while_ego_moves_and_turns(self):
        samples = [sample(i, [10., 2.], yaw=i*.3, translation=i) for i in range(5)]
        targets, diagnostics = robust_track_velocities(samples, return_diagnostics=True)
        self.assertTrue(np.isnan(targets['train_0']).all())
        np.testing.assert_allclose(targets['train_4'], [[0., 0.]], atol=1e-6)
        self.assertEqual(diagnostics['train_4'][0]['status'], 'static')

    def test_moving_target_uses_current_lidar_axes_and_never_z(self):
        samples = [sample(i, [10.+i, 2.], yaw=np.pi/2) for i in range(5)]
        targets = robust_track_velocities(samples)
        np.testing.assert_allclose(targets['train_4'], [[0., -2.]], atol=1e-6)
        self.assertEqual(targets['train_4'].shape, (1, 2))

    def test_future_cannot_change_past_targets(self):
        samples = [sample(i, [10.+i, 2.]) for i in range(5)]
        short = robust_track_velocities(samples[:3])
        samples.append(sample(5, [1000., 1000.]))
        complete = robust_track_velocities(samples)
        for token in short:
            np.testing.assert_allclose(short[token], complete[token], equal_nan=True)

    def test_gaps_and_scene_boundaries_mask_unknown_not_zero(self):
        samples = [sample(i, [10.+i, 2.]) for i in range(3)]
        samples += [sample(8, [18., 2.]), sample(9, [19., 2.], scene='val')]
        targets = robust_track_velocities(samples)
        self.assertTrue(np.isnan(targets['train_8']).all())
        self.assertTrue(np.isnan(targets['val_9']).all())

    def test_huber_fit_resists_interior_outlier(self):
        times = np.arange(5, dtype=np.int64)*500_000_000
        positions = np.column_stack([np.arange(5), np.zeros(5)])
        positions[2, 1] = 2.
        config = MotionTargetConfig(max_residual_m=1.)
        fit = fit_world_xy(times, positions, config)
        self.assertTrue(np.isfinite(fit['velocity_world']).all())
        np.testing.assert_allclose(fit['velocity_world'], [2., 0.], atol=1e-6)
        self.assertLess(abs(fit['fitted_world_xy'][1]), .2)

    def test_endpoint_outlier_and_gross_jump_are_unknown(self):
        times = np.arange(5, dtype=np.int64)*500_000_000
        positions = np.column_stack([np.arange(5), np.zeros(5)])
        positions[-1, 1] = 10.
        fit = fit_world_xy(times, positions)
        self.assertTrue(np.isnan(fit['velocity_world']).all())
        samples = [sample(i, [10.+i, 0.]) for i in range(3)]
        samples += [sample(3, [100., 0.]), sample(4, [14., 0.])]
        targets, diagnostics = robust_track_velocities(samples, return_diagnostics=True)
        self.assertEqual(diagnostics['train_3'][0]['status'], 'jump_speed')
        self.assertTrue(np.isnan(targets['train_3']).all())
        np.testing.assert_allclose(targets['train_4'], [[2., 0.]], atol=1e-6)

    def test_irregular_true_timestamps_and_large_map_offsets(self):
        times = (np.array([0., .42, 1.03, 1.7, 2.1])*1e9).astype(np.int64)+1_700_000_000_000_000_000
        seconds = (times-times[0])*1e-9
        positions = np.column_stack([1e6+seconds*3., -2e6+seconds*.2])
        fit = fit_world_xy(times, positions)
        np.testing.assert_allclose(fit['velocity_world'], [3., .2], atol=1e-7)

    def test_aligned_unique_ids_required(self):
        row = sample(0, [10., 0.])
        row['annos']['gt_ids'] = []
        with self.assertRaises(ValueError):
            robust_track_velocities([row])

    def test_secondary_forecast_changes_only_origin_not_velocity(self):
        samples = []
        angle = .2
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)],
                             [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
        for index in range(5):
            world = np.array([10.+index+.02*np.sin(index), 2., 0.])
            row = sample(index, world[:2])
            row['metainfo']['T_world_CLIDAR'][:3, :3] = rotation
            row['annos']['gt_boxes'][0][:3] = (rotation.T@world).tolist()
            row['annos']['gt_names'] = ['Car']
            samples.append(row)
        _, rows = AUDIT.audit(samples, MotionTargetConfig(), [3])
        targets, diagnostics = robust_track_velocities(
            samples, MotionTargetConfig(window=3), return_diagnostics=True)
        current, future = samples[3], samples[4]
        token = current['token']
        velocity_world = (rotation@np.r_[targets[token][0], 0.])[:2]
        expected_origin = diagnostics[token][0]['fitted_world_xy']
        target_xy = (rotation@np.asarray(future['annos']['gt_boxes'][0][:3]))[:2]
        expected = np.linalg.norm(expected_origin+velocity_world*.5-target_xy)
        actual = next(row for row in rows if row['token'] == token)['robust3_fit_origin_error_m']
        self.assertAlmostEqual(actual, expected, places=9)

    def test_provenance_join_fails_closed_on_geometry_class_and_tokens(self):
        import copy
        row = sample(0, [10., 0.])
        row['token'] = '100'
        row['metainfo'].update(sequence_id='bag', source_frame_id=0, temporal_split='train')
        row['annos'].update(gt_names=['Car'], gt_ids=[10])
        source = dict(frame_id=0, timestamp_ns=100,
            lidar_sensor_timestamp_ns=row['metainfo']['acquisition_timestamp_ns'],
            objects=[dict(track_id=1, name='Car', score=.8, box=row['annos']['gt_boxes'][0])])
        result = PROVENANCE.verify_frame(row, source, {1: 'Car'}, {}, {})
        self.assertEqual(result['objects'][0]['source'], 'lidar')
        for alteration in ('geometry', 'class', 'token', 'timestamp'):
            bad_row, bad_source = copy.deepcopy(row), copy.deepcopy(source)
            if alteration == 'geometry':
                bad_source['objects'][0]['box'][0] += .1
            elif alteration == 'class':
                bad_row['annos']['gt_names'] = ['Truck']
            elif alteration == 'token':
                bad_source['timestamp_ns'] = 101
            else:
                bad_source['lidar_sensor_timestamp_ns'] += 1
            with self.assertRaises(ValueError, msg=alteration):
                PROVENANCE.verify_frame(bad_row, bad_source, {1: 'Car'}, {}, {})

    def test_provenance_join_requires_consistent_bijective_track_ids(self):
        row = sample(0, [10., 0.])
        row['token'] = '100'
        row['metainfo'].update(sequence_id='bag', source_frame_id=0, temporal_split='train')
        row['annos'].update(gt_names=['Car'], gt_ids=[10])
        source = dict(frame_id=0, timestamp_ns=100,
            lidar_sensor_timestamp_ns=row['metainfo']['acquisition_timestamp_ns'],
            objects=[dict(track_id=1, name='Car', score=.8, box=row['annos']['gt_boxes'][0])])
        for forward, reverse in (({1: 11}, {11: 1}), ({2: 10}, {10: 2})):
            with self.assertRaises(ValueError):
                PROVENANCE.verify_frame(row, source, {1: 'Car'}, forward, reverse)

    def test_provenance_join_rejects_stale_source_digest(self):
        import json
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.json'
            source.write_text('[]')
            (root/'summary.json').write_text(json.dumps(dict(frames=1, sequences=[dict(
                name='bag', annotation_file=str(source), annotation_sha256='stale')])))
            with patch.object(PROVENANCE, 'load_pickle', return_value=([{}], 'test-pkl-digest')):
                with self.assertRaisesRegex(ValueError, 'Stale source annotation SHA-256'):
                    PROVENANCE.build_sidecar(root)
            self.assertFalse((root/'provenance.json').exists())


if __name__ == '__main__':
    unittest.main()

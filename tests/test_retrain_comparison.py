"""Small CPU regression tests for the checkpoint-comparison tooling."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/custom_data'))
sys.path.insert(0, str(ROOT / 'projects/ROSA/tools'))
from compare_retrained_fisheye import (
    matched_recall_points, np, operating_counts, select, training_history,
)
from render_rosa_retraining_comparison import align_predictions


class ComparisonTest(unittest.TestCase):
    def test_score_order_and_threshold(self):
        frame = dict(scores=np.array([.2, .5, .1, .2]))
        self.assertEqual(select(frame, .2), [1, 0, 3])

    def test_class_aware_duplicates(self):
        frame = dict(scores=np.array([.9, .8, .7]), labels=np.array([0, 0, 2]),
                     boxes=np.array([[0, 0, 0, 4, 2, 2, 0]] * 3),
                     gt_boxes=np.array([[0, 0, 0, 4, 2, 2, 0]]),
                     gt_names=np.array(['Car']))
        total = operating_counts([frame], [[0, 1, 2]])['total']
        self.assertEqual(total['tp'], 1)
        self.assertEqual(total['duplicate_fp'], 1)
        self.assertEqual(total['wrong_class_or_nearby_fp'], 1)
        points = matched_recall_points([frame])
        self.assertAlmostEqual(points['0.5']['threshold'], .9)
        self.assertEqual(points['0.5']['precision'], 1.)

    def test_history_accepts_timestamp_or_work_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            timestamp = run / '20261030_000000'
            vis = timestamp / 'vis_data'
            vis.mkdir(parents=True)
            row = {'step': 1, 'fisheye_bev/mAP': .1, 'fisheye_bev/NDS': .2}
            (vis / 'scalars.json').write_text(json.dumps(row) + '\n')
            self.assertEqual(training_history(run), training_history(timestamp))

    def test_token_order_and_center_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            box = [1, 2, 3, 4, 5, 6, 0]
            rows = [dict(token=t, names=['Car'], boxes_3d=[box], scores=[.9])
                    for t in ('second', 'first')]
            payload = dict(box_origin='gravity_center', score_threshold=.05, frames=rows)
            path = root / 'predictions.json'
            path.write_text(json.dumps(payload))
            aligned, floor = align_predictions(root, [{'token':'first'}, {'token':'second'}])
            self.assertEqual(aligned[0][0][1][2], 3)
            self.assertEqual(floor, .05)
            payload['frames'][0]['token'] = 'first'
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                align_predictions(root, [{'token':'first'}, {'token':'second'}])

    def test_ambiguous_center_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'predictions.json').write_text(json.dumps(dict(frames=[])))
            with self.assertRaises(ValueError):
                align_predictions(root, [])


if __name__ == '__main__':
    unittest.main()

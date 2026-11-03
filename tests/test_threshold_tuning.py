import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from tune_fisheye_thresholds import CLASSES, choose, curve, evaluate, match_events, policies


def test_equal_score_group_cannot_be_split():
    events = np.array([[.8, 0, 0, 0], [.8, 0, 1, 0], [.4, 0, 0, 0], [.1, 0, 3, 0]])
    values = curve(events, 2)
    assert values['tp'].tolist() == [1, 2, 2]
    assert values['fp'].tolist() == [1, 1, 2]
    cutoff = choose(events, 2, retention=1.)
    assert cutoff == pytest.approx(.4)
    assert np.sum((events[:, 0] >= cutoff) & (events[:, 2] == 0)) == 2


def test_prefix_matching_and_cross_class_independence():
    boxes = np.array([[0, 0, 0, 4, 2, 2, 0], [0, 0, 0, 4, 2, 2, 0],
                      [10, 0, 0, 4, 2, 2, 0]], dtype=float)
    frame = dict(boxes=boxes, scores=np.array([.9, .8, .7]), labels=np.array([0, 0, 3]),
                 gt_boxes=boxes[[0, 2]], gt_names=np.array(['Car', 'Bus']))
    events, gt = match_events([frame])
    assert events[:, 2].tolist() == [0, 1, 0]
    limits = dict.fromkeys(CLASSES, .85)
    limits['Bus'] = .6
    value = evaluate(events, gt, limits)['total']
    assert value['tp'] == 2 and value['fp'] == 0


def test_unsupported_classes_keep_baseline():
    events = np.array([[.8, 0, 0, 0], [.1, 1, 3, 0]])
    gt = np.zeros((1, 6), dtype=int)
    gt[0, 0] = 1
    choices, support = policies(events, gt)
    assert choices['classwise_recall_first'] == dict.fromkeys(CLASSES, .1)
    assert not support['Van']['tuned']
    assert choose(events[:0], 0) == .1


def test_retention_rounds_up_required_true_positives():
    events = np.array([[.9, 0, 0, 0], [.5, 0, 0, 0], [.1, 0, 0, 0]])
    assert choose(events, 3, retention=.95) == pytest.approx(.1)

"""Small arithmetic/zero-count tests for the read-only checkpoint analysis."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from analyze_singleframe_refresh import CLASSES, complete_counts, nds_decomposition


def metric_row():
    result = dict(mAP=.1, mATE=.5, mASE=.2, mAOE=.3)
    result['NDS'] = (3*.1 + .5 + .8 + .7) / 6
    for name in CLASSES:
        result.update({name+'_AP': .1, name+'_ATE': .5,
                       name+'_ASE': .2, name+'_AOE': .3})
    return result


def test_nds_translation_contribution_and_rare_class_effect():
    parent = metric_row()
    current = dict(parent, mATE=.44, Cyclist_ATE=.14, NDS=parent['NDS']+.01)
    result = nds_decomposition(parent, current)
    assert result['total'] == pytest.approx(.01)
    assert sum(result['macro_contributions'].values()) == pytest.approx(.01)
    assert result['per_class']['Cyclist']['total_contribution'] == pytest.approx(.01)


def test_nds_ap_contribution():
    parent = metric_row()
    current = dict(parent, mAP=.12, Cyclist_AP=.22, NDS=parent['NDS']+.01)
    result = nds_decomposition(parent, current)
    assert result['macro_contributions']['mAP'] == pytest.approx(.01)
    assert result['per_class']['Cyclist']['AP'] == pytest.approx(.01)


def test_counts_explicit_fp_fn_and_missing_classes():
    counts = complete_counts(dict(total=dict(gt=5, predictions=6, tp=3,
        precision=.5, recall=.6), per_class={}))
    assert counts['total']['fp'] == 3
    assert counts['total']['fn'] == 2
    assert counts['total']['f1'] == pytest.approx(6/11)
    assert counts['per_class']['Van']['f1'] == 0


def test_empty_counts_are_safe():
    counts = complete_counts(dict(total=dict(precision=0., recall=0.), per_class={}))
    assert counts['total']['fp'] == counts['total']['fn'] == 0
    assert counts['total']['f1'] == 0

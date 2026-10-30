"""Baseline-compatible inference: one label/query, quality ranking, class NMS.

Evaluate the original epoch-16 weights or a selected adaptation checkpoint.
No global score cutoff and no cross-class/vehicle-group suppression.
"""
_base_ = ['./rosa_vov99_balanced_finetune_8e.py']

model = dict(head=dict(decoder=dict(
    single_label=True,
    quality_before_topk=True,
    nms_iou_threshold=.5,
    nms_class_groups=(),
    score_threshold=None,
)))

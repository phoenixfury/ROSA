"""Optional suppression for deployment; evaluate raw and suppressed separately."""
_base_ = ['./rosa_vov99_balanced_finetune_8e.py']
model = dict(head=dict(decoder=dict(nms_iou_threshold=.5,
                                    nms_class_groups=((0, 1, 2, 3),))))

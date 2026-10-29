#!/usr/bin/env python3
"""Remap the released KITTI-360 ROSA classifier to our six-class order.

All non-classification weights are preserved.  The final classifier rows use
the closest source class when the target class was absent from KITTI-360:
Van <- car, Bus <- truck, and Cyclist <- bicycle.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import torch


SOURCE_CLASSES = ("car", "truck", "trailer", "motorcycle", "bicycle", "pedestrian")
TARGET_CLASSES = ("Car", "Van", "Truck", "Bus", "Pedestrian", "Cyclist")
TARGET_FROM_SOURCE = (0, 0, 1, 1, 5, 4)
FINAL_CLASSIFIER = re.compile(
    r"^head\.fisheye_box_decoder\.classifiers\.\d+\.6\.(weight|bias)$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu")
    state = checkpoint["state_dict"]
    changed = []
    for name, value in state.items():
        if not FINAL_CLASSIFIER.match(name):
            continue
        if value.shape[0] != len(SOURCE_CLASSES):
            raise ValueError(f"{name}: expected six source rows, got {value.shape}")
        state[name] = value[list(TARGET_FROM_SOURCE)].clone()
        changed.append(name)
    if len(changed) != 12:
        raise RuntimeError(f"expected 12 final classifier tensors, found {len(changed)}")

    meta = checkpoint.setdefault("meta", {})
    meta["initialization_source"] = str(args.source.resolve())
    meta["source_classes"] = SOURCE_CLASSES
    meta["target_classes"] = TARGET_CLASSES
    meta["target_classifier_rows_from_source"] = TARGET_FROM_SOURCE
    meta["classifier_remap_note"] = (
        "Van initialized from car; Bus from truck; Cyclist from bicycle"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"wrote {args.output} ({len(changed)} tensors remapped)")


if __name__ == "__main__":
    main()

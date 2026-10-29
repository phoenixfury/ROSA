"""Bounded foreground balancing and geometry-preserving image augmentation."""
from __future__ import annotations

import math
import cv2
import numpy as np
import torch
from torch import nn
from mmcv.transforms import BaseTransform
from mmengine.dataset import ClassBalancedDataset
from mmdet.models.losses import FocalLoss
from mmdet3d.registry import DATASETS, MODELS, TRANSFORMS


@MODELS.register_module()
class ForegroundBalancedFocalLoss(nn.Module):
    """Reweight positive class entries; retain ordinary background penalties.

    Multiplying all six class columns by rare-class weights would also amplify
    millions of rare-class negative entries. Only the true class is upweighted.
    No persistent buffer/parameter is added, so old checkpoints remain usable.
    """

    def __init__(self, class_weights, **kwargs):
        super().__init__()
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if weights.ndim != 1 or not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("class_weights must be finite positive values")
        self.register_buffer("class_weights", weights, persistent=False)
        self.focal = FocalLoss(**kwargs)

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        if pred.shape[-1] != len(self.class_weights):
            raise ValueError("class_weights do not match classifier outputs")
        pred = pred.float()
        weights = torch.ones_like(pred)
        class_weights = self.class_weights.to(pred)
        if target.ndim == pred.ndim:
            weights = torch.where(target > 0, class_weights[None], weights)
        else:
            foreground = (target >= 0) & (target < pred.shape[-1])
            rows = torch.nonzero(foreground, as_tuple=True)[0]
            weights[rows, target[rows]] = class_weights[target[rows]]
        if weight is not None:
            if weight.ndim == 1:
                weight = weight[:, None]
            weights = weights * weight
        return self.focal(pred, target, weight=weights, avg_factor=avg_factor,
                          reduction_override=reduction_override)


@DATASETS.register_module()
class CappedClassBalancedDataset(ClassBalancedDataset):
    """Repeat rare-class frames with bounded, seeded fractional rounding.

    Every original frame remains once, including empty frames. Fractional
    repeat factors are rounded stochastically once with a fixed seed rather
    than rounded up: a 1.05 factor must not double every frame of that class.
    DefaultSampler subsequently shuffles the resulting list each epoch.
    """

    def __init__(self, dataset, oversample_thr=.2, max_repeat=3., seed=3407, **kwargs):
        if max_repeat < 1 or not math.isfinite(max_repeat):
            raise ValueError("max_repeat must be finite and >= 1")
        self.max_repeat = float(max_repeat)
        self.seed = int(seed)
        super().__init__(dataset, oversample_thr, **kwargs)

    def full_init(self):
        if self._fully_initialized:
            return
        self.dataset.full_init()
        factors = np.minimum(self._get_repeat_factors(self.dataset, self.oversample_thr),
                             self.max_repeat)
        rng = np.random.RandomState(self.seed)
        repeats = np.floor(factors).astype(int)
        repeats += rng.uniform(size=len(factors)) < factors - repeats
        self.repeat_factors = factors.tolist()
        self.repeat_indices = np.repeat(np.arange(len(factors)), repeats).tolist()
        self._fully_initialized = True


@TRANSFORMS.register_module()
class FisheyePhotometricAugmentation(BaseTransform):
    """Mild exposure/color/noise/blur changes without moving any image pixels."""

    def __init__(self, probability=.8, brightness=12., contrast=(.9, 1.1),
                 color_gain=(.95, 1.05), noise_std=2., blur_probability=.1):
        self.probability = probability
        self.brightness = brightness
        self.contrast = contrast
        self.color_gain = color_gain
        self.noise_std = noise_std
        self.blur_probability = blur_probability

    def transform(self, results):
        if np.random.random() >= self.probability:
            return results
        contrast = np.random.uniform(*self.contrast)
        brightness = np.random.uniform(-self.brightness, self.brightness)
        gains = np.random.uniform(*self.color_gain, size=3).astype(np.float32)
        images = []
        for original in results['img']:
            mask = np.any(original != 0, axis=-1)
            image = original.astype(np.float32, copy=True)
            image = (image * contrast + brightness) * gains
            if np.random.random() < self.blur_probability:
                image = cv2.GaussianBlur(image, (3, 3), 0)
            if self.noise_std > 0:
                image += np.random.normal(0, np.random.uniform(0, self.noise_std),
                                          image.shape).astype(np.float32)
            image = np.clip(image, 0, 255).astype(np.float32)
            image[~mask] = 0  # Keep the black region outside the fisheye lens.
            images.append(image)
        results['img'] = images
        return results

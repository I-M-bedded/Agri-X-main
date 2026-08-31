# -*- coding: utf-8 -*-
"""Resize policies (spec section 7) and exact coordinate round-tripping.

Every predicted pixel/line coordinate must map back to original camera
coordinates, so the resize is expressed as an affine transform
``model = source * scale + pad`` whose parameters are returned with the image.

Modes:
  stretch    P0: anisotropic resize to the target (geometry distorted)
  letterbox  P1: aspect-preserving resize plus symmetric zero padding
  fit        P2/P3: plain resize to a target with the same aspect ratio
             (falls back to anisotropic scaling if aspects differ)
"""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ResizeTransform:
    """Affine map from source pixels to model-input pixels."""

    source_size: Tuple[int, int]   # (W, H)
    target_size: Tuple[int, int]   # (W, H)
    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float

    def source_to_model(self, x: float, y: float) -> Tuple[float, float]:
        return x * self.scale_x + self.pad_x, y * self.scale_y + self.pad_y

    def model_to_source(self, x: float, y: float) -> Tuple[float, float]:
        return (x - self.pad_x) / self.scale_x, (y - self.pad_y) / self.scale_y


def make_transform(
    source_size: Tuple[int, int], target_size: Tuple[int, int], mode: str
) -> ResizeTransform:
    sw, sh = source_size
    tw, th = target_size
    if mode == "stretch" or mode == "fit":
        return ResizeTransform(source_size, target_size, tw / sw, th / sh, 0.0, 0.0)
    if mode == "letterbox":
        scale = min(tw / sw, th / sh)
        pad_x = (tw - sw * scale) / 2.0
        pad_y = (th - sh * scale) / 2.0
        return ResizeTransform(source_size, target_size, scale, scale, pad_x, pad_y)
    raise ValueError(f"unknown resize mode: {mode}")


def _warp(image: np.ndarray, transform: ResizeTransform, interpolation: int) -> np.ndarray:
    tw, th = transform.target_size
    matrix = np.array(
        [[transform.scale_x, 0.0, transform.pad_x], [0.0, transform.scale_y, transform.pad_y]],
        np.float64,
    )
    return cv2.warpAffine(
        image, matrix, (tw, th), flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def preprocess_image(
    image: np.ndarray, target_size: Tuple[int, int], mode: str = "stretch"
) -> Tuple[np.ndarray, ResizeTransform]:
    """Resize an image for the model; returns (image, transform)."""

    transform = make_transform((image.shape[1], image.shape[0]), target_size, mode)
    return _warp(image, transform, cv2.INTER_LINEAR), transform


def preprocess_mask(
    mask: np.ndarray, target_size: Tuple[int, int], mode: str = "stretch"
) -> Tuple[np.ndarray, ResizeTransform]:
    """Resize a label mask with nearest-neighbour so class ids stay exact."""

    transform = make_transform((mask.shape[1], mask.shape[0]), target_size, mode)
    return _warp(mask, transform, cv2.INTER_NEAREST), transform

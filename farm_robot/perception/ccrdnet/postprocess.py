# -*- coding: utf-8 -*-
"""NAV_BAND mask -> navigation line -> FurrowEstimate.

Implements the paper baseline (largest connected component + least squares)
plus the mandatory ablation variants from spec sections 17-18:

  component selection: C0 largest / C1 bottom-center / C2 scored
  line fit:            F0 least squares / F1 PCA-TLS / F2 RANSAC

Everything is deterministic for a given mask and config (RANSAC is seeded).
An empty or unusable mask returns a valid no-guidance result with
confidence 0 (spec section 32).
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import cv2
import numpy as np

from sensors.perception_types import FurrowEstimate


@dataclass(frozen=True)
class PostprocessConfig:
    component_mode: str = "largest"        # largest | bottom_center | scored
    fit_mode: str = "ls"                   # ls | tls | ransac
    min_component_area: int = 40           # pixels at the mask's own resolution
    near_row_ratio: float = 0.80           # near lookahead: y = 0.80 * H
    far_row_ratio: float = 0.45            # far lookahead:  y = 0.45 * H
    full_span_ratio: float = 0.35          # vertical span giving span_score 1.0
    fit_residual_norm: float = 0.06        # residual / width giving fit_score 0
    ransac_iterations: int = 64
    ransac_inlier_px: float = 4.0
    ransac_seed: int = 42
    # scored-mode weights (C2)
    score_area: float = 0.35
    score_center: float = 0.30
    score_bottom: float = 0.20
    score_span: float = 0.15


@dataclass(frozen=True)
class NavigationLine:
    """Fitted centerline x = slope * y + intercept in mask pixel coordinates."""

    slope: float
    intercept: float
    x_near: float
    y_near: float
    x_far: float
    y_far: float
    angle_deg: float          # 0 = straight ahead, + = line leans right going up
    component_area: int
    component_span: int
    fit_residual: float


@dataclass(frozen=True)
class NavigationResult:
    estimate: FurrowEstimate
    line: Optional[NavigationLine]
    reason: str = ""
    model_score: float = 0.0
    span_score: float = 0.0
    fit_score: float = 0.0


def _empty(reason: str, coverage: float = 0.0) -> NavigationResult:
    return NavigationResult(FurrowEstimate(coverage=coverage), None, reason)


def _select_component(
    mask: np.ndarray, config: PostprocessConfig
) -> Optional[np.ndarray]:
    """Return a boolean mask of the selected component, or None."""

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = mask.shape
    best_label, best_key = 0, None
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < config.min_component_area:
            continue
        top = int(stats[label, cv2.CC_STAT_TOP])
        span = int(stats[label, cv2.CC_STAT_HEIGHT])
        bottom = top + span
        cx = float(centroids[label][0])
        if config.component_mode == "largest":
            key = (float(area), -label)
        elif config.component_mode == "bottom_center":
            # prefer components reaching the image bottom, then nearest centre
            key = (float(bottom), -abs(cx - width / 2.0), -label)
        elif config.component_mode == "scored":
            score = (
                config.score_area * min(1.0, area / (height * width * 0.02))
                + config.score_center * max(0.0, 1.0 - abs(cx - width / 2.0) / (width / 2.0))
                + config.score_bottom * (bottom / height)
                + config.score_span * min(1.0, span / (height * config.full_span_ratio))
            )
            key = (score, -label)
        else:
            raise ValueError(f"unknown component_mode: {config.component_mode}")
        if best_key is None or key > best_key:
            best_key, best_label = key, label
    if best_label == 0:
        return None
    return labels == best_label


def _fit_ls(ys: np.ndarray, xs: np.ndarray) -> Tuple[float, float]:
    """Ordinary least squares of x on y (rows are the independent variable)."""
    slope, intercept = np.polyfit(ys, xs, 1)
    return float(slope), float(intercept)


def _fit_tls(ys: np.ndarray, xs: np.ndarray) -> Tuple[float, float]:
    """PCA / total least squares via the principal axis of the point cloud."""
    y_mean, x_mean = ys.mean(), xs.mean()
    cov = np.cov(np.stack([ys - y_mean, xs - x_mean]))
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    dy, dx = float(direction[0]), float(direction[1])
    if abs(dy) < 1e-9:
        # horizontal cloud: degenerate for x = f(y); fall back to LS
        return _fit_ls(ys, xs)
    slope = dx / dy
    return slope, float(x_mean - slope * y_mean)


def _fit_ransac(
    ys: np.ndarray, xs: np.ndarray, config: PostprocessConfig
) -> Tuple[float, float]:
    rng = np.random.default_rng(config.ransac_seed)
    n = len(ys)
    best_inliers: Optional[np.ndarray] = None
    best_count = -1
    for _ in range(config.ransac_iterations):
        i, j = rng.integers(0, n, size=2)
        if ys[i] == ys[j]:
            continue
        slope = (xs[j] - xs[i]) / (ys[j] - ys[i])
        intercept = xs[i] - slope * ys[i]
        residuals = np.abs(xs - (slope * ys + intercept))
        inliers = residuals <= config.ransac_inlier_px
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
    if best_inliers is None or best_inliers.sum() < 2:
        return _fit_ls(ys, xs)
    return _fit_ls(ys[best_inliers], xs[best_inliers])


_FITTERS = {"ls": _fit_ls, "tls": _fit_tls}


def extract_navigation_line(
    nav_mask: np.ndarray,
    config: PostprocessConfig = PostprocessConfig(),
    nav_probability: Optional[np.ndarray] = None,
) -> NavigationResult:
    """Turn a binary NAV_BAND mask into a fitted line and FurrowEstimate.

    ``nav_mask``: HxW, nonzero where NAV_BAND was predicted.
    ``nav_probability``: optional HxW softmax probability of NAV_BAND used for
    the model_score part of confidence; without it the mask mean is 1.0.
    """

    if nav_mask.ndim != 2:
        raise ValueError(f"expected HxW mask, got {nav_mask.shape}")
    work = np.where(nav_mask > 0, 255, 0).astype(np.uint8)
    height, width = work.shape
    coverage = float(np.count_nonzero(work) / work.size)
    if not work.any():
        return _empty("empty NAV_BAND mask")

    component = _select_component(work, config)
    if component is None:
        return _empty("no component above minimum area", coverage)

    ys_idx, xs_idx = np.nonzero(component)
    ys = ys_idx.astype(np.float64)
    xs = xs_idx.astype(np.float64)
    span = int(ys_idx.max() - ys_idx.min() + 1)
    if span < 2 or len(np.unique(ys_idx)) < 2:
        return _empty("component has no vertical extent", coverage)

    if config.fit_mode == "ransac":
        slope, intercept = _fit_ransac(ys, xs, config)
    elif config.fit_mode in _FITTERS:
        slope, intercept = _FITTERS[config.fit_mode](ys, xs)
    else:
        raise ValueError(f"unknown fit_mode: {config.fit_mode}")

    residual = float(np.mean(np.abs(xs - (slope * ys + intercept))))
    y_near = config.near_row_ratio * (height - 1)
    y_far = config.far_row_ratio * (height - 1)
    x_near = slope * y_near + intercept
    x_far = slope * y_far + intercept
    # slope is dx/dy; angle relative to the vertical image axis
    angle_deg = math.degrees(math.atan(slope))

    line = NavigationLine(
        slope=slope,
        intercept=float(intercept),
        x_near=float(x_near),
        y_near=float(y_near),
        x_far=float(x_far),
        y_far=float(y_far),
        angle_deg=float(angle_deg),
        component_area=int(component.sum()),
        component_span=span,
        fit_residual=residual,
    )

    model_score = (
        float(np.clip(np.mean(nav_probability[component]), 0.0, 1.0))
        if nav_probability is not None
        else 1.0
    )
    span_score = float(min(1.0, span / (height * config.full_span_ratio)))
    fit_score = float(max(0.0, 1.0 - residual / (width * config.fit_residual_norm)))
    confidence = model_score * span_score * fit_score

    half_width = width / 2.0
    estimate = FurrowEstimate(
        normalized_error=float(np.clip((x_near - half_width) / half_width, -1.5, 1.5)),
        heading_error=float(np.clip((x_far - x_near) / half_width, -1.5, 1.5)),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        coverage=coverage,
    )
    return NavigationResult(
        estimate=estimate,
        line=line,
        model_score=model_score,
        span_score=span_score,
        fit_score=fit_score,
    )

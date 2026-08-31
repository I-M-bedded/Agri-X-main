# -*- coding: utf-8 -*-
"""Navigation metrics for CCRDNet evaluation (spec section 15).

All metrics compare the *fitted navigation line* of the prediction against the
line fitted on the ground-truth NAV_BAND with the identical postprocess, at the
model input resolution.

  angle error     |theta_pred - theta_gt| in degrees
  lateral error   |x_pred(y) - x_gt(y)| at the near/far lookahead rows, px
  line IoU        IoU between the predicted and GT lines rendered as
                  fixed-width bands (paper-style Line IoU)
  line accuracy   fraction of frames with AE <= 5 deg AND near lateral
                  error <= 10 px (project criterion, documented in summary)
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

_FARM_ROBOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _FARM_ROBOT not in sys.path:
    sys.path.insert(0, _FARM_ROBOT)

from perception.ccrdnet.postprocess import (  # noqa: E402
    NavigationResult,
    PostprocessConfig,
    extract_navigation_line,
)

ACC_ANGLE_DEG = 5.0
ACC_LATERAL_PX = 10.0


@dataclass
class FrameMetrics:
    name: str
    ok: bool                      # both GT and prediction produced a line
    gt_ok: bool
    pred_ok: bool
    angle_error: float = float("nan")
    lateral_near: float = float("nan")
    lateral_far: float = float("nan")
    line_iou: float = float("nan")
    accurate: bool = False
    confidence: float = 0.0
    nav_iou: float = float("nan")  # raw mask IoU of NAV_BAND class


def render_line_band(
    result: NavigationResult, shape, width_px: int
) -> Optional[np.ndarray]:
    if result.line is None:
        return None
    h, w = shape
    line = result.line
    canvas = np.zeros((h, w), np.uint8)
    y0, y1 = 0.0, float(h - 1)
    x0 = line.slope * y0 + line.intercept
    x1 = line.slope * y1 + line.intercept
    cv2.line(canvas, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))),
             255, thickness=width_px)
    return canvas


def band_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return float("nan")
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter / union) if union else float("nan")


def mask_iou(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
    p = pred == class_id
    g = gt == class_id
    union = np.logical_or(p, g).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(p, g).sum() / union)


def compute_frame_metrics(
    name: str,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    nav_class: int,
    postprocess: PostprocessConfig,
    line_width_px: int,
    nav_probability: Optional[np.ndarray] = None,
) -> FrameMetrics:
    pred_nav = (pred_mask == nav_class).astype(np.uint8)
    gt_nav = (gt_mask == nav_class).astype(np.uint8)
    pred_res = extract_navigation_line(pred_nav, postprocess, nav_probability)
    gt_res = extract_navigation_line(gt_nav, postprocess)

    fm = FrameMetrics(
        name=name,
        ok=pred_res.line is not None and gt_res.line is not None,
        gt_ok=gt_res.line is not None,
        pred_ok=pred_res.line is not None,
        confidence=pred_res.estimate.confidence,
        nav_iou=mask_iou(pred_mask, gt_mask, nav_class),
    )
    if not fm.ok:
        return fm

    pl, gl = pred_res.line, gt_res.line
    fm.angle_error = abs(pl.angle_deg - gl.angle_deg)
    fm.lateral_near = abs(pl.x_near - gl.x_near)
    fm.lateral_far = abs(pl.x_far - gl.x_far)
    shape = pred_mask.shape
    fm.line_iou = band_iou(
        render_line_band(pred_res, shape, line_width_px),
        render_line_band(gt_res, shape, line_width_px),
    )
    fm.accurate = fm.angle_error <= ACC_ANGLE_DEG and fm.lateral_near <= ACC_LATERAL_PX
    return fm


def summarize(frames) -> dict:
    def stats(values):
        arr = np.array([v for v in values if not math.isnan(v)], np.float64)
        if arr.size == 0:
            return {"mean": None, "median": None, "p95": None, "n": 0}
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "n": int(arr.size),
        }

    total = len(frames)
    evaluable = [f for f in frames if f.gt_ok]
    detected = [f for f in evaluable if f.pred_ok]
    return {
        "frames_total": total,
        "frames_with_gt_line": len(evaluable),
        "frames_with_pred_line": len(detected),
        "detection_rate": len(detected) / len(evaluable) if evaluable else None,
        "angle_error_deg": stats([f.angle_error for f in detected]),
        "lateral_near_px": stats([f.lateral_near for f in detected]),
        "lateral_far_px": stats([f.lateral_far for f in detected]),
        "line_iou": stats([f.line_iou for f in detected]),
        "nav_mask_iou": stats([f.nav_iou for f in frames]),
        "line_accuracy": (
            sum(1 for f in evaluable if f.accurate) / len(evaluable) if evaluable else None
        ),
        "accuracy_criterion": f"AE<={ACC_ANGLE_DEG}deg AND lateral_near<={ACC_LATERAL_PX}px",
    }

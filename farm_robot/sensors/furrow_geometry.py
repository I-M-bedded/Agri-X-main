from __future__ import annotations

"""Turn segmentation masks into low-cost furrow steering geometry.

The network is deliberately responsible only for pixels.  This module owns the
camera-geometry assumptions and can therefore be unit-tested without PyTorch or
ONNX Runtime.  Boundary masks use a probabilistic Hough transform; filled
corridor masks use horizontal-band centroids.
"""

from dataclasses import dataclass
import math
from typing import Optional

import cv2
import numpy as np

from sensors.perception_types import FurrowEstimate


@dataclass(frozen=True)
class BoundaryLine:
    x_bottom: float
    x_lookahead: float
    strength: float


@dataclass(frozen=True)
class FurrowGeometry:
    estimate: FurrowEstimate
    line_count: int
    boundaries: tuple[BoundaryLine, ...]
    selected_pair: Optional[tuple[int, int]]
    centerline: Optional[tuple[tuple[int, int], tuple[int, int]]]
    reason: str = ""


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    result = np.where(mask > 0, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _merge_lines(
    lines: list[BoundaryLine], width: int
) -> tuple[BoundaryLine, ...]:
    if not lines:
        return ()
    bottom_tol = max(8.0, width * 0.035)
    look_tol = max(12.0, width * 0.060)
    clusters: list[list[BoundaryLine]] = []
    for line in sorted(lines, key=lambda item: item.x_bottom):
        if (
            clusters
            and abs(line.x_bottom - np.average(
                [item.x_bottom for item in clusters[-1]],
                weights=[item.strength for item in clusters[-1]],
            )) <= bottom_tol
            and abs(line.x_lookahead - np.average(
                [item.x_lookahead for item in clusters[-1]],
                weights=[item.strength for item in clusters[-1]],
            )) <= look_tol
        ):
            clusters[-1].append(line)
        else:
            clusters.append([line])

    merged = []
    for cluster in clusters:
        weights = np.asarray([max(1.0, item.strength) for item in cluster])
        merged.append(
            BoundaryLine(
                x_bottom=float(np.average([item.x_bottom for item in cluster], weights=weights)),
                x_lookahead=float(
                    np.average([item.x_lookahead for item in cluster], weights=weights)
                ),
                strength=float(weights.sum()),
            )
        )
    return tuple(merged)


def boundary_mask_to_geometry(
    mask: np.ndarray,
    roi_start_ratio: float = 0.30,
    lookahead_ratio: float = 0.55,
    bottom_ratio: float = 0.93,
) -> FurrowGeometry:
    """Extract the corridor between the two row boundaries nearest image centre."""

    work = _binary(mask)
    height, width = work.shape
    y_start = int(height * roi_start_ratio)
    y_look = int(height * lookahead_ratio)
    y_bottom = min(height - 1, int(height * bottom_ratio))
    work[:y_start] = 0
    roi_pixels = max(1, (height - y_start) * width)
    coverage = float(np.count_nonzero(work[y_start:]) / roi_pixels)

    segments = cv2.HoughLinesP(
        work,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(12, int(height * 0.07)),
        minLineLength=max(18, int(height * 0.16)),
        maxLineGap=max(8, int(height * 0.08)),
    )
    raw: list[BoundaryLine] = []
    if segments is not None:
        # OpenCV normally returns (N, 1, 4), but some builds return (N, 4).
        # Normalizing here prevents a perception-thread crash on valid masks.
        for packed in np.asarray(segments).reshape(-1, 4):
            x1, y1, x2, y2 = (float(v) for v in packed)
            dy = y2 - y1
            if abs(dy) < height * 0.10:
                continue
            slope = (x2 - x1) / dy
            if abs(slope) > 2.2:
                continue
            x_bottom = x1 + slope * (y_bottom - y1)
            x_look = x1 + slope * (y_look - y1)
            if not (-0.35 * width <= x_bottom <= 1.35 * width):
                continue
            if not (-0.35 * width <= x_look <= 1.35 * width):
                continue
            strength = math.hypot(x2 - x1, y2 - y1)
            raw.append(BoundaryLine(x_bottom, x_look, strength))

    boundaries = _merge_lines(raw, width)
    if len(boundaries) < 2:
        return FurrowGeometry(
            estimate=FurrowEstimate(coverage=coverage),
            line_count=len(boundaries),
            boundaries=boundaries,
            selected_pair=None,
            centerline=None,
            reason="fewer than two stable boundary lines",
        )

    centre = width / 2.0
    choices: list[tuple[float, int, int]] = []
    for index in range(len(boundaries) - 1):
        left, right = boundaries[index], boundaries[index + 1]
        gap_bottom = right.x_bottom - left.x_bottom
        gap_look = right.x_lookahead - left.x_lookahead
        if not width * 0.10 <= gap_bottom <= width * 0.80:
            continue
        if gap_look <= width * 0.025:
            continue
        midpoint = (left.x_bottom + right.x_bottom) / 2.0
        straddles = left.x_bottom <= centre <= right.x_bottom
        perspective_penalty = max(0.0, gap_look / gap_bottom - 1.25)
        score = abs(midpoint - centre) / (width / 2.0) + perspective_penalty
        if not straddles:
            score += 0.40
        choices.append((score, index, index + 1))

    if not choices:
        return FurrowGeometry(
            estimate=FurrowEstimate(coverage=coverage),
            line_count=len(boundaries),
            boundaries=boundaries,
            selected_pair=None,
            centerline=None,
            reason="no plausible central boundary pair",
        )

    _, left_i, right_i = min(choices)
    left, right = boundaries[left_i], boundaries[right_i]
    centre_bottom = (left.x_bottom + right.x_bottom) / 2.0
    centre_look = (left.x_lookahead + right.x_lookahead) / 2.0
    gap_bottom = right.x_bottom - left.x_bottom
    gap_look = right.x_lookahead - left.x_lookahead

    normalized_error = _clamp((centre_bottom - centre) / (width / 2.0), -1.5, 1.5)
    heading_error = _clamp((centre_look - centre_bottom) / (width / 2.0), -1.5, 1.5)
    line_score = min(1.0, len(boundaries) / 4.0)
    centre_score = max(0.0, 1.0 - abs(centre_bottom - centre) / (width * 0.55))
    perspective_ratio = gap_look / max(1.0, gap_bottom)
    perspective_score = max(0.0, 1.0 - abs(perspective_ratio - 0.65) / 0.85)
    coverage_score = max(0.0, 1.0 - abs(coverage - 0.08) / 0.18)
    confidence = _clamp(
        0.30 * line_score
        + 0.30 * centre_score
        + 0.25 * perspective_score
        + 0.15 * coverage_score
    )
    centerline = (
        (int(round(centre_look)), y_look),
        (int(round(centre_bottom)), y_bottom),
    )
    return FurrowGeometry(
        estimate=FurrowEstimate(
            normalized_error=normalized_error,
            heading_error=heading_error,
            confidence=confidence,
            coverage=coverage,
        ),
        line_count=len(boundaries),
        boundaries=boundaries,
        selected_pair=(left_i, right_i),
        centerline=centerline,
    )


def corridor_mask_to_geometry(
    mask: np.ndarray,
    roi_start_ratio: float = 0.45,
) -> FurrowGeometry:
    """Centroid-fit fallback for a filled traversable-corridor mask."""

    work = _binary(mask)
    height, width = work.shape
    y_start = int(height * roi_start_ratio)
    work[:y_start] = 0
    coverage = float(np.count_nonzero(work[y_start:]) / max(1, (height - y_start) * width))
    ys = np.linspace(int(height * 0.55), int(height * 0.92), 7).astype(int)
    points = []
    for y in ys:
        band = work[max(0, y - 2) : min(height, y + 3)]
        xs = np.flatnonzero(np.any(band > 0, axis=0))
        if len(xs) >= max(4, int(width * 0.03)):
            points.append((float(y), float(np.median(xs))))
    if len(points) < 4:
        return FurrowGeometry(
            FurrowEstimate(coverage=coverage), 0, (), None, None,
            "corridor is not continuous through enough horizontal bands",
        )
    y_values = np.asarray([item[0] for item in points])
    x_values = np.asarray([item[1] for item in points])
    slope, intercept = np.polyfit(y_values, x_values, 1)
    y_look, y_bottom = int(height * 0.55), int(height * 0.92)
    x_look = slope * y_look + intercept
    x_bottom = slope * y_bottom + intercept
    centre = width / 2.0
    residual = float(np.mean(np.abs(x_values - (slope * y_values + intercept))))
    confidence = _clamp((len(points) / len(ys)) * max(0.0, 1.0 - residual / (width * 0.12)))
    return FurrowGeometry(
        estimate=FurrowEstimate(
            normalized_error=_clamp((x_bottom - centre) / (width / 2.0), -1.5, 1.5),
            heading_error=_clamp((x_look - x_bottom) / (width / 2.0), -1.5, 1.5),
            confidence=confidence,
            coverage=coverage,
        ),
        line_count=0,
        boundaries=(),
        selected_pair=None,
        centerline=((int(round(x_look)), y_look), (int(round(x_bottom)), y_bottom)),
    )

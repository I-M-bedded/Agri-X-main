# -*- coding: utf-8 -*-
"""Derive the central-furrow band from a multi-row STRUCTURE mask.

The multi-line route (spec section 35): instead of trusting the directly
predicted NAV_BAND, take the segmented row/ridge structure, walk image rows
bottom-up, and put the corridor centre at the midpoint between the nearest
structure edges left/right of the tracked centre.  Works on bare seeded
ridges where a vegetation-based central line does not exist.

Used both offline (tools/build_pilot_targets.py, on ground-truth polygons)
and at runtime (sensors/ccrdnet_line_detector.py, on the predicted STRUCTURE
class).
"""

import numpy as np


def derive_nav_band(
    structure: np.ndarray,
    band_width: int,
    max_missing_rows: int = 40,
    min_gap_px: int = 4,
) -> np.ndarray:
    """Fixed-width uint8 band (1 = band) along the central corridor midline.

    ``structure``: HxW nonzero where row/ridge structure is present.
    ``max_missing_rows``: consecutive rows without both flanks before the walk
    stops (structure ended / horizon reached, scale with mask height).
    ``min_gap_px``: below this flank gap the tracked centre is inside one
    structure blob and the row is skipped.
    """

    h, w = structure.shape
    nav = np.zeros((h, w), np.uint8)
    center = w / 2.0
    half = band_width / 2.0
    misses = 0
    for y in range(h - 1, -1, -1):
        xs = np.nonzero(structure[y])[0]
        left = xs[xs < center]
        right = xs[xs > center]
        if len(left) == 0 or len(right) == 0:
            misses += 1
            if misses > max_missing_rows:
                break
            continue
        misses = 0
        l, r = left.max(), right.min()
        if r - l < min_gap_px:
            continue
        mid = (l + r) / 2.0
        # track the corridor so convergence toward the vanishing point follows
        # THIS furrow rather than jumping to a neighbour
        center = mid
        a = int(round(mid - half))
        b = int(round(mid + half))
        nav[y, max(0, a):min(w, b + 1)] = 1
    return nav

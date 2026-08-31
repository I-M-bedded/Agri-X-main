from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sensors.furrow_geometry import boundary_mask_to_geometry, corridor_mask_to_geometry


class FurrowGeometryTest(unittest.TestCase):
    def test_converging_boundaries_produce_centerline(self):
        mask = np.zeros((240, 320), np.uint8)
        for bottom, top in ((20, 105), (85, 135), (235, 185), (305, 215)):
            cv2.line(mask, (bottom, 225), (top, 80), 255, 6)
        result = boundary_mask_to_geometry(mask)
        self.assertGreaterEqual(result.line_count, 3)
        self.assertIsNotNone(result.centerline)
        self.assertLess(abs(result.estimate.normalized_error), 0.12)
        self.assertGreater(result.estimate.confidence, 0.45)

    def test_right_shift_has_positive_error(self):
        mask = np.zeros((240, 320), np.uint8)
        cv2.line(mask, (135, 225), (165, 90), 255, 7)
        cv2.line(mask, (285, 225), (220, 90), 255, 7)
        result = boundary_mask_to_geometry(mask)
        self.assertIsNotNone(result.centerline)
        self.assertGreater(result.estimate.normalized_error, 0.15)

    def test_accepts_flat_hough_output(self):
        mask = np.zeros((240, 320), np.uint8)
        segments = np.array([[80, 220, 130, 90], [240, 220, 190, 90]], np.int32)
        with patch("sensors.furrow_geometry.cv2.HoughLinesP", return_value=segments):
            result = boundary_mask_to_geometry(mask)
        self.assertEqual(result.line_count, 2)
        self.assertIsNotNone(result.centerline)

    def test_filled_corridor_centroid(self):
        mask = np.zeros((240, 320), np.uint8)
        polygon = np.array([[140, 100], [190, 100], [250, 230], [90, 230]], np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        result = corridor_mask_to_geometry(mask)
        self.assertIsNotNone(result.centerline)
        self.assertLess(abs(result.estimate.normalized_error), 0.08)
        self.assertGreater(result.estimate.confidence, 0.7)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.ccrdnet.postprocess import (
    NavigationResult,
    PostprocessConfig,
    extract_navigation_line,
)

H, W = 256, 256


def line_mask(x_bottom, x_top, thickness=8, y_bottom=250, y_top=40):
    mask = np.zeros((H, W), np.uint8)
    cv2.line(mask, (x_bottom, y_bottom), (x_top, y_top), 255, thickness)
    return mask


class PostprocessTest(unittest.TestCase):
    def test_empty_mask_returns_zero_confidence(self):
        result = extract_navigation_line(np.zeros((H, W), np.uint8))
        self.assertIsNone(result.line)
        self.assertEqual(result.estimate.confidence, 0.0)
        self.assertEqual(result.estimate.normalized_error, 0.0)

    def test_centered_vertical_line(self):
        result = extract_navigation_line(line_mask(128, 128))
        self.assertIsNotNone(result.line)
        self.assertLess(abs(result.estimate.normalized_error), 0.05)
        self.assertLess(abs(result.line.angle_deg), 2.0)
        self.assertGreater(result.estimate.confidence, 0.6)

    def test_shifted_vertical_line_sign(self):
        right = extract_navigation_line(line_mask(190, 190))
        self.assertGreater(right.estimate.normalized_error, 0.3)
        left = extract_navigation_line(line_mask(60, 60))
        self.assertLess(left.estimate.normalized_error, -0.3)

    def test_tilted_lines_recover_angle(self):
        for sign in (+1, -1):
            dx = int(np.tan(np.radians(15)) * 210)  # y spans 250 -> 40
            mask = line_mask(128, 128 + sign * dx)
            result = extract_navigation_line(mask)
            # slope is dx/dy in image coords; going up (y down) x moves by -sign*dx
            self.assertAlmostEqual(abs(result.line.angle_deg), 15.0, delta=3.0)

    def test_broken_line_still_recovered_with_scored_selection(self):
        mask = np.zeros((H, W), np.uint8)
        cv2.line(mask, (128, 250), (128, 180), 255, 8)
        cv2.line(mask, (128, 150), (128, 60), 255, 8)
        config = PostprocessConfig(component_mode="scored")
        result = extract_navigation_line(mask, config)
        self.assertIsNotNone(result.line)
        self.assertLess(abs(result.estimate.normalized_error), 0.08)

    def test_two_competing_lines_bottom_center_prefers_center(self):
        mask = line_mask(120, 120)
        # taller/larger distractor on the far left, but it ends high above bottom
        cv2.line(mask, (20, 200), (20, 10), 255, 14)
        config = PostprocessConfig(component_mode="bottom_center")
        result = extract_navigation_line(mask, config)
        self.assertLess(abs(result.line.x_near - 120), 8.0)

    def test_large_false_blob_defeats_largest_but_not_scored(self):
        mask = line_mask(128, 128, thickness=6)
        cv2.rectangle(mask, (0, 0), (90, 60), 255, -1)  # big blob, top-left
        largest = extract_navigation_line(mask, PostprocessConfig(component_mode="largest"))
        scored = extract_navigation_line(mask, PostprocessConfig(component_mode="scored"))
        self.assertGreater(abs(largest.line.x_near - 128), 10.0)
        self.assertLess(abs(scored.line.x_near - 128), 8.0)

    def test_short_near_field_line_has_low_confidence(self):
        mask = np.zeros((H, W), np.uint8)
        cv2.line(mask, (128, 250), (128, 215), 255, 8)
        result = extract_navigation_line(mask)
        self.assertIsNotNone(result.line)
        self.assertLess(result.estimate.confidence, 0.5)

    def test_curved_line_reports_high_residual(self):
        mask = np.zeros((H, W), np.uint8)
        ys = np.arange(40, 251)
        xs = (128 + 60 * np.sin((ys - 40) / 210 * np.pi)).astype(int)
        for x, y in zip(xs, ys):
            cv2.circle(mask, (x, y), 4, 255, -1)
        result = extract_navigation_line(mask)
        self.assertIsNotNone(result.line)
        self.assertGreater(result.line.fit_residual, 8.0)
        self.assertLess(result.fit_score, 0.5)

    def test_ransac_ignores_outlier_arm(self):
        mask = line_mask(128, 128, thickness=6)
        cv2.line(mask, (128, 100), (220, 60), 255, 6)  # attached diagonal arm
        ls = extract_navigation_line(mask, PostprocessConfig(fit_mode="ls"))
        ransac = extract_navigation_line(mask, PostprocessConfig(fit_mode="ransac"))
        self.assertLess(abs(ransac.line.x_near - 128), abs(ls.line.x_near - 128) + 1e-6)

    def test_deterministic(self):
        mask = line_mask(140, 100)
        for fit in ("ls", "tls", "ransac"):
            config = PostprocessConfig(fit_mode=fit)
            first = extract_navigation_line(mask, config)
            second = extract_navigation_line(mask, config)
            self.assertEqual(first.line, second.line)
            self.assertEqual(first.estimate, second.estimate)

    def test_tls_matches_ls_on_clean_line(self):
        mask = line_mask(110, 160)
        ls = extract_navigation_line(mask, PostprocessConfig(fit_mode="ls"))
        tls = extract_navigation_line(mask, PostprocessConfig(fit_mode="tls"))
        self.assertAlmostEqual(ls.line.x_near, tls.line.x_near, delta=2.0)
        self.assertAlmostEqual(ls.line.angle_deg, tls.line.angle_deg, delta=1.5)


if __name__ == "__main__":
    unittest.main()

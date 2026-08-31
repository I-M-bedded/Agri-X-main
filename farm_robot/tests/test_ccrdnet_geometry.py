from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.ccrdnet.preprocess import (
    make_transform,
    preprocess_image,
    preprocess_mask,
)

SOURCE = (640, 480)  # W, H
KNOWN_POINTS = [(0.0, 0.0), (639.0, 479.0), (320.0, 240.0), (100.0, 400.0)]


class GeometryTest(unittest.TestCase):
    def _roundtrip(self, mode, target):
        transform = make_transform(SOURCE, target, mode)
        for x, y in KNOWN_POINTS:
            mx, my = transform.source_to_model(x, y)
            self.assertTrue(0.0 <= mx <= target[0] and 0.0 <= my <= target[1])
            bx, by = transform.model_to_source(mx, my)
            self.assertAlmostEqual(bx, x, places=6)
            self.assertAlmostEqual(by, y, places=6)

    def test_stretch_roundtrip(self):
        self._roundtrip("stretch", (256, 256))

    def test_letterbox_roundtrip(self):
        self._roundtrip("letterbox", (256, 256))

    def test_fit_roundtrip(self):
        self._roundtrip("fit", (320, 240))
        self._roundtrip("fit", (256, 192))

    def test_letterbox_pads_vertically_for_4_3(self):
        transform = make_transform(SOURCE, (256, 256), "letterbox")
        self.assertAlmostEqual(transform.pad_x, 0.0)
        self.assertAlmostEqual(transform.pad_y, (256 - 480 * (256 / 640)) / 2.0)
        # source centre must stay at the model-input centre column
        mx, _ = transform.source_to_model(320.0, 240.0)
        self.assertAlmostEqual(mx, 128.0, places=4)

    def test_known_point_maps_to_expected_model_pixel(self):
        transform = make_transform(SOURCE, (256, 256), "stretch")
        mx, my = transform.source_to_model(320.0, 240.0)
        self.assertAlmostEqual(mx, 128.0, places=4)
        self.assertAlmostEqual(my, 128.0, places=4)

    def test_mask_resize_preserves_class_ids(self):
        mask = np.zeros((480, 640), np.uint8)
        mask[200:280, 300:340] = 2
        mask[100:140, 100:200] = 1
        resized, _ = preprocess_mask(mask, (256, 256), "letterbox")
        self.assertEqual(set(np.unique(resized)) - {0}, {1, 2})
        # nearest-neighbour must not invent blended ids
        self.assertTrue(np.isin(resized, (0, 1, 2)).all())

    def test_mask_geometry_survives_roundtrip(self):
        mask = np.zeros((480, 640), np.uint8)
        mask[:, 316:324] = 2  # 8px-wide vertical band at source centre
        for mode in ("stretch", "letterbox"):
            resized, transform = preprocess_mask(mask, (256, 256), mode)
            ys, xs = np.nonzero(resized == 2)
            self.assertGreater(len(xs), 0)
            centre_x = float(xs.mean())
            back_x, _ = transform.model_to_source(centre_x, float(ys.mean()))
            self.assertAlmostEqual(back_x, 319.5, delta=1.5)

    def test_image_resize_shape(self):
        image = np.zeros((480, 640, 3), np.uint8)
        for mode, target in (("stretch", (256, 256)), ("letterbox", (256, 256)), ("fit", (320, 240))):
            out, _ = preprocess_image(image, target, mode)
            self.assertEqual(out.shape[:2], (target[1], target[0]))


if __name__ == "__main__":
    unittest.main()

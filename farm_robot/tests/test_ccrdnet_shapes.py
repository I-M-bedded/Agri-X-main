from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.ccrdnet.config import CCRDNetConfig
from perception.ccrdnet.model import CCRDNet, count_macs, count_parameters


class CCRDNetShapeTest(unittest.TestCase):
    def test_output_matches_input_resolution(self):
        model = CCRDNet().eval()
        for height, width in ((256, 256), (256, 192), (320, 240)):
            with torch.no_grad():
                out = model(torch.zeros(2, 3, height, width))
            self.assertEqual(tuple(out.shape), (2, 3, height, width))

    def test_ablation_variants_run(self):
        for use_dsc in (True, False):
            for aspp_skips in (0, 3):
                config = CCRDNetConfig(use_dsc=use_dsc, aspp_skip_count=aspp_skips)
                model = CCRDNet(config).eval()
                with torch.no_grad():
                    out = model(torch.zeros(1, 3, 256, 256))
                self.assertEqual(tuple(out.shape), (1, 3, 256, 256))

    def test_default_scale_matches_paper(self):
        model = CCRDNet()
        params = count_parameters(model)
        macs = count_macs(model)
        # paper: 33,621 params / 38.226M FLOPs; stay within 10%
        self.assertLess(abs(params - 33_621) / 33_621, 0.10)
        self.assertLess(abs(macs - 38_226_000) / 38_226_000, 0.10)

    def test_gradients_flow(self):
        model = CCRDNet()
        out = model(torch.randn(1, 3, 64, 64))
        out.mean().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None for g in grads))


if __name__ == "__main__":
    unittest.main()

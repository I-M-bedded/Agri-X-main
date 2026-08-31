# -*- coding: utf-8 -*-
"""Export a trained CCRDNet checkpoint to ONNX and verify numerical agreement.

    python tools/export_ccrdnet_onnx.py --checkpoint ../runs/pilot_ft_v2/best.pt \
        --out models/ccrdnet_pilot_v2.onnx

Verification runs PyTorch and ONNX Runtime on identical random inputs and
requires max |logit difference| < 1e-4 (spec Task 7: verify agreement before
any benchmarking).
"""

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_FARM_ROBOT = os.path.abspath(os.path.join(_HERE, ".."))
if _FARM_ROBOT not in sys.path:
    sys.path.insert(0, _FARM_ROBOT)

from tools.eval_ccrdnet import load_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    model, config, ckpt = load_model(args.checkpoint, torch.device("cpu"))
    h, w = config.input_size
    dummy = torch.zeros(1, config.in_channels, h, w)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.export(
        model, (dummy,), out_path,
        input_names=["image"], output_names=["logits"],
        opset_version=17, dynamo=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(4):
        x = rng.random((1, config.in_channels, h, w), dtype=np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        out = session.run(["logits"], {"image": x})[0]
        worst = max(worst, float(np.abs(ref - out).max()))
    size_kb = os.path.getsize(out_path) / 1024
    print(f"exported {out_path} ({size_kb:.0f} KB, epoch {ckpt.get('epoch')})")
    print(f"max |pytorch - onnx| logit diff over 4 random inputs: {worst:.2e}")
    if worst >= args.tolerance:
        print(f"FAIL: exceeds tolerance {args.tolerance}")
        sys.exit(1)
    print("PARITY OK")


if __name__ == "__main__":
    main()

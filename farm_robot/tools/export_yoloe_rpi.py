# -*- coding: utf-8 -*-
"""Bake agricultural text prompts into YOLOE-26n and export NCNN for Pi 4.

Run this once on a normal development machine with internet access.  The first
YOLOE text prompt setup may download its text encoder; the exported model no
longer needs that encoder or text prompts at runtime.
"""

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sensors.ai_perception import PROMPT_CLASSES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="yoloe-26n-seg.pt")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument(
        "--output",
        default="models/agri_yoloe26n_ncnn_model",
        help="destination directory relative to farm_robot unless absolute",
    )
    return p.parse_args()


def main():
    from ultralytics import YOLOE

    args = parse_args()
    model = YOLOE(args.checkpoint)
    model.set_classes(list(PROMPT_CLASSES))

    exported = Path(model.export(format="ncnn", imgsz=args.imgsz))
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(exported), str(destination))

    print("Export complete:", destination)
    print("Baked prompts:", ", ".join(PROMPT_CLASSES))
    print("Run: python3 pipeline_main.py --model", destination)


if __name__ == "__main__":
    main()

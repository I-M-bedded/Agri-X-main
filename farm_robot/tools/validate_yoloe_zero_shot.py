# -*- coding: utf-8 -*-
"""Run YOLOE zero-shot segmentation on field images and save a compact report.

Intended for workstation / CI validation before Raspberry Pi deployment.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE

from sensors.ai_perception import FURROW_PROMPTS, PROMPT_CLASSES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--checkpoint", default="yoloe-26n-seg.pt")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--output", default="artifacts/yoloe_zero_shot")
    return p.parse_args()


def class_name(names, idx):
    if isinstance(names, dict):
        return str(names.get(int(idx), idx))
    return str(names[int(idx)])


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLOE(args.checkpoint)
    model.set_classes(list(PROMPT_CLASSES))

    report = {
        "checkpoint": args.checkpoint,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "prompts": list(PROMPT_CLASSES),
        "images": [],
    }

    for image_path in args.images:
        path = Path(image_path)
        result = model.predict(str(path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        names = result.names
        detections = []
        furrow_pixels = 0
        total_pixels = 0

        boxes = result.boxes
        masks = result.masks
        if boxes is not None and len(boxes) and masks is not None and masks.data is not None:
            cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
            confs = boxes.conf.detach().cpu().numpy().astype(float)
            mask_data = masks.data.detach().cpu().numpy()
            total_pixels = int(mask_data.shape[1] * mask_data.shape[2])
            union = np.zeros(mask_data.shape[1:], dtype=bool)

            for cls_id, score, mask in zip(cls_ids, confs, mask_data):
                label = class_name(names, cls_id).strip().lower()
                area_ratio = float((mask > 0.5).mean())
                detections.append({
                    "label": label,
                    "confidence": round(float(score), 4),
                    "mask_area_ratio": round(area_ratio, 5),
                })
                if label in FURROW_PROMPTS:
                    union |= mask > 0.5
            furrow_pixels = int(union.sum())

        annotated = result.plot()
        cv2.imwrite(str(out / f"{path.stem}_annotated.jpg"), annotated)

        report["images"].append({
            "file": path.name,
            "detections": detections,
            "furrow_detected": any(d["label"] in FURROW_PROMPTS for d in detections),
            "furrow_mask_ratio_model_input": round(
                furrow_pixels / float(total_pixels), 5
            ) if total_pixels else 0.0,
        })

    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

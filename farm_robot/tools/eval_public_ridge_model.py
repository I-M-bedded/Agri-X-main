# -*- coding: utf-8 -*-
"""Evaluate an already-trained public Ridge/Furrow segmentation model.

No text prompts or zero-shot classes are used. The evaluator sends ordinary
bare-field images to the fine-tuned Roboflow Universe model and rasterizes its
instance polygons into separate Ridge and Furrow masks.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--model-id", default="plant-jytyh/1")
    p.add_argument("--output", default="artifacts/ridge_model_eval")
    return p.parse_args()


def polygon_mask(points, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if not points:
        return mask
    poly = np.asarray(
        [[int(round(p["x"])), int(round(p["y"]))] for p in points],
        dtype=np.int32,
    )
    if len(poly) >= 3:
        cv2.fillPoly(mask, [poly], 255)
    return mask


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    diagnostic = out / "diagnostic.txt"

    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    diagnostic.write_text(
        f"model_id={args.model_id}\napi_key_present={bool(api_key)}\n",
        encoding="utf-8",
    )

    try:
        if not api_key:
            raise RuntimeError(
                "ROBOFLOW_API_KEY is not configured. The Universe project is public, "
                "but its hosted custom-model inference endpoint requires authentication."
            )

        from inference_sdk import InferenceHTTPClient

        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )

        report = {
            "model_id": args.model_id,
            "api_key_present": True,
            "images": [],
        }

        for image_name in args.images:
            path = Path(image_name)
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            h, w = image.shape[:2]

            result = client.infer(str(path), model_id=args.model_id)
            predictions = result.get("predictions", [])

            ridge_union = np.zeros((h, w), dtype=np.uint8)
            furrow_union = np.zeros((h, w), dtype=np.uint8)
            annotated = image.copy()
            rows = []

            for pred in predictions:
                label = str(pred.get("class", pred.get("class_name", ""))).strip()
                confidence = float(pred.get("confidence", 0.0))
                mask = polygon_mask(pred.get("points", []), h, w)
                low = label.lower()

                if low == "ridge":
                    ridge_union = cv2.bitwise_or(ridge_union, mask)
                elif low == "furrow":
                    furrow_union = cv2.bitwise_or(furrow_union, mask)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated, contours, -1, (255, 255, 255), 2)
                x = int(pred.get("x", 0))
                y = int(pred.get("y", 0))
                cv2.putText(
                    annotated,
                    f"{label} {confidence:.2f}",
                    (max(0, x - 80), max(20, y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                rows.append({
                    "label": label,
                    "confidence": confidence,
                    "mask_area_ratio": float((mask > 0).mean()),
                })

            # Simple visual overlay: white = predicted class mask. Separate files
            # are the important output; the original image is never treated as a mask.
            ridge_overlay = image.copy()
            ridge_overlay[ridge_union > 0] = (
                0.45 * ridge_overlay[ridge_union > 0] + 0.55 * 255
            ).astype(np.uint8)
            furrow_overlay = image.copy()
            furrow_overlay[furrow_union > 0] = (
                0.45 * furrow_overlay[furrow_union > 0] + 0.55 * 255
            ).astype(np.uint8)

            cv2.imwrite(str(out / f"{path.stem}_annotated.jpg"), annotated)
            cv2.imwrite(str(out / f"{path.stem}_ridge_mask.png"), ridge_union)
            cv2.imwrite(str(out / f"{path.stem}_furrow_mask.png"), furrow_union)
            cv2.imwrite(str(out / f"{path.stem}_ridge_overlay.jpg"), ridge_overlay)
            cv2.imwrite(str(out / f"{path.stem}_furrow_overlay.jpg"), furrow_overlay)

            report["images"].append({
                "file": path.name,
                "detections": rows,
                "ridge_mask_ratio": float((ridge_union > 0).mean()),
                "furrow_mask_ratio": float((furrow_union > 0).mean()),
            })

        report_path = out / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        with diagnostic.open("a", encoding="utf-8") as f:
            f.write("SUCCESS\n")
        print(json.dumps(report, indent=2))

    except Exception as exc:
        text = f"FAILED: {type(exc).__name__}: {exc}\n{traceback.format_exc()}\n"
        with diagnostic.open("a", encoding="utf-8") as f:
            f.write(text)
        print(text, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

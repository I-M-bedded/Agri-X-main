# -*- coding: utf-8 -*-
"""Evaluate a pre-trained public ridge/furrow segmentation model on bare-field images.

The primary target is the Roboflow Universe Plant model, which is already trained
for separate Furrow and Ridge instance masks. This script deliberately does no
prompting or zero-shot class definition.
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
    p.add_argument("--confidence", type=float, default=0.15)
    p.add_argument("--output", default="artifacts/ridge_model_eval")
    return p.parse_args()


def prediction_to_sv(prediction):
    """Convert inference-models prediction into supervision detections."""
    if hasattr(prediction, "to_supervision"):
        return prediction.to_supervision()
    if hasattr(prediction, "predictions") and hasattr(prediction.predictions, "to_supervision"):
        return prediction.predictions.to_supervision()
    raise TypeError(f"No to_supervision() method on {type(prediction)!r}")


def class_names_from_model(model):
    for attr in ("class_names", "classes", "names"):
        value = getattr(model, attr, None)
        if value:
            return value
    return None


def label_for(class_id, class_names):
    if class_names is None:
        return str(int(class_id))
    if isinstance(class_names, dict):
        return str(class_names.get(int(class_id), class_id))
    try:
        return str(class_names[int(class_id)])
    except Exception:
        return str(int(class_id))


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    diagnostic = out / "diagnostic.txt"

    api_key = os.getenv("ROBOFLOW_API_KEY") or None
    try:
        from inference_models import AutoModel
        import supervision as sv

        diagnostic.write_text(
            f"model_id={args.model_id}\napi_key_present={bool(api_key)}\n",
            encoding="utf-8",
        )

        # Public custom models may still require an API key for the Roboflow
        # platform weight download. Pass a configured key if the repo has one,
        # but never print it.
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        model = AutoModel.from_pretrained(args.model_id, **kwargs)
        class_names = class_names_from_model(model)

        report = {
            "model_id": args.model_id,
            "confidence": args.confidence,
            "api_key_present": bool(api_key),
            "model_type": type(model).__name__,
            "class_names": class_names,
            "images": [],
        }

        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()

        for image_name in args.images:
            path = Path(image_name)
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)

            raw = model(image, confidence=args.confidence)
            prediction = raw[0] if isinstance(raw, (list, tuple)) else raw
            detections = prediction_to_sv(prediction)

            image_rows = []
            ridge_union = np.zeros(image.shape[:2], dtype=np.uint8)
            furrow_union = np.zeros(image.shape[:2], dtype=np.uint8)

            masks = getattr(detections, "mask", None)
            class_ids = getattr(detections, "class_id", None)
            confidences = getattr(detections, "confidence", None)
            n = len(detections)

            labels = []
            for i in range(n):
                cid = int(class_ids[i]) if class_ids is not None else -1
                name = label_for(cid, class_names).strip()
                conf = float(confidences[i]) if confidences is not None else None
                labels.append(f"{name} {conf:.2f}" if conf is not None else name)

                area_ratio = None
                if masks is not None:
                    mask = np.asarray(masks[i], dtype=bool)
                    if mask.shape != image.shape[:2]:
                        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                    area_ratio = float(mask.mean())
                    low = name.lower()
                    if low == "ridge":
                        ridge_union[mask] = 255
                    elif low == "furrow":
                        furrow_union[mask] = 255

                image_rows.append({
                    "class_id": cid,
                    "label": name,
                    "confidence": conf,
                    "mask_area_ratio": area_ratio,
                })

            annotated = mask_annotator.annotate(scene=image.copy(), detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
            cv2.imwrite(str(out / f"{path.stem}_annotated.jpg"), annotated)
            cv2.imwrite(str(out / f"{path.stem}_ridge_mask.png"), ridge_union)
            cv2.imwrite(str(out / f"{path.stem}_furrow_mask.png"), furrow_union)

            report["images"].append({
                "file": path.name,
                "detections": image_rows,
                "ridge_mask_ratio": float((ridge_union > 0).mean()),
                "furrow_mask_ratio": float((furrow_union > 0).mean()),
            })

        (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        with diagnostic.open("a", encoding="utf-8") as f:
            f.write("SUCCESS\n")
            f.write(json.dumps(report, indent=2, default=str))
            f.write("\n")
        print(json.dumps(report, indent=2, default=str))

    except Exception as exc:
        text = f"FAILED: {type(exc).__name__}: {exc}\n{traceback.format_exc()}\n"
        with diagnostic.open("a", encoding="utf-8") as f:
            f.write(text)
        print(text, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

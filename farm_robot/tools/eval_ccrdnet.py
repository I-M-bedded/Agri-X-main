# -*- coding: utf-8 -*-
"""Evaluate a trained CCRDNet checkpoint on the test split and write a report.

    python tools/eval_ccrdnet.py --data-root ../data/ccrdnet/prepared \
        --checkpoint ../runs/<run>/best.pt --out ../reports/<experiment>

Produces metrics.json, per-frame metrics_frames.csv, overlay images for a
sample of frames plus the worst failures, and summary.md (spec sections 30-31).
"""

import argparse
import csv
import json
import math
import os
import sys

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_FARM_ROBOT = os.path.abspath(os.path.join(_HERE, ".."))
if _FARM_ROBOT not in sys.path:
    sys.path.insert(0, _FARM_ROBOT)

from perception.ccrdnet.config import CCRDNetConfig  # noqa: E402
from perception.ccrdnet.model import CCRDNet  # noqa: E402
from perception.ccrdnet.postprocess import (  # noqa: E402
    PostprocessConfig, extract_navigation_line,
)
from tools.ccrdnet_data import CCRDDataset  # noqa: E402
from tools.ccrdnet_metrics import compute_frame_metrics, summarize  # noqa: E402

# overlay colours (BGR)
COL_GT_BAND = (60, 200, 60)
COL_PRED_BAND = (60, 60, 230)
COL_STRUCT = (200, 140, 40)
COL_GT_LINE = (0, 255, 0)
COL_PRED_LINE = (0, 0, 255)


def load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    fields = set(CCRDNetConfig.__dataclass_fields__)
    kwargs = {}
    for k, v in cfg_dict.items():
        if k in fields:
            kwargs[k] = tuple(v) if isinstance(v, list) else v
    config = CCRDNetConfig(**kwargs)
    model = CCRDNet(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, ckpt


def draw_overlay(image_rgb, gt_mask, pred_mask, gt_res, pred_res, fm, config):
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    over = img.copy()
    over[gt_mask == config.class_structure] = COL_STRUCT
    over[gt_mask == config.class_nav_band] = COL_GT_BAND
    over[pred_mask == config.class_nav_band] = COL_PRED_BAND
    both = (gt_mask == config.class_nav_band) & (pred_mask == config.class_nav_band)
    over[both] = (40, 220, 220)
    img = cv2.addWeighted(over, 0.45, img, 0.55, 0)

    h, w = gt_mask.shape
    for res, colour in ((gt_res, COL_GT_LINE), (pred_res, COL_PRED_LINE)):
        if res.line is None:
            continue
        line = res.line
        x0 = int(round(line.slope * 0 + line.intercept))
        x1 = int(round(line.slope * (h - 1) + line.intercept))
        cv2.line(img, (x0, 0), (x1, h - 1), colour, 2)
        cv2.circle(img, (int(round(line.x_near)), int(round(line.y_near))), 4, colour, -1)
        cv2.circle(img, (int(round(line.x_far)), int(round(line.y_far))), 4, colour, 1)

    ae = "-" if math.isnan(fm.angle_error) else f"{fm.angle_error:.2f}deg"
    lat = "-" if math.isnan(fm.lateral_near) else f"{fm.lateral_near:.1f}px"
    iou = "-" if math.isnan(fm.line_iou) else f"{fm.line_iou:.2f}"
    for i, text in enumerate(
        (f"conf {fm.confidence:.2f}  AE {ae}", f"lat {lat}  lineIoU {iou}")
    ):
        cv2.putText(img, text, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--line-width", type=int, default=6)
    parser.add_argument("--num-overlays", type=int, default=24)
    parser.add_argument("--num-failures", type=int, default=16)
    parser.add_argument("--component-mode", default="largest",
                        choices=["largest", "bottom_center", "scored"])
    parser.add_argument("--fit-mode", default="ls", choices=["ls", "tls", "ransac"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, ckpt = load_model(args.checkpoint, device)
    postprocess = PostprocessConfig(
        component_mode=args.component_mode, fit_mode=args.fit_mode
    )

    dataset = CCRDDataset(args.data_root, args.split, config.input_size, augment=False)
    print(f"{args.split} frames: {len(dataset)}  device={device}")

    os.makedirs(os.path.join(args.out, "overlays"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "failure_cases"), exist_ok=True)

    frames = []
    per_frame_art = {}
    with torch.no_grad():
        for idx in range(len(dataset)):
            tensor, target, name = dataset[idx]
            logits = model(tensor.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)[0]
            pred = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)
            nav_prob = probs[config.class_nav_band].cpu().numpy()
            gt = target.numpy().astype(np.uint8)
            fm = compute_frame_metrics(
                name, pred, gt, config.class_nav_band, postprocess,
                args.line_width, nav_probability=nav_prob,
            )
            frames.append(fm)
            image_rgb = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            per_frame_art[name] = (image_rgb, gt, pred, nav_prob)

    summary = summarize(frames)
    summary["checkpoint"] = os.path.abspath(args.checkpoint)
    summary["checkpoint_epoch"] = ckpt.get("epoch")
    summary["split"] = args.split
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.out, "metrics_frames.csv"), "w", newline="",
              encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "gt_ok", "pred_ok", "angle_error", "lateral_near",
                         "lateral_far", "line_iou", "nav_iou", "confidence", "accurate"])
        for fm in frames:
            writer.writerow([fm.name, fm.gt_ok, fm.pred_ok,
                             f"{fm.angle_error:.4f}", f"{fm.lateral_near:.2f}",
                             f"{fm.lateral_far:.2f}", f"{fm.line_iou:.4f}",
                             f"{fm.nav_iou:.4f}", f"{fm.confidence:.3f}", fm.accurate])

    def save_overlay(fm, directory):
        image_rgb, gt, pred, nav_prob = per_frame_art[fm.name]
        gt_res = extract_navigation_line(
            (gt == config.class_nav_band).astype(np.uint8), postprocess)
        pred_res = extract_navigation_line(
            (pred == config.class_nav_band).astype(np.uint8), postprocess, nav_prob)
        img = draw_overlay(image_rgb, gt, pred, gt_res, pred_res, fm, config)
        cv2.imwrite(os.path.join(args.out, directory, f"{fm.name}.jpg"), img)

    # evenly sampled overlays across the split
    ok_frames = [f for f in frames if f.gt_ok]
    step = max(1, len(ok_frames) // max(1, args.num_overlays))
    sampled = ok_frames[::step][: args.num_overlays]
    for fm in sampled:
        save_overlay(fm, "overlays")

    # worst failures: no line predicted first, then largest angle error
    def badness(f):
        if not f.pred_ok:
            return (2, 0.0)
        return (1 if not f.accurate else 0,
                0.0 if math.isnan(f.angle_error) else f.angle_error)

    worst = sorted(ok_frames, key=badness, reverse=True)[: args.num_failures]
    for fm in worst:
        save_overlay(fm, "failure_cases")

    ae = summary["angle_error_deg"]
    lat = summary["lateral_near_px"]
    iou = summary["line_iou"]
    lines = [
        "# CCRDNet evaluation summary",
        "",
        f"- checkpoint: `{args.checkpoint}` (epoch {ckpt.get('epoch')})",
        f"- split: {args.split}, {summary['frames_total']} frames "
        f"({summary['frames_with_gt_line']} with a GT line)",
        f"- detection rate: {summary['detection_rate']:.4f}"
        if summary["detection_rate"] is not None else "- detection rate: n/a",
        "",
        "| metric | mean | median | P95 |",
        "|---|---:|---:|---:|",
        f"| angle error (deg) | {ae['mean']:.3f} | {ae['median']:.3f} | {ae['p95']:.3f} |"
        if ae["mean"] is not None else "| angle error | - | - | - |",
        f"| lateral near (px) | {lat['mean']:.2f} | {lat['median']:.2f} | {lat['p95']:.2f} |"
        if lat["mean"] is not None else "| lateral near | - | - | - |",
        f"| line IoU | {iou['mean']:.4f} | {iou['median']:.4f} | - |"
        if iou["mean"] is not None else "| line IoU | - | - | - |",
        "",
        f"line accuracy ({summary['accuracy_criterion']}): "
        f"{summary['line_accuracy']:.4f}"
        if summary["line_accuracy"] is not None else "line accuracy: n/a",
        "",
        "Overlays in `overlays/`, worst cases in `failure_cases/`.",
        "Colours: green = GT band/line, red = predicted band/line, "
        "yellow = overlap, orange = STRUCTURE (GT).",
    ]
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

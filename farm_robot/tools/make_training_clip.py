# -*- coding: utf-8 -*-
"""
farm_robot/tools/make_training_clip.py
---------------------------------------
학습 데이터와 학습 진행 상황을 영상 프레임으로 렌더한다.

    A 구간  데이터셋 샘플 : 원본 | 3클래스 라벨 오버레이
    B 구간  학습 곡선     : --curve-frames 로 켠다(기본 꺼짐)

한글을 쓰므로 텍스트는 전부 PIL 로 그린다(OpenCV 는 한글을 못 그린다).

    python tools/make_training_clip.py --out ../media/frames_train
"""

import argparse
import csv
import json
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
BG = (16, 18, 22)
FG = (238, 240, 244)
DIM = (150, 156, 166)
ACCENT = (86, 200, 120)
COL_STRUCT = (255, 176, 60)      # RGB - 작물열
COL_BAND = (120, 235, 120)       # RGB - 고랑 주행띠
COL_TRAIN = (110, 170, 255)
COL_VAL = (255, 130, 110)
COL_IOU = (86, 200, 120)

FONT_DIR = "C:/Windows/Fonts"


def font(size, bold=False):
    return ImageFont.truetype(
        f"{FONT_DIR}/{'malgunbd.ttf' if bold else 'malgun.ttf'}", size)


def new_canvas():
    img = Image.new("RGB", (W, H), BG)
    ImageDraw.Draw(img).rectangle([0, 0, W, 5], fill=ACCENT)
    return img


def paste_cv(pil, bgr, box):
    """OpenCV BGR 이미지를 (x, y, w, h) 영역에 비율 유지로 얹는다."""
    x, y, bw, bh = box
    h, w = bgr.shape[:2]
    s = min(bw / w, bh / h)
    nw, nh = int(w * s), int(h * s)
    small = cv2.resize(bgr, (nw, nh))
    pil.paste(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)),
              (x + (bw - nw) // 2, y + (bh - nh) // 2))


def dataset_frames(root, out, count, hold):
    """A 구간: 원본 | 라벨 오버레이."""
    imgs = sorted((root / "images").glob("train_*.jpg"),
                  key=lambda p: int(p.stem.split("_")[1]))
    rng = random.Random(7)
    picks = rng.sample(imgs, min(count, len(imgs)))
    picks.sort(key=lambda p: int(p.stem.split("_")[1]))

    n = 0
    for path in picks:
        bgr = cv2.imread(str(path))
        mask = cv2.imread(str(root / "masks" / f"{path.stem}.png"), 0)
        if bgr is None or mask is None:
            continue

        overlay = bgr.copy()
        for cls, rgb in ((1, COL_STRUCT), (2, COL_BAND)):
            m = mask == cls
            if m.any():
                overlay[m] = (overlay[m] * 0.35
                              + np.array(rgb[::-1], np.float32) * 0.65).astype(np.uint8)

        pil = new_canvas()
        d = ImageDraw.Draw(pil)
        d.text((44, 34), "학습 데이터  ·  CRDLD v2.1", font=font(38, True), fill=FG)
        d.text((44, 84), "작물열을 6px 선으로 라벨 → 인접한 두 열 사이를 고랑 주행띠로 생성",
               font=font(21), fill=DIM)

        paste_cv(pil, bgr, (44, 140, 580, 440))
        paste_cv(pil, overlay, (656, 140, 580, 440))
        d.text((44, 596), "원본", font=font(24, True), fill=FG)
        d.text((656, 596), "3클래스 라벨", font=font(24, True), fill=FG)

        # 범례
        lx = 656 + 190
        d.rectangle([lx, 600, lx + 22, 620], fill=COL_STRUCT)
        d.text((lx + 32, 598), "작물열", font=font(21), fill=FG)
        d.rectangle([lx + 130, 600, lx + 152, 620], fill=COL_BAND)
        d.text((lx + 162, 598), "고랑 주행띠", font=font(21), fill=FG)

        d.text((44, 648), f"train 1,250장  ·  val 250장  ·  test 430장   ({path.stem})",
               font=font(20), fill=DIM)

        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        for _ in range(hold):
            cv2.imwrite(str(out / f"{n:05d}.png"), frame)
            n += 1
    return n


def curve_frames(run_dir, out, start_index, frames):
    """B 구간: 학습 곡선을 에폭 순서로 드러낸다."""
    rows = list(csv.DictReader(open(run_dir / "metrics.csv", encoding="utf-8")))
    cfg = json.load(open(run_dir / "run_config.json", encoding="utf-8"))
    ep = [int(r["epoch"]) for r in rows]
    tl = [float(r["train_loss"]) for r in rows]
    vl = [float(r["val_loss"]) for r in rows]
    iou = [float(r["val_line_iou"]) for r in rows]
    best_i = int(np.argmax(iou))

    px, py, pw, ph = 96, 178, W - 200, 400
    lmax = max(max(tl), max(vl))
    imax = max(iou) * 1.15

    def to_xy(i, v, vmax):
        return (px + pw * i / max(1, len(ep) - 1), py + ph * (1 - v / vmax))

    n = start_index
    for f in range(frames):
        k = max(2, int(round(len(ep) * (f + 1) / frames)))
        pil = new_canvas()
        d = ImageDraw.Draw(pil)
        d.text((44, 34), "학습  ·  CCRDNet 재현", font=font(38, True), fill=FG)
        d.text((44, 84),
               f"파라미터 {cfg['parameters']:,}개  ·  MACs {cfg['macs_256']/1e6:.1f}M  "
               f"·  가중 CE 1:4:4  ·  {cfg['device']}",
               font=font(21), fill=DIM)

        # 격자 + 축
        for g in range(5):
            gy = py + ph * g / 4
            d.line([px, gy, px + pw, gy], fill=(40, 44, 50))
        d.line([px, py, px, py + ph], fill=(90, 96, 106))
        d.line([px, py + ph, px + pw, py + ph], fill=(90, 96, 106))

        for series, col, vmax in ((tl, COL_TRAIN, lmax), (vl, COL_VAL, lmax),
                                  (iou, COL_IOU, imax)):
            pts = [to_xy(i, series[i], vmax) for i in range(k)]
            if len(pts) > 1:
                d.line(pts, fill=col, width=3, joint="curve")

        # 최고 성능 지점
        if k > best_i:
            bx, by = to_xy(best_i, iou[best_i], imax)
            d.ellipse([bx - 7, by - 7, bx + 7, by + 7], outline=COL_IOU, width=3)
            d.text((bx + 14, by - 30),
                   f"best  epoch {ep[best_i]}   line IoU {iou[best_i]:.3f}",
                   font=font(20, True), fill=COL_IOU)

        d.text((px, py + ph + 14), "epoch", font=font(19), fill=DIM)
        d.text((px + pw - 60, py + ph + 14), str(ep[k - 1]), font=font(19), fill=DIM)
        # 두 축(왼쪽 loss / 오른쪽 IoU)의 최댓값을 적어 곡선을 오해하지 않게 한다
        d.text((px - 52, py - 6), f"{lmax:.1f}", font=font(17), fill=DIM)
        d.text((px - 52, py + 16), "loss", font=font(16), fill=DIM)
        d.text((px + pw + 10, py - 6), f"{imax:.2f}", font=font(17), fill=DIM)
        d.text((px + pw + 10, py + 16), "IoU", font=font(16), fill=DIM)

        ly = 620
        for col, label in ((COL_TRAIN, "train loss"), (COL_VAL, "val loss"),
                           (COL_IOU, "val line IoU")):
            d.line([px, ly + 10, px + 30, ly + 10], fill=col, width=4)
            d.text((px + 40, ly), label, font=font(21), fill=FG)
            px_shift = d.textlength(label, font=font(21))
            px += 40 + px_shift + 46
        px = 96

        d.text((96, 672),
               f"epoch {ep[k-1]} / {ep[-1]}    "
               f"train {tl[k-1]:.3f}   val {vl[k-1]:.3f}   IoU {iou[k-1]:.3f}",
               font=font(21), fill=DIM)

        cv2.imwrite(str(out / f"{n:05d}.png"),
                    cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--data", default=str(root / "data" / "crdld" / "prepared"))
    ap.add_argument("--run", default=str(root / "runs" / "crdld_furrow_v1"))
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--hold", type=int, default=24, help="샘플 1장당 프레임 수")
    ap.add_argument("--curve-frames", type=int, default=0,
                    help="학습 곡선 구간 프레임 수. 0 이면 데이터셋만 렌더")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        for f in out.glob("*.png"):
            f.unlink()
    out.mkdir(parents=True, exist_ok=True)

    n = dataset_frames(Path(args.data), out, args.samples, args.hold)
    if args.curve_frames > 0:
        n = curve_frames(Path(args.run), out, n, args.curve_frames)
    print(f"프레임 {n}장 -> {out}")


if __name__ == "__main__":
    main()

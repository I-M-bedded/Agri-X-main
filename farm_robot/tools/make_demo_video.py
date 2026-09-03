# -*- coding: utf-8 -*-
"""
farm_robot/tools/make_demo_video.py
------------------------------------
학습 데이터/학습 + Gazebo 주행 실험을 1분짜리 영상으로 합친다.

    소재 1  media/frames_train      <- tools/make_training_clip.py
    소재 2  sim_gazebo/out/frames_gz <- policy_harness.py --record
    카드    이 스크립트가 PIL 로 생성 (OpenCV 는 한글을 못 그린다)

구성 (30fps, 1280x720)

    0:00 타이틀            5s
    0:05 1. 학습 데이터     4s 카드 + 16s 클립
    0:25 2. Gazebo 실험    4s 카드 + 31s 클립
                          --------
                          60s

    python tools/make_demo_video.py --out ../media/agrix_demo.mp4
"""

import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
BG = (16, 18, 22)
FG = (238, 240, 244)
ACCENT = (86, 200, 120)
DIM = (150, 156, 166)

FONT_DIR = "C:/Windows/Fonts"


def font(size, bold=False):
    name = "malgunbd.ttf" if bold else "malgun.ttf"
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def card(lines, seconds, out_dir, start_index):
    """lines = [(텍스트, 크기, 볼드, 색)]. 가운데 정렬 카드."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # 상단 악센트 바
    d.rectangle([0, 0, W, 6], fill=ACCENT)

    heights, rendered = [], []
    for text, size, bold, col in lines:
        f = font(size, bold)
        bbox = d.textbbox((0, 0), text, font=f)
        rendered.append((text, f, col, bbox[2] - bbox[0], bbox[3] - bbox[1]))
        heights.append(bbox[3] - bbox[1] + int(size * 0.55))
    y = (H - sum(heights)) // 2
    for (text, f, col, tw, th), hgt in zip(rendered, heights):
        d.text(((W - tw) // 2, y), text, font=f, fill=col)
        y += hgt

    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    n = int(seconds * FPS)
    for i in range(n):
        cv2.imwrite(str(out_dir / f"{start_index + i:05d}.png"), frame)
    return start_index + n


def letterbox(img):
    """비율 유지하며 캔버스 전체에 얹는다."""
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    nw, nh = int(w * s), int(h * s)
    canvas = np.full((H, W, 3), BG[::-1], np.uint8)
    canvas[(H - nh) // 2:(H - nh) // 2 + nh,
           (W - nw) // 2:(W - nw) // 2 + nw] = cv2.resize(img, (nw, nh))
    return canvas


def clip(src_dir, seconds, out_dir, start_index, every=1):
    files = sorted(Path(src_dir).glob("*.png"))[::every]
    if not files:
        raise SystemExit(f"소재 없음: {src_dir}")
    n = int(seconds * FPS)
    for i in range(n):
        # 소재 길이에 맞춰 균등 샘플링 -> 요청한 초 수를 정확히 채운다
        f = files[min(len(files) - 1, int(i * len(files) / n))]
        img = cv2.imread(str(f))
        if img is None:
            continue
        cv2.imwrite(str(out_dir / f"{start_index + i:05d}.png"), letterbox(img))
    return start_index + n


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--train", default=str(root / "media" / "frames_train"))
    ap.add_argument("--gazebo", default=str(root / "sim_gazebo" / "out" / "frames_gz"))
    ap.add_argument("--work", default=str(root / "media" / "_seq"))
    ap.add_argument("--out", default=str(root / "media" / "agrix_demo.mp4"))
    args = ap.parse_args()

    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    i = 0
    i = card([
        ("Agri-X", 92, True, FG),
        ("데이터 수집 / 비전 학습 / 알고리즘 검증", 34, False, FG),
        ("", 10, False, FG),
        ("CCRDNet / CRDLD 비전 · 1D ToF 융합 · Gazebo 폐루프 실험", 24, False, DIM),
    ], 5, work, i)

    i = card([
        ("1. 학습 데이터", 62, True, ACCENT),
        ("", 12, False, FG),
        ("CRDLD v2.1 작물열 라벨에서 고랑 주행띠를 생성해 3클래스로 학습", 26, False, FG),
        ("CCRDNet 재현 · 33,694 파라미터 · 37.6M MACs", 24, False, DIM),
    ], 4, work, i)
    i = clip(args.train, 16, work, i)

    i = card([
        ("2. Gazebo 주행 실험", 64, True, ACCENT),
        ("", 12, False, FG),
        ("ArUco 검출기 + 1D ToF + 실측 근사 비전 오차", 26, False, FG),
        ("탐색 → 진입 → 고랑 추종 → 유턴 → 재진입 → 복귀", 24, False, DIM),
    ], 4, work, i)
    i = clip(args.gazebo, 31, work, i)

    print(f"프레임 {i}장 ({i / FPS:.1f}초) 생성 완료")

    cmd = ["ffmpeg", "-y", "-framerate", str(FPS),
           "-i", str(work / "%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-movflags", "+faststart", args.out]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    size = os.path.getsize(args.out) / 1e6
    print(f"영상 저장: {args.out}  ({size:.1f} MB, {i / FPS:.0f}초)")


if __name__ == "__main__":
    main()

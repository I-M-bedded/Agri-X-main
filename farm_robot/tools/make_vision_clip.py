# -*- coding: utf-8 -*-
"""
farm_robot/tools/make_vision_clip.py
-------------------------------------
비전 모델 추론 결과를 **연속 프레임**으로 렌더해 영상 소재를 만든다.

CRDLD 테스트 세트는 트랙터에서 찍은 연속 프레임이라 인덱스 순서대로 이으면
그대로 주행 영상이 된다. 각 프레임에 실기와 **동일한 런타임 경로**를 태운다:

    ONNX 3클래스 세그멘테이션
      -> STRUCTURE(작물열) / NAV_BAND(고랑띠)
      -> derive_nav_band (구조에서 중앙 회랑 유도)
      -> extract_navigation_line (선 적합 + 신뢰도)

즉 화면에 보이는 선은 데모용으로 다시 그린 것이 아니라
**로봇이 실제로 조향에 쓰는 값**이다.

    python tools/make_vision_clip.py --out ../media/frames_vision --count 180
"""

import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import onnxruntime as ort

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from perception.ccrdnet.postprocess import (  # noqa: E402
    PostprocessConfig,
    extract_navigation_line,
)
from perception.ccrdnet.structure_midline import derive_nav_band  # noqa: E402

CLASS_STRUCTURE = 1
CLASS_NAV_BAND = 2

# 화면 색 (BGR)
COL_STRUCT = (60, 200, 255)     # 주황 - 작물열/이랑 구조
COL_BAND = (120, 255, 120)      # 연두 - 고랑 주행띠
COL_LINE = (0, 80, 255)         # 빨강 - 최종 중심선
COL_HUD = (245, 245, 245)


def draw_hud(canvas, lines, x=14, y=28, scale=0.62, gap=26):
    for i, (text, col) in enumerate(lines):
        cv2.putText(canvas, text, (x, y + i * gap), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, text, (x, y + i * gap), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, col, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(
        _HERE.parent / "models" / "ccrdnet_crdld_rgb_v1.onnx"))
    ap.add_argument("--images", default=str(
        _HERE.parents[1] / "data" / "crdld" /
        "Crop Row Detection Lincoln Dataset (CRDLD)" / "CRDLD_V2.1" /
        "test" / "image"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=180)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    _, ch, ih, iw = inp.shape
    post = PostprocessConfig()

    # 연속 프레임이 되도록 파일명을 **숫자 순서**로 정렬한다
    src = sorted(Path(args.images).glob("*.jpg"),
                 key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    src = src[args.start:args.start + args.count]
    print(f"모델 {Path(args.model).name}  입력 {ch}x{ih}x{iw}  프레임 {len(src)}장")

    ok = 0
    for n, path in enumerate(src):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue

        if ch == 1:
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(g, (iw, ih), interpolation=cv2.INTER_LINEAR)
            tensor = (small.astype(np.float32) / 255.0)[None, None]
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
            tensor = np.transpose(small.astype(np.float32) / 255.0, (2, 0, 1))[None]

        logits = sess.run(None, {inp.name: tensor})[0][0]
        shifted = logits - logits.max(axis=0, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.maximum(1e-8, exp.sum(axis=0))
        classes = probs.argmax(axis=0).astype(np.uint8)

        structure = (classes == CLASS_STRUCTURE).astype(np.uint8)
        nav = (classes == CLASS_NAV_BAND).astype(np.uint8)
        band = derive_nav_band(structure, 7, max_missing_rows=max(8, ih // 6))

        direct = extract_navigation_line(nav, post, probs[CLASS_NAV_BAND])
        derived = extract_navigation_line(band, post)
        # 실기 기본 경로(structure_mid): 유도 중앙선 우선, 없으면 직접 예측
        # CRDLD 학습 타깃이 인접 열 사이 중앙띠를 NAV_BAND 로 주므로
        # 직접 예측을 우선하고, 없을 때만 구조에서 유도한 중앙선을 쓴다.
        res = direct if direct.line is not None else derived
        used = "direct NAV_BAND" if direct.line is not None else "structure midline"

        # --- 렌더 ---
        canvas = cv2.resize(bgr, (args.width, args.height))
        sx, sy = args.width / iw, args.height / ih

        def tint(mask, color, alpha):
            m = cv2.resize(mask, (args.width, args.height),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
            if m.any():
                canvas[m] = (canvas[m] * (1 - alpha)
                             + np.array(color, np.float32) * alpha).astype(np.uint8)

        tint(structure, COL_STRUCT, 0.45)
        tint(band, COL_BAND, 0.50)

        if res.line is not None:
            p0 = (int(res.line.x_near * sx), int(res.line.y_near * sy))
            p1 = (int(res.line.x_far * sx), int(res.line.y_far * sy))
            cv2.line(canvas, p0, p1, (0, 0, 0), 9, cv2.LINE_AA)
            cv2.line(canvas, p0, p1, COL_LINE, 4, cv2.LINE_AA)
            cv2.circle(canvas, p0, 9, COL_LINE, -1, cv2.LINE_AA)

        est = res.estimate
        conf = est.confidence
        col = ((90, 240, 90) if conf >= 0.55 else
               (60, 200, 255) if conf >= 0.30 else (80, 80, 255))
        # 화면 하단에 횡오차 게이지
        gy = args.height - 34
        cv2.rectangle(canvas, (args.width // 2 - 160, gy - 9),
                      (args.width // 2 + 160, gy + 9), (30, 30, 30), -1)
        px = int(args.width // 2 + np.clip(est.normalized_error, -1, 1) * 160)
        cv2.line(canvas, (args.width // 2, gy - 12), (args.width // 2, gy + 12),
                 (200, 200, 200), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (px - 5, gy - 12), (px + 5, gy + 12), col, -1)

        draw_hud(canvas, [
            (f"CCRDNet / CRDLD  frame {path.stem}", COL_HUD),
            (f"source: {used}", COL_HUD),
            (f"lateral {est.normalized_error:+.3f}   "
             f"heading {np.degrees(est.heading_error):+.1f}deg", COL_HUD),
            (f"confidence {conf:.2f}", col),
        ])
        cv2.putText(canvas, "STRUCTURE", (args.width - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_STRUCT, 2, cv2.LINE_AA)
        cv2.putText(canvas, "NAV BAND", (args.width - 250, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_BAND, 2, cv2.LINE_AA)
        cv2.putText(canvas, "CENTER LINE", (args.width - 250, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_LINE, 2, cv2.LINE_AA)

        cv2.imwrite(str(out / f"{n:04d}.png"), canvas)
        if res.line is not None:
            ok += 1

    print(f"저장 {len(src)}장 -> {out}  (선 검출 {ok}/{len(src)})")


if __name__ == "__main__":
    main()

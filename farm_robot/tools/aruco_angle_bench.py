# -*- coding: utf-8 -*-
"""
tools/aruco_angle_bench.py
---------------------------
**실제 OpenCV ArUco 검출기**로 "팻말을 몇 도까지, 몇 미터까지 인식하는가"를
직접 측정한다.

왜 필요한가
  2D 시뮬레이터(tools/simulation.py)의 마커 가시성 모델은 조잡하다:
    - 면각 70도 하드컷 하나뿐 (실제로는 연속적으로 열화)
    - 거리에 따른 **화면상 마커 크기**를 전혀 모델링하지 않는다
    - 자세(yaw) 추정 정확도는 검출보다 훨씬 빨리 무너지는데 그것도 없다
    - 모션 블러/조명/초점 없음
  그래서 시뮬의 팻말 각도 비교(0 vs 30 vs 45도)는 **기하학적 차단만** 본 것이고,
  "0도 팻말을 비스듬히 다가가며 실제로 인식하는가"는 답하지 못한다.

방법
  마커 이미지를 만들어 **원근 투영**으로 카메라 평면에 렌더링한 뒤
  cv2.aruco.detectMarkers 를 그대로 돌린다. 카메라 내부 파라미터는
  config(해상도, 화각)에서 만든다. 즉 우리 로봇이 쓸 검출기 그대로다.

한계 (여전히 남는 것)
  조명/그림자/모션블러/렌즈 왜곡/인쇄 품질은 모델링하지 않는다.
  따라서 여기 숫자는 **낙관적 상한**이다. 실기에서는 이보다 나쁘다.
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ARUCO_DICTIONARY, CAMERA_RESOLUTION, MARKER_SIZE_M  # noqa: E402

HFOV_DEG = 62.0          # Pi Camera v2 수평 화각
MARKER_PX = 600          # 렌더링용 마커 원본 해상도


def camera_matrix(width, height, hfov_deg=HFOV_DEG):
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0, width / 2.0],
                     [0, fx, height / 2.0],
                     [0, 0, 1.0]], np.float64)


def render(marker_img, size_m, distance_m, view_angle_deg, K, out_wh):
    """마커를 view_angle 만큼 돌려서 distance 앞에 놓고 원근 투영한다."""
    w, h = out_wh
    half = size_m / 2.0
    a = math.radians(view_angle_deg)
    # 마커 평면을 y축(수직) 기준으로 회전 -> 비스듬히 본 상태
    corners3d = []
    for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1)):   # TL,TR,BR,BL
        X, Y = sx * half, sy * half
        corners3d.append([X * math.cos(a), Y, distance_m - X * math.sin(a)])
    pts = []
    for X, Y, Z in corners3d:
        if Z <= 1e-6:
            return None
        u = K[0, 0] * X / Z + K[0, 2]
        v = K[1, 1] * (-Y) / Z + K[1, 2]
        pts.append([u, v])
    dst = np.array(pts, np.float32)
    src = np.array([[0, 0], [MARKER_PX - 1, 0],
                    [MARKER_PX - 1, MARKER_PX - 1], [0, MARKER_PX - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    canvas = np.full((h, w), 255, np.uint8)      # 흰 배경
    warped = cv2.warpPerspective(marker_img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_TRANSPARENT, dst=canvas)
    return warped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=float, default=MARKER_SIZE_M)
    args = ap.parse_args()

    w, h = CAMERA_RESOLUTION
    K = camera_matrix(w, h)
    dist = np.zeros((5, 1))
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, ARUCO_DICTIONARY))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    marker = cv2.aruco.generateImageMarker(dictionary, 1, MARKER_PX)

    angles = list(range(0, 90, 10))
    dists = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)

    print("=" * 92)
    print(f"실제 ArUco 검출기 측정  (마커 {args.size*100:.0f}cm, "
          f"{w}x{h}, 화각 {HFOV_DEG:.0f}도, {ARUCO_DICTIONARY})")
    print("  값 = 검출 O/X, 괄호 안은 yaw 추정 오차(도)")
    print("=" * 92)
    header = "  거리\\각도 |" + "".join(f"{a:>8d}도" for a in angles)
    print(header); print("  " + "-" * (len(header) - 2))

    for d in dists:
        row = f"  {d:5.1f}m   |"
        for ang in angles:
            img = render(marker, args.size, d, ang, K, (w, h))
            if img is None:
                row += f"{'-':>9s}"; continue
            corners, ids, _ = detector.detectMarkers(img)
            if ids is None or len(ids) == 0:
                row += f"{'X':>9s}"
                continue
            # OpenCV 4.7+ 에서 estimatePoseSingleMarkers 가 제거됐다.
            # 실기 코드(sensors/aruco_detector.py)와 동일하게 solvePnP 를 쓴다.
            half = args.size / 2.0
            objp = np.array([[-half, half, 0], [half, half, 0],
                             [half, -half, 0], [-half, -half, 0]], np.float64)
            ok, rvec, _tvec = cv2.solvePnP(
                objp, corners[0].reshape(4, 2).astype(np.float64), K, dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                row += f"{'O(?)':>9s}"; continue
            R, _ = cv2.Rodrigues(rvec)
            yaw = abs(math.degrees(math.atan2(R[0, 2], R[2, 2])))
            row += f"{'O(' + f'{abs(yaw-ang):.0f}' + ')':>9s}"
        print(row)
    print()
    print("  O(n) = 검출 성공, n = yaw 추정 오차(도).  X = 검출 실패")


if __name__ == "__main__":
    main()

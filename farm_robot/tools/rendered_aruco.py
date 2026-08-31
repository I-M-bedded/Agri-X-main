# -*- coding: utf-8 -*-
"""
tools/rendered_aruco.py
------------------------
2D 시뮬의 마커 검출을 **실제 OpenCV ArUco 검출기**로 대체한다.

왜
  기존 SimAruco 는 "면각 70도 하드컷" 하나로 검출 여부를 정했다. 그 모델은
  해상도·원근·모션블러·포즈 모호성을 전혀 모른다. 실제로 이 때문에 결론이
  한 번 뒤집혔다(팻말 yaw 기반 밭 방위 유도가 시뮬에선 완벽, 실기에선 불가).

무엇을 하는가
  매 틱 로봇 자세에서 보이는 각 팻말을 **원근 투영으로 렌더링**하고,
  cv2.aruco.detectMarkers 를 그대로 돌려 검출 여부를 판정한다.
  즉 "몇 도까지 보이는가"를 가정하지 않고 **실제로 검출해 본다.**

  SimAruco 와 같은 인터페이스(detect())라 FSM 은 차이를 모른다.

한계 (Gazebo 가 필요한 이유는 남는다)
  조명/그림자/이랑에 의한 가림/렌즈 왜곡/실제 모션블러는 여전히 없다.
  여기서 얻는 검출률은 **낙관적 상한**이다.
"""

import math

import cv2
import numpy as np

from config import (
    ARUCO_DICTIONARY,
    CAMERA_RESOLUTION,
    MARKER_CONFIRM_FRAMES,
    MARKER_SIZE_M,
    SIGN_MARKER_LATERAL,
)
from sensors.aruco_detector import MarkerObservation

_MARKER_PX = 400
_HFOV_DEG = 62.0


def _camera_matrix(width, height, hfov_deg=_HFOV_DEG):
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0, width / 2.0],
                     [0, fx, height / 2.0],
                     [0, 0, 1.0]], np.float64)


class RenderedAruco:
    """SimAruco 와 같은 계약. 다만 검출을 **실제로** 해본다.

    world  : SimWorld (마커 위치/자세와 로봇 자세를 제공)
    blur_px: 주행 중 모션블러 근사(0이면 없음)
    """

    def __init__(self, world, blur_px: int = 0, marker_size_m: float = MARKER_SIZE_M):
        self.world = world
        self.blur_px = int(blur_px)
        self.size = marker_size_m
        self._streak = {}
        w, h = CAMERA_RESOLUTION
        self._wh = (w, h)
        self._K = _camera_matrix(w, h)
        self._dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, ARUCO_DICTIONARY))
        self._detector = cv2.aruco.ArucoDetector(
            self._dict, cv2.aruco.DetectorParameters())
        self._tiles = {}          # id -> 마커 이미지(여백 포함)
        # 통계: 기하학적으로 화각 안에 있었는데 실제로는 검출 실패한 횟수
        self.stats = {"in_fov": 0, "detected": 0}

    # ------------------------------------------------------------------
    def _tile(self, mid):
        if mid not in self._tiles:
            img = cv2.aruco.generateImageMarker(self._dict, int(mid), _MARKER_PX)
            q = _MARKER_PX // 8            # 여백(quiet zone) 없으면 검출률 급락
            canvas = np.full((_MARKER_PX + 2 * q, _MARKER_PX + 2 * q), 255, np.uint8)
            canvas[q:q + _MARKER_PX, q:q + _MARKER_PX] = img
            self._tiles[mid] = canvas
        return self._tiles[mid]

    def _render(self, canvas, mid, dist_m, view_angle_rad, bearing_right_rad):
        """마커 하나를 카메라 평면에 원근 투영해 canvas 에 그린다."""
        half = self.size / 2.0
        a = view_angle_rad
        # 마커 중심의 카메라 좌표
        cx = dist_m * math.sin(bearing_right_rad)
        cz = dist_m * math.cos(bearing_right_rad)
        pts = []
        for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1)):     # TL,TR,BR,BL
            X = cx + sx * half * math.cos(a)
            Z = cz - sx * half * math.sin(a)
            Y = sy * half
            if Z <= 0.05:
                return False
            u = self._K[0, 0] * X / Z + self._K[0, 2]
            v = self._K[1, 1] * (-Y) / Z + self._K[1, 2]
            pts.append([u, v])
        dst = np.array(pts, np.float32)
        # 화면 밖이면 그리지 않는다
        w, h = self._wh
        if dst[:, 0].max() < 0 or dst[:, 0].min() > w:
            return False
        if dst[:, 1].max() < 0 or dst[:, 1].min() > h:
            return False
        tile = self._tile(mid)
        n = tile.shape[0]
        src = np.array([[0, 0], [n - 1, 0], [n - 1, n - 1], [0, n - 1]], np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        cv2.warpPerspective(tile, M, (w, h), dst=canvas,
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_TRANSPARENT)
        return True

    # ------------------------------------------------------------------
    def render_frame(self):
        """지금 자세에서 카메라가 볼 장면(마커만)을 렌더링해 반환."""
        w, h = self._wh
        canvas = np.full((h, w), 235, np.uint8)      # 밝은 배경(흙 대용)
        present = {}
        for mid, (mx, my, facing) in self.world.markers.items():
            dx, dy = mx - self.world.x, my - self.world.y
            dist = math.hypot(dx, dy)
            if dist < 0.05 or dist > 12.0:
                continue
            bearing_ccw = math.atan2(dy, dx) - self.world.theta
            bearing_ccw = math.atan2(math.sin(bearing_ccw), math.cos(bearing_ccw))
            bearing_right = -bearing_ccw
            # 마커 법선과 '마커->로봇' 방향 사이 각 = 비스듬히 보는 정도
            to_robot = math.atan2(-dy, -dx)
            view = math.atan2(math.sin(to_robot - facing), math.cos(to_robot - facing))
            if abs(view) >= math.pi / 2:
                continue                      # 뒷면
            if self._render(canvas, mid, dist, view, bearing_right):
                present[mid] = (dist, bearing_right)
        if self.blur_px > 0:
            k = self.blur_px * 2 + 1
            canvas = cv2.GaussianBlur(canvas, (k, k), 0)
        return canvas, present

    def detect(self):
        """SimAruco.detect() 와 같은 계약. 실제 검출기를 돌린다."""
        frame, present = self.render_frame()
        self.stats["in_fov"] += len(present)

        corners, ids, _ = self._detector.detectMarkers(frame)
        raw = {}
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                mid = int(i)
                if mid not in present:
                    continue          # 렌더하지 않은 것이 잡히면 오검출 - 버린다
                dist, bearing_right = present[mid]
                raw[mid] = MarkerObservation(
                    marker_id=mid,
                    distance_m=dist,
                    forward_m=dist * math.cos(bearing_right),
                    lateral_offset_m=dist * math.sin(bearing_right) * SIGN_MARKER_LATERAL,
                    # 단독 마커 yaw 는 신뢰할 수 없다(실측). 0 으로 둔다.
                    yaw_error_rad=0.0,
                )
        self.stats["detected"] += len(raw)

        for mid in list(self._streak):
            if mid not in raw:
                del self._streak[mid]
        for mid in raw:
            self._streak[mid] = self._streak.get(mid, 0) + 1
        return {m: o for m, o in raw.items()
                if self._streak[m] >= MARKER_CONFIRM_FRAMES}

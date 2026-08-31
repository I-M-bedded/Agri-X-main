# -*- coding: utf-8 -*-
"""
sensors/aruco_detector.py
--------------------------
전방 카메라로 ArUco 마커를 검출하고 거리/횡오프셋/각도를 추정한다.

이전 버전 대비 수정 사항
  1) cv2.aruco.estimatePoseSingleMarkers() 제거.
     이 함수는 OpenCV 4.7 에서 deprecated 되어 최신 버전에서는 사라진다.
     반면 cv2.aruco.ArucoDetector 는 4.7 이상에서만 존재하므로,
     예전 코드는 두 API 가 동시에 존재하는 아주 좁은 버전 구간에서만 동작했다.
     -> solvePnP(SOLVEPNP_IPPE_SQUARE) 로 교체하고, 구/신 API 를 모두 지원.
  2) COLOR_RGB2GRAY -> COLOR_BGR2GRAY (Camera 는 BGR 을 반환한다).
  3) 마커의 yaw 를 진입각으로 쓰지 않는다. 작은 평면 마커의 단독 자세추정은
     포즈 모호성(pose ambiguity) 때문에 부호가 튀는 것으로 악명이 높다.
     팻말은 고랑당 1개뿐이라 두 마커의 상대 위치로 각도를 구할 수도 없다.
     -> compute_post_bearing() 은 **방위와 거리만** 반환하고(yaw 는 참고용),
        진입각은 _field_heading(밭 안쪽 방위)에서 얻는다.
        중심선은 비전이 고랑 자체를 보고 직접 찾는다.
  4) 오검출 방지를 위한 다중 프레임 확인(MARKER_CONFIRM_FRAMES).
  5) cv2 가 없어도 import 는 성공하고 빈 결과를 반환한다(PC 구조 점검용).
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from config import (
    ARUCO_DICTIONARY,
    CAMERA_MATRIX,
    CAMERA_RESOLUTION,
    DIST_COEFFS,
    MARKER_CONFIRM_FRAMES,
    MARKER_SIZE_M,
    SIGN_MARKER_LATERAL,
    SIGN_MARKER_YAW,
)
from logutil import get_logger
from sensors.camera import Camera

log = get_logger("aruco")

try:
    import cv2
    import numpy as np

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


@dataclass
class MarkerObservation:
    marker_id: int
    distance_m: float        # 카메라~마커 직선 거리
    forward_m: float         # 카메라 광축(z) 방향 거리 - 진입 판정에 사용
    lateral_offset_m: float  # 좌우 오프셋 (음수=좌, 양수=우)
    yaw_error_rad: float     # 마커 단독 yaw 추정 (참고용, 노이즈 큼)


@dataclass
class EntranceAlignment:
    lateral_error: float     # m, 양수 = 입구 중심이 오른쪽 -> 오른쪽으로 가야 함
    heading_error: float     # rad, 양수 = 오른쪽으로 돌아야 함
    forward_distance: float  # m, 게이트까지의 전방 거리
    valid: bool              # 기하 검증 통과 여부


class ArucoDetector:
    def __init__(self, camera: Camera):
        self._camera = camera
        self._detector = None
        self._legacy = False
        self._seen_streak: Dict[int, int] = {}

        if not _HAS_CV2:
            log.warning("OpenCV 없음 - ArUco 검출이 비활성화됩니다.")
            self._camera_matrix = None
            self._dist_coeffs = None
            return

        dict_id = getattr(cv2.aruco, ARUCO_DICTIONARY)
        if hasattr(cv2.aruco, "ArucoDetector"):
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            # 야외 역광/그림자에 강하도록 서브픽셀 코너 보정 사용
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

            # [실측 튜닝] Gazebo 렌더 프레임(640x480, 10cm 마커, 거리 2.1m,
            #   화면상 약 25px)에서 **기본 파라미터로는 검출 0건**이었다.
            #   2배 확대하면 검출되므로 원인은 해상도이고, 아래 세 값을
            #   낮추면 원본 해상도에서도 검출된다.
            #     minMarkerPerimeterRate      작은 마커를 후보에서 버리지 않게
            #     perspectiveRemovePixelPerCell 셀당 샘플 수를 줄여 저해상도 대응
            #     polygonalApproxAccuracyRate  코너 근사 허용오차 완화
            #   ※ 너무 낮추면 오검출이 늘어난다. 실기 영상으로 재확인할 것.
            params.minMarkerPerimeterRate = 0.01
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 15
            params.adaptiveThreshWinSizeStep = 2
            params.perspectiveRemovePixelPerCell = 8
            params.polygonalApproxAccuracyRate = 0.05
            self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        else:  # OpenCV < 4.7
            self._legacy = True
            self._aruco_dict = cv2.aruco.Dictionary_get(dict_id)
            self._params = cv2.aruco.DetectorParameters_create()

        self._camera_matrix = CAMERA_MATRIX
        self._dist_coeffs = DIST_COEFFS
        if self._camera_matrix is None:
            w, h = CAMERA_RESOLUTION
            focal = float(w)  # 매우 거친 근사 - 반드시 캘리브레이션으로 교체할 것
            self._camera_matrix = np.array(
                [[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64
            )
            self._dist_coeffs = np.zeros((5, 1), dtype=np.float64)
            log.warning(
                "CAMERA_MATRIX 가 None 입니다. 근사값을 사용하므로 거리/각도 추정이 "
                "부정확합니다. cv2.calibrateCamera 로 캘리브레이션 후 config 에 넣으세요."
            )

        # 마커 한 변 길이 기준 3D 코너 좌표 (마커 중심이 원점, 반시계 순서)
        half = MARKER_SIZE_M / 2.0
        self._object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    def _detect_markers(self, gray):
        if self._legacy:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._aruco_dict, parameters=self._params
            )
        else:
            corners, ids, _ = self._detector.detectMarkers(gray)
        return corners, ids

    def detect(self) -> Dict[int, MarkerObservation]:
        """현재 프레임에서 검출된 마커를 {id: MarkerObservation} 로 반환."""
        if not _HAS_CV2:
            return {}

        frame = self._camera.capture_frame()
        if frame is None:
            return {}

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        corners, ids = self._detect_markers(gray)
        if ids is None or len(ids) == 0:
            self._decay_streaks(set())
            return {}

        raw: Dict[int, MarkerObservation] = {}
        for i, marker_id in enumerate(ids.flatten()):
            image_points = corners[i].reshape(4, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(
                self._object_points,
                image_points,
                self._camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue

            t = tvec.reshape(3)
            rot_mat, _ = cv2.Rodrigues(rvec)
            yaw = float(math.atan2(rot_mat[0, 2], rot_mat[2, 2]))

            raw[int(marker_id)] = MarkerObservation(
                marker_id=int(marker_id),
                distance_m=float(np.linalg.norm(t)),
                forward_m=float(t[2]),
                lateral_offset_m=float(t[0]) * SIGN_MARKER_LATERAL,
                yaw_error_rad=yaw * SIGN_MARKER_YAW,
            )

        # 다중 프레임 확인: 연속 N프레임 이상 본 마커만 인정 (오검출 방지)
        self._decay_streaks(set(raw.keys()))
        confirmed = {
            mid: obs
            for mid, obs in raw.items()
            if self._seen_streak.get(mid, 0) >= MARKER_CONFIRM_FRAMES
        }
        return confirmed

    def _decay_streaks(self, present):
        for mid in list(self._seen_streak.keys()):
            if mid not in present:
                del self._seen_streak[mid]
        for mid in present:
            self._seen_streak[mid] = min(
                MARKER_CONFIRM_FRAMES + 5, self._seen_streak.get(mid, 0) + 1
            )

    def find_pair(
        self, left_id: int, right_id: int
    ) -> Tuple[Optional[MarkerObservation], Optional[MarkerObservation]]:
        observed = self.detect()
        return observed.get(left_id), observed.get(right_id)


# ======================================================================
def compute_post_bearing(obs: "MarkerObservation") -> "EntranceAlignment":
    """
    팻말 마커까지의 **방위와 거리**만 반환한다.

    [중요] 이 함수는 '고랑 중심이 어디인가'를 알려주지 않는다.
      마커는 "여기 고랑 입구가 있다"만 말해줄 뿐, 중심선이 마커에서
      왼쪽 30cm 인지 오른쪽 50cm 인지는 마커 안에 담긴 정보가 아니다.
      그걸 상수로 미리 넣어두면 현장에서 팻말을 조금만 옮겨 박아도
      로봇이 이랑으로 돌진한다. 그런 사전 지식은 쓰지 않는다.

      중심선은 **비전이 고랑 자체를 보고** 찾는다.
      마커의 역할은 여기까지다.
        - 이 고랑이 몇 번인지 (ID)
        - 그 입구가 어느 방향에 있는지 (방위)
        - 얼마나 남았는지 (거리)

    heading_error 도 계산하지 않는다(항상 0). 마커 하나의 자세(yaw)는
    포즈 모호성 때문에 프레임마다 수십 도씩 튀기 때문이다.
    진입각은 _field_heading(밭 안쪽 방위)에서 얻는다.
    """
    return EntranceAlignment(
        lateral_error=obs.lateral_offset_m,   # 마커 자체의 좌우 위치(중심 아님)
        heading_error=0.0,
        forward_distance=obs.forward_m,
        valid=obs.forward_m > 0.05,
    )

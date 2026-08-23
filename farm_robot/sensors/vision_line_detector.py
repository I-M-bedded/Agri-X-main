# -*- coding: utf-8 -*-
"""
sensors/vision_line_detector.py
--------------------------------
전방 카메라 영상만으로 고랑(흙 trough)의 중앙선을 추정한다.
고전적 컴퓨터비전만 사용하므로 라즈베리파이4에서 실시간 처리가 가능하다.

이전 버전의 구조적 결함
  ROI 전체에서 "흙 픽셀의 가로 무게중심"을 중앙선으로 썼는데,
  고랑 안에서는 **바닥도 흙이고 양옆 이랑도 흙**이라 화면 대부분이 마스크에
  잡힌다. 그러면 무게중심은 로봇 위치와 무관하게 항상 화면 중앙 근처가 되어
  오차가 0 으로 수렴한다. 게다가 신뢰도를 "흙 픽셀 비율"로 정의해서,
  신뢰도가 높을수록 오히려 추정이 무의미해지는 역상관 구조였다.
  여기에 COLOR_RGB2HSV 오용까지 겹쳐(카메라는 BGR 출력) 흙(H≈10)이
  파랑(H≈110)으로 뒤집혀 신뢰도가 사실상 항상 0 이었다.

이번 버전
  1) COLOR_BGR2HSV 로 수정.
  2) ROI 를 위/아래 여러 밴드로 나눠 밴드별 무게중심을 구하고 직선 피팅한다.
     -> 횡오프셋(offset)뿐 아니라 고랑선의 기울기(heading)까지 얻는다.
        로봇 가까운 쪽(아래 밴드)이 현재 위치, 먼 쪽(위 밴드)이 진행 방향.
  3) 신뢰도를 "적당한 커버리지에서 최대"가 되는 창(window) 형태로 계산하고,
     밴드별 중심이 들쭉날쭉하면(일관성 낮음) 감점한다.
     -> 화면 전체가 흙일 때(=정보 없음) 신뢰도가 낮아진다.
  4) 모폴로지 연산으로 잡초/돌 노이즈 제거.

주의: 이 방식도 야외 조명에 민감하다. SOIL_HSV_LOWER/UPPER 는 반드시
현장 영상으로 재캘리브레이션할 것 (tools/setup.py 3번).
"""

from dataclasses import dataclass
from typing import Optional

from config import (
    SIGN_VISION_ERROR,
    SOIL_HSV_LOWER,
    SOIL_HSV_UPPER,
    VISION_BAND_CONSISTENCY_TOL,
    VISION_COVERAGE_IDEAL_HI,
    VISION_COVERAGE_IDEAL_LO,
    VISION_COVERAGE_MAX,
    VISION_COVERAGE_MIN,
    VISION_NUM_BANDS,
    VISION_ROI_Y_END_RATIO,
    VISION_ROI_Y_START_RATIO,
)
from logutil import get_logger
from sensors.camera import Camera

log = get_logger("vision")

try:
    import cv2
    import numpy as np

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


@dataclass
class VisionLineResult:
    normalized_error: float  # -1(왼쪽 치우침) ~ +1(오른쪽 치우침). 양수 = 오른쪽으로 가야 함
    heading_error: float     # -1 ~ +1, 고랑선의 기울기. 양수 = 고랑이 오른쪽으로 휨
    confidence: float        # 0.0 ~ 1.0
    coverage: float          # ROI 중 흙으로 판정된 비율 (디버깅용)


def _coverage_score(coverage: float) -> float:
    """
    커버리지 -> 신뢰도 계수(0~1).
    너무 적으면 흙을 못 찾은 것, 너무 많으면 화면 전체가 흙이라 중심이 무의미.
    이상 구간에서 1.0, 양끝으로 갈수록 선형으로 0 이 된다.
    """
    if coverage <= VISION_COVERAGE_MIN or coverage >= VISION_COVERAGE_MAX:
        return 0.0
    if VISION_COVERAGE_IDEAL_LO <= coverage <= VISION_COVERAGE_IDEAL_HI:
        return 1.0
    if coverage < VISION_COVERAGE_IDEAL_LO:
        span = VISION_COVERAGE_IDEAL_LO - VISION_COVERAGE_MIN
        return (coverage - VISION_COVERAGE_MIN) / span if span > 0 else 0.0
    span = VISION_COVERAGE_MAX - VISION_COVERAGE_IDEAL_HI
    return (VISION_COVERAGE_MAX - coverage) / span if span > 0 else 0.0


class VisionLineDetector:
    def __init__(self, camera: Camera):
        self._camera = camera
        self._kernel = None
        if _HAS_CV2:
            self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # ------------------------------------------------------------------
    def compute(self) -> Optional[VisionLineResult]:
        """
        현재 프레임을 분석해 중앙선 오차를 계산한다.
        None 은 **카메라에서 프레임을 못 받은 경우에만** 반환한다.
        고랑이 잘 안 보이는 경우에는 confidence 가 낮은 결과를 반환해서,
        상위 로직이 "신뢰도가 낮다"는 사실을 명시적으로 알 수 있게 한다.
        """
        if not _HAS_CV2:
            return None

        frame = self._camera.capture_frame()
        if frame is None:
            return None

        return self.compute_from_frame(frame)

    def compute_from_frame(self, frame) -> VisionLineResult:
        """프레임을 직접 받아 처리 (테스트/오프라인 튜닝에서 재사용)."""
        height, width = frame.shape[:2]
        y_start = int(height * VISION_ROI_Y_START_RATIO)
        y_end = int(height * VISION_ROI_Y_END_RATIO)
        roi = frame[y_start:y_end, :]

        if roi.size == 0:
            return VisionLineResult(0.0, 0.0, 0.0, 0.0)

        # Camera 는 BGR 을 반환한다 (config.CAMERA_OUTPUT_IS_BGR 참고)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(SOIL_HSV_LOWER), np.array(SOIL_HSV_UPPER))

        # 잡초/돌 같은 점 노이즈 제거 + 구멍 메우기
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        total = mask.shape[0] * mask.shape[1]
        soil_count = int(cv2.countNonZero(mask))
        coverage = soil_count / total if total > 0 else 0.0

        if soil_count == 0:
            return VisionLineResult(0.0, 0.0, 0.0, coverage)

        # --- 밴드별 무게중심 ---
        band_h = max(1, mask.shape[0] // VISION_NUM_BANDS)
        centers = []   # (밴드 중심 y 비율, 정규화된 x 오차)
        image_center_x = width / 2.0
        half_width = width / 2.0

        for b in range(VISION_NUM_BANDS):
            y0 = b * band_h
            y1 = mask.shape[0] if b == VISION_NUM_BANDS - 1 else (b + 1) * band_h
            band = mask[y0:y1, :]
            if band.size == 0:
                continue
            count = int(cv2.countNonZero(band))
            # 밴드 안에 최소한의 픽셀이 있어야 중심이 의미가 있다
            if count < band.size * 0.02:
                continue
            col_sum = band.sum(axis=0, dtype=np.float64)
            xs = np.arange(band.shape[1], dtype=np.float64)
            centroid_x = float((col_sum * xs).sum() / col_sum.sum())
            norm_x = (centroid_x - image_center_x) / half_width
            # y 비율: 0 = ROI 맨 위(먼 곳), 1 = 맨 아래(로봇 가까운 곳)
            y_ratio = (y0 + y1) / 2.0 / mask.shape[0]
            centers.append((y_ratio, max(-1.0, min(1.0, norm_x))))

        if not centers:
            return VisionLineResult(0.0, 0.0, 0.0, coverage)

        ys = np.array([c[0] for c in centers])
        xs_n = np.array([c[1] for c in centers])

        # 로봇 바로 앞(y_ratio=1.0)에서의 횡오프셋을 직선 피팅으로 외삽
        if len(centers) >= 2 and ys.std() > 1e-6:
            slope, intercept = np.polyfit(ys, xs_n, 1)
            offset_at_robot = slope * 1.0 + intercept
            # 기울기가 양수 = 아래로 갈수록 고랑이 오른쪽 = 로봇이 왼쪽에 있음
            # heading 은 "위쪽(먼 곳)이 어디로 향하는가"이므로 부호를 뒤집는다
            heading = -slope
            residual = float(np.std(xs_n - (slope * ys + intercept)))
        else:
            offset_at_robot = float(xs_n.mean())
            heading = 0.0
            residual = 0.0

        offset_at_robot = max(-1.0, min(1.0, float(offset_at_robot)))
        heading = max(-1.0, min(1.0, float(heading)))

        # --- 신뢰도 ---
        cov_score = _coverage_score(coverage)
        # 밴드 개수가 적으면 감점
        band_score = len(centers) / float(VISION_NUM_BANDS)
        # 밴드 중심이 직선에서 많이 벗어나면(일관성 없음) 감점
        consistency = max(
            0.0, 1.0 - (residual / VISION_BAND_CONSISTENCY_TOL)
        ) if VISION_BAND_CONSISTENCY_TOL > 0 else 1.0

        confidence = cov_score * band_score * consistency
        confidence = max(0.0, min(1.0, confidence))

        return VisionLineResult(
            normalized_error=offset_at_robot * SIGN_VISION_ERROR,
            heading_error=heading * SIGN_VISION_ERROR,
            confidence=confidence,
            coverage=coverage,
        )

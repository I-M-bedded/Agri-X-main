# -*- coding: utf-8 -*-
"""
tools/vision_error_model.py
----------------------------
시뮬레이터의 '완벽한 비전'을 **실측 오차 분포**로 오염시키는 모델.

왜 이게 필요한가
  기존 SimWorld.vision_result() 는 로봇 참값 자세에서 계산한 오라클이라
  confidence 가 항상 0.8 이고 heading_error 가 항상 0.0 이다. 즉 시뮬은
  "비전이 틀렸을 때" 를 한 번도 겪어보지 않는다. 그런데 실측(CRDLD 430장)
  에서는 **conf 0.7 인데 각도 오차 31도** 인 프레임이 실재한다.
  이 모델은 그 실측 분포를 그대로 재생해 폐루프가 견디는지 본다.

왜 합성 영상 렌더러가 아닌가
  렌더러가 만든 깨끗한 이랑 영상에 우리 모델을 돌리면 당연히 잘 맞는다
  (그림자·잡초·역광이 없으므로). 그건 순환 논증이라 위 31도 실패를
  재현하지 못한다. 여기서는 **이미 측정된 실패를 그대로 주입**한다.

단위 변환 (2단계)
  (1) 화면 정규화: perception/ccrdnet/postprocess.py 의 정의상
        normalized_error = (x_near - W/2) / (W/2),  W = 256
      이므로 평가 CSV 의 lateral_near/far [px @256] 를 128 로 나누면
      **화면 기준** 무차원 오차가 된다. 여기까진 가정이 없다.
  (2) 고랑 정규화: 그런데 시뮬레이터의 vision_result() 는 오차를
      **고랑 반폭**으로 나눈다(norm = lateral / FURROW_HALF_WIDTH_M).
      두 정규화 기준이 다르므로 변환 계수가 필요하다:
        furrow_scale = (근접 행에서 화면 반폭이 덮는 지면 거리) / (고랑 반폭)
      ★ 이 값은 카메라 내부/장착 파라미터가 있어야 정확히 계산되는데
        config.CAMERA_MATRIX 가 아직 None 이고 틸트각도 없다. 따라서
        **가정값**이며, 실험에서 반드시 민감도를 함께 스윕해야 한다.
      중심 추정 2.5 근거: 카메라 높이 0.45m/HFOV 62도에서 근접 행의 화면
      반폭이 지면 약 0.50m 를 덮는다고 보면 0.50 / 0.20 = 2.5.

모델링 가정 (측정값이 아님 - 반드시 구분할 것)
  1) CSV 는 오차의 **절대값**만 담고 있어 부호를 알 수 없다.
     -> 샘플마다 부호 s = ±1 를 뽑고, near/far 에 **같은 부호**를 준다
        (하나의 잘못된 직선은 한쪽으로 기울어져 있으므로).
  2) furrow_scale(위 참조)은 카메라 캘리브레이션 전까지 가정값이다.
  3) CRDLD 테스트셋은 연속 영상이 아니라 여러 밭/조건에서 뽑은 표본이라
     오차의 **시간 상관을 추출할 수 없다**. 실제 주행에서는 그림자 구간을
     지나는 동안 연속으로 틀릴 가능성이 크므로, burst_len 파라미터로
     "연속 N프레임 같은 오차 유지" 를 강제 주입할 수 있게 했다.
     burst_len=1 이 CSV 그대로(독립), >1 은 보수적 가정이다.
"""

import csv
import os
import random

from sensors.vision_line_detector import VisionLineResult

# 평가 해상도(256) 기준 정규화 상수. postprocess 의 half_width 와 동일.
_HALF_WIDTH_PX = 128.0

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "reports", "crdld_furrow_v1", "metrics_frames.csv",
)


def load_samples(csv_path=DEFAULT_CSV):
    """평가 CSV -> [(err_delta_abs, heading_delta_abs, confidence), ...]"""
    samples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["pred_ok"] != "True":
                # 라인 자체를 못 뽑은 프레임 = confidence 0 으로 취급
                samples.append((0.0, 0.0, 0.0))
                continue
            near = float(row["lateral_near"]) / _HALF_WIDTH_PX
            far = float(row["lateral_far"]) / _HALF_WIDTH_PX
            samples.append((near, far - near, float(row["confidence"])))
    if not samples:
        raise ValueError(f"샘플이 비어 있습니다: {csv_path}")
    return samples


class ReplayVision:
    """SimVision 을 감싸서 실측 오차/지연/신뢰도를 입히는 어댑터.

    compute() 계약은 VisionLineDetector 와 동일하므로 LineFollower 는
    이것이 시뮬인지 실기인지 알지 못한다.
    """

    def __init__(
        self,
        world,
        samples=None,
        seed: int = 0,
        burst_len: int = 1,
        furrow_scale: float = 2.5,     # 화면 정규화 -> 고랑 정규화 (가정, 위 참조)
        update_every_ticks: int = 4,   # 20Hz 제어 / 5fps 추론 = 4틱마다 갱신
        force_blind: bool = False,
    ):
        self._world = world
        self._samples = samples if samples is not None else load_samples()
        self._rng = random.Random(seed)
        self._burst_len = max(1, int(burst_len))
        self._furrow_scale = float(furrow_scale)
        self._update_every = max(1, int(update_every_ticks))
        self.force_blind = force_blind

        self._tick = 0
        self._held = None          # 마지막으로 발행한 결과 (지연 재현)
        self._cur = None           # 현재 적용 중인 오차 샘플
        self._burst_left = 0
        # 계측용 통계
        self.stats = {"frames": 0, "gated_out": 0, "accepted": 0,
                      "accepted_bad": 0, "max_accepted_err": 0.0}

    # ------------------------------------------------------------------
    def _next_sample(self):
        """burst_len 동안 같은 오차를 유지하고, 그 뒤 새 샘플을 뽑는다."""
        if self._burst_left <= 0 or self._cur is None:
            near, dhead, conf = self._rng.choice(self._samples)
            sign = self._rng.choice((-1.0, 1.0))
            k = self._furrow_scale
            self._cur = (sign * near * k, sign * dhead * k, conf)
            self._burst_left = self._burst_len
        self._burst_left -= 1
        return self._cur

    def compute(self):
        self._tick += 1
        if self.force_blind:
            return VisionLineResult(0.0, 0.0, 0.0, 0.0)

        # 추론은 5fps 이므로 그 사이 틱은 **묵은 값**을 그대로 돌려준다.
        if self._held is not None and (self._tick % self._update_every) != 0:
            return self._held

        truth = self._world.vision_result()
        # 고랑이 화면에 없으면(참값 신뢰도 0) 오차를 입힐 대상이 없다.
        if truth.confidence <= 0.0:
            self._held = truth
            return truth

        err_d, head_d, conf = self._next_sample()
        result = VisionLineResult(
            normalized_error=max(-1.5, min(1.5, truth.normalized_error + err_d)),
            heading_error=max(-1.5, min(1.5, truth.heading_error + head_d)),
            confidence=conf,
            coverage=truth.coverage,
        )

        # --- 계측: 게이트를 통과한 오답이 얼마나 되는가 ---
        from config import VISION_MIN_CONFIDENCE

        self.stats["frames"] += 1
        if conf < VISION_MIN_CONFIDENCE:
            self.stats["gated_out"] += 1
        else:
            self.stats["accepted"] += 1
            if abs(err_d) > 0.10:      # 무차원 0.10 ≈ 화면폭의 5%
                self.stats["accepted_bad"] += 1
            self.stats["max_accepted_err"] = max(
                self.stats["max_accepted_err"], abs(err_d)
            )

        self._held = result
        return result

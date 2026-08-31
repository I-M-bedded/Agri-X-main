# -*- coding: utf-8 -*-
"""
control/line_follower.py
-------------------------
고랑 안에서 "가상의 중앙선"을 따라가는 주행 제어.

역할 분담
  1) 카메라 비전 = 주 제어(primary). 입구에서 ArUco 로 진입 방향을 잡은 뒤,
     고랑 안에서는 마커가 안 보이므로 흙 영역 추적으로 그 선을 계속 따라간다.
  2) 1D ToF = 보조(secondary). 조향에 상시 관여하지 않고
     (a) 좌우 중앙 여부 교차검증, (b) 고랑 끝 감지 두 가지만 담당.
  3) 엔코더 Odometry = 비전 폴백. 비전 신뢰도가 낮으면 직전 헤딩을 유지.

이전 버전 대비 수정 사항 (매우 중요)
  1) **조향 부호 수정.** 예전에는
        left = base - steer,  right = base + steer
     이었는데, error = right_mm - left_mm > 0 (=오른쪽에 공간이 많다=로봇이
     왼쪽에 치우침) 일 때 steer > 0 -> 좌회전이 되어 **오차가 커질수록 더
     벌어지는 양의 피드백**이었다. 비전 오차도 같은 문제를 갖고 있었다.
     이제 규약(양수=오른쪽)에 맞춰 MotorDriver.drive() 로 일원화한다.
  2) **오차 정규화.** 예전에는 mm 단위 오차에 Kp=0.8, 출력 클램프 0.35 라서
     0.44mm 만 넘어도 포화되는 bang-bang 제어였다. 이제 무차원으로 정규화.
  3) **헤딩 폴백 부호 수정.** Odometry theta 는 CCW 양수인데 주석이 반대로
     적혀 있어 폴백 보정이 반대 방향으로 걸리고 있었다(직진 대신 나선 주행).
  4) **ToF 이중 읽기 제거.** 예전에는 read() 후 both_out_of_range() 가 또
     측정을 해서 틱당 2회 샘플링 -> EMA 시정수가 의도의 절반이 되었다.
  5) tof_centered 를 실제로 사용한다(비전-ToF 교차검증 불일치 경고).
"""

import time
from dataclasses import dataclass
from typing import Optional

from config import (
    BASE_SPEED,
    CROSS_CHECK_DISAGREE_WARN_SEC,
    HEADING_HOLD_GAIN,
    LINE_PID_D_FILTER_HZ,
    LINE_PID_INTEGRAL_LIMIT,
    LINE_PID_KD,
    LINE_PID_KI,
    LINE_PID_KP,
    MAX_STEER_CORRECTION,
    SIGN_HEADING_ERROR,
    SIGN_TOF_ERROR,
    TOF_CENTERED_THRESHOLD_MM,
    TOF_NOMINAL_WALL_DISTANCE_MM,
    USE_VISION_LINE_FOLLOWING,
    VISION_HEADING_WEIGHT,
    VISION_MIN_CONFIDENCE,
    VISION_TOF_DISAGREE_LIMIT,
    VISION_TRUST_FULL_CONFIDENCE,
)
from control.pid_controller import PIDController
from config import (
    FURROW_END_REQUIRE_VISION_AGREE,
    FURROW_END_VISION_CONFIDENCE_MAX,
    TOF_ASSIST_WEIGHT,
)
from logutil import get_logger
from sensors.odometry import normalize_angle
from sensors.tof_sensor import ToFPair
from sensors.vision_line_detector import VisionLineDetector

log = get_logger("line")


@dataclass
class LineFollowResult:
    steer: float                # 부호 규약: 양수 = 오른쪽으로 조향
    base_speed: float
    error: float                # 이번 틱에 실제로 쓰인 무차원 오차
    furrow_end_detected: bool
    using_vision: bool
    tof_centered: bool
    vision_confidence: float
    left_mm: float
    right_mm: float
    cross_check_ok: bool        # 비전과 ToF 판단이 서로 모순되지 않는가
    vision_weight: float = 0.0  # 이번 틱에 비전에 준 가중치 (0=순수 ToF)
    vision_vetoed: bool = False # ToF 와 크게 어긋나 비전을 폐기했는가




def vision_trust_weight(confidence: float) -> float:
    """비전 신뢰도 -> 비전 가중치(0.0 ~ 1-TOF_ASSIST_WEIGHT).

    게이트(VISION_MIN_CONFIDENCE)에서 0 -> 순수 ToF 와 같아지고,
    VISION_TRUST_FULL_CONFIDENCE 이상에서 최대가 된다.
    "비전은 신뢰도가 높을 때만 주도권을 갖는다" 는 정책의 구현부.
    """
    span = VISION_TRUST_FULL_CONFIDENCE - VISION_MIN_CONFIDENCE
    ratio = 1.0 if span <= 0 else (confidence - VISION_MIN_CONFIDENCE) / span
    ratio = max(0.0, min(1.0, ratio))
    return ratio * (1.0 - TOF_ASSIST_WEIGHT)


class LineFollower:
    def __init__(
        self,
        tof_pair: ToFPair,
        vision_detector: Optional[VisionLineDetector] = None,
        odometry=None,
        base_speed: float = BASE_SPEED,
    ):
        self._tof = tof_pair
        self._vision = vision_detector if USE_VISION_LINE_FOLLOWING else None
        self.odom = odometry
        self._pid = PIDController(
            LINE_PID_KP,
            LINE_PID_KI,
            LINE_PID_KD,
            output_limit=MAX_STEER_CORRECTION,
            integral_limit=LINE_PID_INTEGRAL_LIMIT,
            d_filter_hz=LINE_PID_D_FILTER_HZ,
        )
        self.base_speed = base_speed
        self._target_heading = 0.0
        self._disagree_since = None

    # ------------------------------------------------------------------
    def reset(self):
        self._pid.reset()
        self._tof.reset_end_detection()
        self._disagree_since = None
        if self.odom is not None:
            self._target_heading = self.odom.theta

    # ------------------------------------------------------------------
    def step(self) -> LineFollowResult:
        """
        한 제어 주기 실행.
        주의: Odometry.update() 는 상위(FSM)에서 매 틱 1회 호출한다.
              여기서 또 호출하면 이중 적분은 아니지만 책임이 흩어지므로 하지 않는다.
        """
        # --- ToF: 이 틱의 유일한 샘플링 ---
        left_mm, right_mm = self._tof.read()
        furrow_end = self._tof.both_out_of_range()

        # ToF 오차 정규화: 벽까지의 공칭 거리로 나눠 무차원화
        tof_diff_mm = (right_mm - left_mm) * SIGN_TOF_ERROR
        tof_error = tof_diff_mm / TOF_NOMINAL_WALL_DISTANCE_MM
        tof_error = max(-1.5, min(1.5, tof_error))

        # 한쪽이라도 벽이 안 보이면 ToF 오차는 신뢰할 수 없다.
        # (EMA 값이 아니라 raw 기준으로 판정 - tof_sensor.sample() 주석 참고)
        tof_valid = self._tof.walls_visible()
        tof_centered = tof_valid and abs(tof_diff_mm) <= TOF_CENTERED_THRESHOLD_MM

        # --- 비전: 주 제어 ---
        using_vision = False
        vision_confidence = 0.0
        vision_error = 0.0

        if self._vision is not None:
            vr = self._vision.compute()
            if vr is not None:
                vision_confidence = vr.confidence
                if vision_confidence >= VISION_MIN_CONFIDENCE:
                    vision_error = (
                        vr.normalized_error
                        + VISION_HEADING_WEIGHT * vr.heading_error
                    )
                    using_vision = True

        # --- 오차 선택: 비전 > ToF > 헤딩유지 ---
        # [수정] 마커가 입구에만 있으므로 고랑 안에서는 인공 표식이 없다.
        #   중심선 추종은 전적으로 비전과 ToF 가 담당한다.
        # [수정] 예전에는 게이트만 넘으면 신뢰도와 무관하게 비전에 75% 를 고정
        #   배분했다. 그래서 신뢰도 0.26 짜리 오답도 조향을 지배했고, ToF 25%
        #   로는 지워지지 않았다(실측 오차 주입 시 완주율 6.7%).
        #   이제 (a) ToF 와 크게 어긋나면 비전을 폐기하고,
        #        (b) 살아남아도 신뢰도에 비례한 가중치만 준다.
        vision_weight = 0.0
        vision_vetoed = False

        # (a) 거부권: 물리량을 직접 재는 ToF 가 비전 오답을 걸러낸다.
        #     비전이 옆 고랑을 잡으면 오차가 통째로 어긋나므로 여기서 걸린다.
        if (
            using_vision
            and tof_valid
            and VISION_TOF_DISAGREE_LIMIT > 0
            and abs(vision_error - tof_error) > VISION_TOF_DISAGREE_LIMIT
        ):
            using_vision = False
            vision_vetoed = True

        if using_vision:
            if tof_valid:
                # (b) 신뢰도 비례 배분. 게이트 턱걸이 = 사실상 순수 ToF.
                vision_weight = vision_trust_weight(vision_confidence)
                error = (
                    vision_weight * vision_error
                    + (1.0 - vision_weight) * tof_error
                )
            else:
                # ToF 를 못 쓰면 비전이 유일한 근거다.
                vision_weight = 1.0
                error = vision_error
            # [중요] 헤딩 기준각은 **비전이 실제로 주도권을 가졌을 때만** 갱신한다.
            #   예전에는 게이트만 넘으면 매 틱 갱신했다. 그래서 비전이 로봇을
            #   틀린 방향으로 몰고 가는 동안에도 그 헤딩을 "옳다"고 저장했고,
            #   나중에 비전이 끊겨 폴백할 때 이미 틀어진 각을 목표로 유지했다.
            if self.odom is not None and vision_weight >= 0.5:
                self._target_heading = self.odom.theta
        else:
            # 폴백: ToF(유효할 때) + 엔코더 헤딩 유지
            error = tof_error if tof_valid else 0.0
            if self.odom is not None:
                # theta 는 CCW 양수. theta > target 이면 왼쪽으로 틀어진 것이므로
                # 오른쪽으로 가야 한다 -> 양수 오차. 규약 일치.
                heading_error = normalize_angle(self.odom.theta - self._target_heading)
                error += SIGN_HEADING_ERROR * HEADING_HOLD_GAIN * heading_error

        error = max(-1.5, min(1.5, error))
        steer = self._pid.compute(error)

        # --- 교차검증: 비전은 "중앙"이라는데 ToF 는 계속 치우쳤다면 경고 ---
        cross_check_ok = True
        if using_vision and tof_valid:
            vision_says_centered = abs(vision_error) < 0.15
            if vision_says_centered != tof_centered:
                now = time.monotonic()
                if self._disagree_since is None:
                    self._disagree_since = now
                elif now - self._disagree_since > CROSS_CHECK_DISAGREE_WARN_SEC:
                    cross_check_ok = False
                    log.warning(
                        "비전과 ToF 판단이 %.0f초 이상 불일치합니다 "
                        "(vision_err=%.2f, tof_diff=%.0fmm). "
                        "흙 HSV 임계값 또는 ToF 장착 각도를 점검하세요.",
                        CROSS_CHECK_DISAGREE_WARN_SEC, vision_error, tof_diff_mm,
                    )
                    self._disagree_since = now
            else:
                self._disagree_since = None

        # --- 고랑 끝 판정 ---
        # [수정] 출구에 마커가 없으므로 스스로 알아내야 한다.
        #   주 근거는 ToF: 좌우 이랑 벽이 둘 다 사라짐(연속 확인 포함).
        #   FURROW_END_REQUIRE_VISION_AGREE=True 면 비전 신뢰도 급락까지
        #   함께 요구해서, 고랑 중간에 흙이 유실된 구간에서의 오판을 막는다.
        if furrow_end and FURROW_END_REQUIRE_VISION_AGREE:
            vision_lost = vision_confidence <= FURROW_END_VISION_CONFIDENCE_MAX
            if not vision_lost:
                # 벽은 사라졌는데 고랑 구조는 여전히 뚜렷하다.
                # -> 진짜 끝이 아니라 벽이 잠깐 끊긴 구간일 가능성이 크다.
                furrow_end = False

        return LineFollowResult(
            steer=steer,
            base_speed=self.base_speed,
            error=error,
            furrow_end_detected=furrow_end,
            using_vision=using_vision,
            tof_centered=tof_centered,
            vision_confidence=vision_confidence,
            left_mm=left_mm,
            right_mm=right_mm,
            cross_check_ok=cross_check_ok,
            vision_weight=vision_weight,
            vision_vetoed=vision_vetoed,
        )

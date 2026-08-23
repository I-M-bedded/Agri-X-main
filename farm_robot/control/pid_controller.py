# -*- coding: utf-8 -*-
"""
control/pid_controller.py
--------------------------
범용 PID 컨트롤러.

이전 버전 대비 수정 사항
  1) 적분 와인드업 방지 (integral_limit 클램프 + 출력 포화 시 적분 정지)
  2) 미분항 1차 저역통과 필터 (엔코더/비전 노이즈가 D항에서 폭발하는 것 방지)
  3) dt 안전 범위 클램프 (루프 지연이나 일시정지로 dt가 튀는 것 방지)
  4) 입력 오차는 **무차원(정규화)** 값을 받는 것을 전제로 한다.
     예전에는 mm 단위 오차에 Kp=0.8을 곱해서 0.44mm만 넘어도 출력이 포화되는
     사실상 bang-bang 제어였다. 호출부에서 반드시 정규화해서 넣을 것.
"""

import time

from config import PID_DT_MAX, PID_DT_MIN


class PIDController:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limit: float = 1.0,
        integral_limit: float = 1.0,
        d_filter_hz: float = 0.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = abs(output_limit)
        self.integral_limit = abs(integral_limit)
        self.d_filter_hz = d_filter_hz

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
        self._d_filtered = 0.0
        self._has_prev = False

        # 디버깅용 마지막 항별 기여도
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0
        self.saturated = False

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
        self._d_filtered = 0.0
        self._has_prev = False
        self.last_p = self.last_i = self.last_d = 0.0
        self.saturated = False

    def compute(self, error: float, dt: float = None) -> float:
        """
        error: 무차원 오차. 부호 규약상 **양수 = 오른쪽으로 가야 함**.
        반환:  -output_limit ~ +output_limit 로 클램프된 조향 출력.
               부호 규약상 **양수 = 오른쪽(시계방향)으로 조향**.
        """
        now = time.monotonic()

        if dt is None:
            dt = (now - self._prev_time) if self._prev_time is not None else 0.0
        self._prev_time = now

        # dt 안전 클램프: 첫 호출(dt=0)이나 비정상적으로 큰 dt를 걸러낸다.
        if dt <= 0.0:
            dt = PID_DT_MIN
        dt = max(PID_DT_MIN, min(PID_DT_MAX, dt))

        # --- P ---
        p_term = self.kp * error

        # --- D (첫 호출에서는 미분 킥을 방지하기 위해 0) ---
        if self._has_prev:
            raw_d = (error - self._prev_error) / dt
        else:
            raw_d = 0.0
            self._has_prev = True

        if self.d_filter_hz > 0.0:
            # 1차 저역통과: alpha = dt / (tau + dt), tau = 1/(2*pi*fc)
            tau = 1.0 / (2.0 * 3.141592653589793 * self.d_filter_hz)
            alpha = dt / (tau + dt)
            self._d_filtered += alpha * (raw_d - self._d_filtered)
        else:
            self._d_filtered = raw_d
        d_term = self.kd * self._d_filtered

        self._prev_error = error

        # --- I (조건부 적분: 출력이 포화된 방향으로는 더 쌓지 않는다) ---
        tentative = p_term + self.ki * self._integral + d_term
        will_saturate_high = tentative >= self.output_limit and error > 0
        will_saturate_low = tentative <= -self.output_limit and error < 0

        if self.ki != 0.0 and not (will_saturate_high or will_saturate_low):
            self._integral += error * dt
            self._integral = max(
                -self.integral_limit, min(self.integral_limit, self._integral)
            )
        i_term = self.ki * self._integral

        output = p_term + i_term + d_term
        clamped = max(-self.output_limit, min(self.output_limit, output))

        self.last_p, self.last_i, self.last_d = p_term, i_term, d_term
        self.saturated = clamped != output
        return clamped

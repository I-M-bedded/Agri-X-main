# -*- coding: utf-8 -*-
"""
actuators/motor_driver.py
--------------------------
좌/우 바퀴 차동구동(differential drive) 모터 제어.

부호 규약 (config.py 참조)
  steer > 0  =  오른쪽(시계방향)으로 조향
  믹싱       :  left = base + steer,  right = base - steer

이전 버전 대비 수정 사항
  1) 데드밴드 보상 추가. DC 기어모터는 듀티가 일정값 이하이면 아예 안 돈다.
     정렬 단계의 미세 조향(예: 0.05)이 무시되면 수렴하지 못하고
     MAX_APPROACH_DURATION_SEC 초과 -> SAFE_HALT 로 빠졌다.
  2) turn_180_blocking 의 무한루프 위험 제거.
     예전에는 |theta - target| < 0.05 라는 "창"으로 판정했는데,
     엔코더 틱 1개당 각도 변화가 약 0.061 rad (틱 20/회전, 축간 0.17m) 이라
     창보다 양자화 스텝이 더 커서 목표창을 건너뛰면 영원히 회전했다.
     -> "부호 교차(crossing)" 판정 + 절대 타임아웃 + 엔코더 stall 감지로 교체.
  3) 임의 각도 회전(turn_by_angle_blocking) 일반화 - 헤드랜드 90도 선회에 사용.
  4) 모터 극성 뒤집기용 SIGN_LEFT_MOTOR / SIGN_RIGHT_MOTOR 지원.
"""

import math
import time

from config import (
    GPIO_WARNINGS,
    MOTOR_DEADBAND_EPS,
    MOTOR_MIN_DUTY,
    MOTOR_PINS,
    PWM_FREQUENCY_HZ,
    SIGN_LEFT_MOTOR,
    SIGN_RIGHT_MOTOR,
    TURN_180_DURATION_SEC,
    TURN_ENCODER_STALL_SEC,
    TURN_SPEED,
    TURN_TIMEOUT_MARGIN,
)
from logutil import get_logger

log = get_logger("motor")

try:
    import RPi.GPIO as GPIO

    _HAS_GPIO = True
except ImportError:
    _HAS_GPIO = False


class MotorDriver:
    def __init__(self, odometry=None):
        self._pins = MOTOR_PINS
        self.odom = odometry
        self.last_left = 0.0
        self.last_right = 0.0
        self._gpio_ready = False

        if _HAS_GPIO:
            try:
                GPIO.setwarnings(GPIO_WARNINGS)
                GPIO.setmode(GPIO.BCM)
                # [수정] initial=LOW 를 명시한다. 명시하지 않으면 setup 직후
                # 핀 상태가 보장되지 않아 전원 인가/재실행 순간 모터가
                # 순간적으로 튈 수 있다.
                for pin in self._pins.values():
                    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
                self._left_pwm = GPIO.PWM(self._pins["left_pwm"], PWM_FREQUENCY_HZ)
                self._right_pwm = GPIO.PWM(self._pins["right_pwm"], PWM_FREQUENCY_HZ)
                self._left_pwm.start(0)
                self._right_pwm.start(0)
                self._gpio_ready = True
            except Exception as exc:
                log.error("모터 GPIO 초기화 실패: %s", exc)
                self._left_pwm = self._right_pwm = None
        else:
            self._left_pwm = None
            self._right_pwm = None
            log.info("RPi.GPIO 없음 - 모터 명령은 기록만 되고 실제 출력은 없습니다.")

    # ------------------------------------------------------------------
    @staticmethod
    def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _apply_deadband(speed: float) -> float:
        """
        [-1,1] 속도 명령을 실제 듀티로 변환.
        |speed| 가 아주 작으면 0(정지), 그 외에는 MOTOR_MIN_DUTY ~ 1.0 구간으로 매핑.
        모터가 "명령은 갔는데 안 도는" 구간을 없앤다.
        """
        magnitude = abs(speed)
        if magnitude < MOTOR_DEADBAND_EPS:
            return 0.0
        span = 1.0 - MOTOR_MIN_DUTY
        return MOTOR_MIN_DUTY + span * min(1.0, magnitude)

    def _drive_wheel(self, side: str, speed: float):
        speed = self._clamp(speed)
        sign = SIGN_LEFT_MOTOR if side == "left" else SIGN_RIGHT_MOTOR
        effective = speed * sign

        # 엔코더는 단일 채널이라 방향을 모른다 -> 지금 지시한 방향을 알려준다.
        if self.odom is not None:
            direction = 0 if abs(speed) < MOTOR_DEADBAND_EPS else (1 if effective >= 0 else -1)
            # 정지 중에도 관성으로 틱이 들어올 수 있어 방향은 유지한다.
            if direction != 0:
                if side == "left":
                    self.odom.left_dir = direction
                else:
                    self.odom.right_dir = direction

        if side == "left":
            self.last_left = speed
        else:
            self.last_right = speed

        if not self._gpio_ready:
            return

        duty = self._apply_deadband(effective) * 100.0
        in1 = self._pins[f"{side}_in1"]
        in2 = self._pins[f"{side}_in2"]
        pwm = self._left_pwm if side == "left" else self._right_pwm

        if duty == 0.0:
            # 브레이크(양쪽 LOW = 코스트). 미세 명령에서 덜덜거리지 않게.
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.LOW)
        elif effective >= 0:
            GPIO.output(in1, GPIO.HIGH)
            GPIO.output(in2, GPIO.LOW)
        else:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.HIGH)
        pwm.ChangeDutyCycle(duty)

    # ------------------------------------------------------------------
    def set_speeds(self, left_speed: float, right_speed: float):
        """좌/우 바퀴 속도 동시 설정 (-1.0 ~ 1.0)."""
        self._drive_wheel("left", left_speed)
        self._drive_wheel("right", right_speed)

    def drive(self, base_speed: float, steer: float):
        """
        부호 규약을 한 곳에서만 구현하기 위한 헬퍼.
        steer > 0 -> 오른쪽으로 조향 (좌륜이 빠름).
        상위 모듈은 반드시 이 함수를 통해 조향할 것.

        [수정] 예전에는 좌/우를 각각 독립적으로 [-1,1] 로 클램프했다.
        base=0.9, steer=0.35 이면 left=1.25 -> 1.0 으로 잘리는데 right=0.55 는
        그대로라서 **의도한 조향량(0.35)이 0.225 로 줄어든다**. 즉 빨리 달릴수록
        회전이 약해지는 비선형이 생겨 PID 튜닝이 어긋난다.
        이제 넘치는 만큼 양쪽을 함께 낮춰서 좌우 차이(=회전율)를 보존한다.
        """
        left = base_speed + steer
        right = base_speed - steer

        peak = max(abs(left), abs(right))
        if peak > 1.0:
            excess = peak - 1.0
            # 조향량은 유지하고 기준 속도만 낮춘다
            shift = math.copysign(excess, base_speed if base_speed != 0 else 1.0)
            left -= shift
            right -= shift

        self.set_speeds(left, right)

    def forward(self, speed: float):
        self.set_speeds(speed, speed)

    def stop(self):
        self.set_speeds(0.0, 0.0)

    def rotate_in_place(self, clockwise: bool = True, speed: float = TURN_SPEED):
        """
        제자리 회전.
        clockwise=True  -> 오른쪽(시계방향) 회전 -> theta 감소 (CCW 양수 규약)
        """
        if clockwise:
            self.set_speeds(speed, -speed)
        else:
            self.set_speeds(-speed, speed)

    # ------------------------------------------------------------------
    def turn_by_angle_blocking(self, delta_rad: float, speed: float = TURN_SPEED) -> bool:
        """
        제자리에서 delta_rad 만큼 회전한다 (블로킹).
          delta_rad > 0 : 반시계(좌회전),  < 0 : 시계(우회전)
        반환: 엔코더 기준으로 목표 각도에 도달했으면 True,
              타임아웃/엔코더 고장으로 중단했으면 False.

        엔코더가 없거나 죽어 있으면 시간 기반으로 근사 회전한다.
        어떤 경우에도 **무한 루프에 빠지지 않는다**.
        """
        if abs(delta_rad) < 1e-6:
            return True

        # 예상 소요 시간 -> 절대 타임아웃 산정
        nominal = TURN_180_DURATION_SEC * (abs(delta_rad) / math.pi)
        timeout = max(0.5, nominal * TURN_TIMEOUT_MARGIN)
        clockwise = delta_rad < 0

        if self.odom is None:
            self.rotate_in_place(clockwise=clockwise, speed=speed)
            time.sleep(nominal)
            self.stop()
            return True

        start_theta = self.odom.theta
        target = start_theta + delta_rad

        start_time = time.monotonic()
        last_tick_count = self.odom.total_ticks
        last_tick_time = start_time

        self.rotate_in_place(clockwise=clockwise, speed=speed)

        reached = False
        while True:
            self.odom.update()
            now = time.monotonic()

            # 창(window)이 아니라 "부호 교차"로 판정 -> 양자화로 건너뛰어도 안전
            if delta_rad > 0:
                if self.odom.theta >= target:
                    reached = True
                    break
            else:
                if self.odom.theta <= target:
                    reached = True
                    break

            if now - start_time > timeout:
                log.warning(
                    "회전 타임아웃(%.1fs). 목표 %.2f rad, 실제 %.2f rad 에서 정지합니다.",
                    timeout, delta_rad, self.odom.theta - start_theta,
                )
                break

            # 엔코더 stall 감지: 틱이 전혀 안 들어오면 엔코더 고장으로 간주
            if self.odom.total_ticks != last_tick_count:
                last_tick_count = self.odom.total_ticks
                last_tick_time = now
            elif now - last_tick_time > TURN_ENCODER_STALL_SEC:
                log.warning(
                    "회전 중 엔코더 틱이 %.1f초간 없습니다. 시간 기반으로 전환합니다.",
                    TURN_ENCODER_STALL_SEC,
                )
                remaining = max(0.0, nominal - (now - start_time))
                time.sleep(remaining)
                break

            time.sleep(0.005)

        self.stop()
        return reached

    def turn_180_blocking(self, speed: float = TURN_SPEED) -> bool:
        """고랑 끝 유턴. 시계방향(-pi)으로 제자리 180도."""
        return self.turn_by_angle_blocking(-math.pi, speed=speed)

    # ------------------------------------------------------------------
    def cleanup(self):
        self.stop()
        if self._gpio_ready:
            try:
                self._left_pwm.stop()
                self._right_pwm.stop()
                GPIO.cleanup(list(self._pins.values()))
            except Exception:
                pass
            self._gpio_ready = False

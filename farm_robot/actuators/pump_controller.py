# -*- coding: utf-8 -*-
"""
actuators/pump_controller.py
-----------------------------
펌프는 전압이 인가되는 순간 바로 작동하므로,
"고랑 밖에서는 절대 켜지지 않도록" 소프트웨어 인터록을 강제한다.

이전 버전 대비 수정 사항
  1) tick() 추가 - 매 제어 주기마다 릴레이 상태를 다시 확정(re-assert)한다.
     외부 노이즈나 코드 경로 누락으로 상태가 어긋나는 것을 막는 방어 코드.
  2) 최대 연속 가동 시간 워치독(PUMP_MAX_CONTINUOUS_SEC) 추가.
     상위 로직이 어떤 이유로든 OFF를 못 부른 경우의 마지막 방어선.
  3) 물 부족(water_low)일 때는 zone 과 무관하게 펌프를 잠근다(공회전 방지).
"""

import time

from config import (
    GPIO_WARNINGS,
    PUMP_MAX_CONTINUOUS_SEC,
    PUMP_RELAY_ACTIVE_HIGH,
    PUMP_RELAY_PIN,
)
from logutil import get_logger

log = get_logger("pump")

try:
    import RPi.GPIO as GPIO

    _HAS_GPIO = True
except ImportError:
    _HAS_GPIO = False


class PumpController:
    def __init__(self):
        self._in_furrow = False        # 안전 게이트: 지금 고랑 안인가
        self._requested_on = False     # 상위 로직이 켜기를 원하는가
        self._locked_out = False       # 물 부족 등으로 잠금
        self._relay_state = False
        self._on_since = None
        self._gpio_ready = False

        if _HAS_GPIO:
            try:
                GPIO.setwarnings(GPIO_WARNINGS)
                GPIO.setmode(GPIO.BCM)
                # [수정/중요] initial 을 명시하지 않으면 setup 직후 핀이 LOW 가 된다.
                # active-low 릴레이 모듈(PUMP_RELAY_ACTIVE_HIGH=False)에서는
                # LOW = 펌프 ON 이므로, 프로그램이 뜨는 순간 물이 쏟아진다.
                # 반드시 'OFF 에 해당하는 레벨'로 시작해야 한다.
                off_level = GPIO.LOW if PUMP_RELAY_ACTIVE_HIGH else GPIO.HIGH
                GPIO.setup(PUMP_RELAY_PIN, GPIO.OUT, initial=off_level)
                self._gpio_ready = True
            except Exception as exc:
                log.error("펌프 GPIO 초기화 실패: %s", exc)
        self._apply_relay(False)

    # ------------------------------------------------------------------
    def _apply_relay(self, on: bool):
        self._relay_state = bool(on)
        if on:
            if self._on_since is None:
                self._on_since = time.monotonic()
        else:
            self._on_since = None

        if not self._gpio_ready:
            return
        level = GPIO.HIGH if (on == PUMP_RELAY_ACTIVE_HIGH) else GPIO.LOW
        GPIO.output(PUMP_RELAY_PIN, level)

    def _desired_state(self) -> bool:
        return self._requested_on and self._in_furrow and not self._locked_out

    # ------------------------------------------------------------------
    def set_zone(self, in_furrow: bool):
        """상위 내비게이션이 '지금 고랑 안/밖'을 알려줄 때 호출."""
        changed = self._in_furrow != bool(in_furrow)
        self._in_furrow = bool(in_furrow)
        if not self._in_furrow:
            # 고랑을 벗어나는 순간 요청 자체를 취소하고 즉시 OFF
            self._requested_on = False
        if changed:
            log.debug("펌프 zone -> %s", "고랑 안" if in_furrow else "고랑 밖")
        self._apply_relay(self._desired_state())

    def set_lockout(self, locked: bool):
        """물 부족 등으로 펌프를 강제로 잠근다(공회전 방지)."""
        if self._locked_out != bool(locked):
            log.info("펌프 잠금 %s", "설정" if locked else "해제")
        self._locked_out = bool(locked)
        self._apply_relay(self._desired_state())

    def turn_on(self) -> bool:
        """펌프 ON 요청. 반환값 = 실제로 켜졌는지 (고랑 밖/잠금이면 False)."""
        self._requested_on = True
        desired = self._desired_state()
        self._apply_relay(desired)
        if not desired:
            log.debug("펌프 ON 요청이 인터록에 의해 차단되었습니다.")
        return desired

    def turn_off(self):
        """OFF는 구역과 무관하게 항상 즉시 반영."""
        self._requested_on = False
        self._apply_relay(False)

    def tick(self):
        """
        매 제어 주기마다 호출. 릴레이 상태를 다시 확정하고
        최대 연속 가동 시간을 감시한다.
        """
        desired = self._desired_state()

        if (
            desired
            and self._on_since is not None
            and PUMP_MAX_CONTINUOUS_SEC > 0
            and time.monotonic() - self._on_since > PUMP_MAX_CONTINUOUS_SEC
        ):
            log.warning(
                "펌프가 %.0f초 이상 연속 가동되어 워치독이 강제로 껐습니다.",
                PUMP_MAX_CONTINUOUS_SEC,
            )
            self._requested_on = False
            self._apply_relay(False)
            return

        if desired != self._relay_state:
            self._apply_relay(desired)

    def is_on(self) -> bool:
        return self._relay_state

    def cleanup(self):
        self.turn_off()
        if self._gpio_ready:
            try:
                GPIO.cleanup(PUMP_RELAY_PIN)
            except Exception:
                pass
            self._gpio_ready = False

# -*- coding: utf-8 -*-
"""
sensors/water_tank_sensor.py
-----------------------------
물통 수위를 적외선(IR) 센서로 감지. 디지털 임계값 방식.

이전 버전의 문제
  is_water_low() 안에 디바운스 타이머가 들어 있는데, 이 함수가
  고랑 1개당 딱 1번(_state_evaluate_mission)만 호출되고 있었다.
  디바운스 특성상 "첫 호출은 항상 False" 이므로, 물이 실제로 떨어져도
  다음 고랑 하나를 통째로 빈 펌프로 돌고 나서야 감지되는 구조였다.

이번 버전
  poll() 을 매 제어 틱마다 호출해서 디바운스를 정상 동작시키고,
  is_water_low() 는 확정된 상태만 반환하는 순수 조회 함수로 분리했다.
  복구 쪽에도 디바운스를 걸어 채터링으로 상태가 왔다갔다 하는 것을 막는다.
"""

import time

from config import (
    GPIO_WARNINGS,
    WATER_LEVEL_DEBOUNCE_SEC,
    WATER_LEVEL_SENSOR_PIN,
    WATER_LOW_SIGNAL_ACTIVE_HIGH,
    WATER_SENSOR_PULL,
)
from logutil import get_logger

log = get_logger("water")

try:
    import RPi.GPIO as GPIO

    _HAS_GPIO = True
except ImportError:
    _HAS_GPIO = False


class WaterTankSensor:
    def __init__(self, pin: int = WATER_LEVEL_SENSOR_PIN):
        self.pin = pin
        self._raw_low = False
        self._confirmed_low = False
        self._changed_since = None
        self._gpio_ready = False
        self._sim_low = False  # 시뮬레이션/테스트용 강제 값

        if _HAS_GPIO:
            try:
                GPIO.setwarnings(GPIO_WARNINGS)
                GPIO.setmode(GPIO.BCM)
                # [수정] 내부 풀 저항을 명시한다. 배선이 빠지거나 오픈컬렉터
                # 출력이면 핀이 뜬 상태(floating)가 되어 값이 랜덤으로 읽히고,
                # 그 결과 물이 가득 차 있는데도 '물 부족'으로 판정되어
                # 고랑 하나만 돌고 HOME 으로 복귀해 버린다.
                pull = {
                    "up": GPIO.PUD_UP,
                    "down": GPIO.PUD_DOWN,
                }.get(str(WATER_SENSOR_PULL).lower(), GPIO.PUD_OFF)
                GPIO.setup(self.pin, GPIO.IN, pull_up_down=pull)
                self._gpio_ready = True
            except Exception as exc:
                log.error("수위 센서 GPIO 초기화 실패: %s", exc)

    # ------------------------------------------------------------------
    def _read_raw(self) -> bool:
        """센서 원시 신호를 '물 부족(True)' / '정상(False)' 으로 변환."""
        if self._gpio_ready:
            level = GPIO.input(self.pin)
            return bool(level) if WATER_LOW_SIGNAL_ACTIVE_HIGH else not bool(level)
        return self._sim_low

    def poll(self):
        """**매 제어 틱마다** 호출. 디바운스를 진행한다."""
        raw = self._read_raw()
        now = time.monotonic()

        if raw != self._raw_low:
            self._raw_low = raw
            self._changed_since = now
            return

        if raw == self._confirmed_low:
            self._changed_since = None
            return

        if self._changed_since is None:
            self._changed_since = now
            return

        if now - self._changed_since >= WATER_LEVEL_DEBOUNCE_SEC:
            self._confirmed_low = raw
            self._changed_since = None
            log.info("수위 상태 변경 -> %s", "물 부족" if raw else "정상")

    def is_water_low(self) -> bool:
        """디바운스로 확정된 상태만 반환 (측정하지 않음)."""
        return self._confirmed_low

    def cleanup(self):
        if self._gpio_ready:
            try:
                GPIO.cleanup(self.pin)
            except Exception:
                pass
            self._gpio_ready = False

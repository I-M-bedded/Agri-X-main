# -*- coding: utf-8 -*-
"""
sensors/imu.py
---------------
MPU-6050 급 I2C IMU 에서 요레이트(z축 각속도)를 읽는다.

용도: 회전/헤딩을 엔코더 대신 자이로로 재서 궤도 미끄러짐과 무관하게 만든다
(README "알려진 한계" 1·4번의 근본 해결책). 위치(거리)는 여전히 엔코더 몫이고,
IMU 는 **theta 만** 담당한다 — ImuFusedOdometry(odometry.py) 참고.

설계 원칙
  - 정지 상태에서 바이어스를 실측해 빼준다 (자이로는 가만히 있어도 드리프트).
  - 하드웨어/라이브러리가 없으면 available=False 로 조용히 물러난다.
    상위(odometry)는 이때 엔코더 경로로 자동 폴백한다.
  - 다른 IMU 로 바꾸려면 read_yaw_rate() 만 같은 계약으로 구현하면 된다.
    (반환: rad/s, 반시계(CCW) 양수 — 프로젝트 theta 규약과 동일)
"""

import time

from config import IMU_CALIBRATION_SEC, IMU_I2C_ADDRESS, IMU_I2C_BUS, SIGN_IMU_YAW
from logutil import get_logger

log = get_logger("imu")

# MPU-6050 레지스터
_PWR_MGMT_1 = 0x6B
_GYRO_CONFIG = 0x1B
_CONFIG_DLPF = 0x1A
_GYRO_ZOUT_H = 0x47
_GYRO_SCALE_250DPS = 131.0    # LSB per (deg/s), FS_SEL=0


class MPU6050Yaw:
    def __init__(self, bus_id: int = IMU_I2C_BUS, address: int = IMU_I2C_ADDRESS):
        self.available = False
        self.bias_rad_s = 0.0
        self._bus = None
        self._addr = address
        try:
            from smbus2 import SMBus

            self._bus = SMBus(bus_id)
            self._bus.write_byte_data(address, _PWR_MGMT_1, 0x00)   # 슬립 해제
            self._bus.write_byte_data(address, _GYRO_CONFIG, 0x00)  # ±250 dps
            self._bus.write_byte_data(address, _CONFIG_DLPF, 0x03)  # DLPF 44Hz
            time.sleep(0.05)
            self.available = True
            log.info("MPU-6050 준비 (bus=%d addr=0x%02X)", bus_id, address)
        except Exception as exc:
            log.warning("IMU 초기화 실패 - 엔코더 헤딩으로 폴백합니다: %s", exc)

    # ------------------------------------------------------------------
    def _read_raw_z(self) -> int:
        high = self._bus.read_byte_data(self._addr, _GYRO_ZOUT_H)
        low = self._bus.read_byte_data(self._addr, _GYRO_ZOUT_H + 1)
        value = (high << 8) | low
        return value - 65536 if value > 32767 else value

    def read_yaw_rate(self) -> float:
        """요레이트 [rad/s], CCW 양수. 실패 시 0.0 (호출측이 폴백 판단)."""
        if not self.available:
            return 0.0
        try:
            import math

            dps = self._read_raw_z() / _GYRO_SCALE_250DPS
            return SIGN_IMU_YAW * math.radians(dps) - self.bias_rad_s
        except Exception as exc:
            log.warning("IMU 읽기 실패 - 비활성화: %s", exc)
            self.available = False
            return 0.0

    # ------------------------------------------------------------------
    def calibrate_bias(self, duration_sec: float = IMU_CALIBRATION_SEC):
        """**로봇이 완전히 정지한 상태에서** 호출. 자이로 바이어스를 실측한다."""
        if not self.available:
            return
        samples = []
        t_end = time.monotonic() + max(0.2, duration_sec)
        self.bias_rad_s = 0.0            # 측정 동안 보정 없이 원시값 수집
        while time.monotonic() < t_end:
            samples.append(self.read_yaw_rate())
            time.sleep(0.005)
        if samples and self.available:
            self.bias_rad_s = sum(samples) / len(samples)
            log.info("IMU 바이어스 보정: %.5f rad/s (%d샘플)",
                     self.bias_rad_s, len(samples))

    def cleanup(self):
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
        self.available = False

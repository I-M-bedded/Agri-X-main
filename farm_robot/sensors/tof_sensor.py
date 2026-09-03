# -*- coding: utf-8 -*-
"""
sensors/tof_sensor.py
----------------------
좌/우 1D ToF 거리 센서 (VL53L1X 기준).

이전 버전의 가장 큰 문제
  _read_raw_mm() 이 항상 TOF_OUT_OF_RANGE_MM 을 반환하는 껍데기였다.
  실기에서도 좌우가 항상 out-of-range -> both_out_of_range() 항상 True ->
  고랑 진입 2초 뒤 즉시 "고랑 끝" 오판 -> 유턴 루프.
  또한 VL53L1X 를 2개 쓰려면 XSHUT 로 하나씩 켜면서 I2C 주소를 재할당해야
  하는데(둘 다 공장 기본 주소 0x29) 그 시퀀스가 통째로 빠져 있었다.

이번 버전
  1) 실제 드라이버 백엔드 3종 지원: adafruit / pimoroni / sim
  2) ToFPair 가 XSHUT 주소 재할당 시퀀스를 책임진다.
  3) read() 를 한 번 호출하면 그 틱의 값이 캐시된다.
     예전에는 line_follower 가 read() 후 both_out_of_range() 를 또 불러서
     센서당 2회 측정이 일어났고, EMA 가 두 번 갱신되어 필터 시정수가
     의도의 절반이 되고 있었다.
  4) 고랑 끝 판정을 연속 N틱 확인으로 바꿔 노이즈 1회에 오판하지 않는다.
"""

import time

from config import (
    TOF_ALLOW_SIM_FALLBACK,
    TOF_BACKEND,
    TOF_DEFAULT_I2C_ADDRESS,
    TOF_EMA_ALPHA,
    TOF_END_CONFIRM_TICKS,
    TOF_INIT_SETTLE_SEC,
    TOF_INVALID_MM,
    TOF_NOMINAL_WALL_DISTANCE_MM,
    TOF_OUT_OF_RANGE_MM,
    TOF_STALE_TIMEOUT_SEC,
    TOF_TIMING_BUDGET_MS,
)
from logutil import get_logger

log = get_logger("tof")

try:
    import RPi.GPIO as GPIO

    _HAS_GPIO = True
except ImportError:
    _HAS_GPIO = False


class ToFSensor:
    """단일 1D ToF 센서 하나를 감싸는 클래스."""

    def __init__(self, name: str, i2c_address: int, xshut_pin: int):
        self.name = name
        self.i2c_address = i2c_address
        self.xshut_pin = xshut_pin

        self._driver = None
        self._ema_value = None
        self._last_mm = TOF_INVALID_MM
        self.last_raw_mm = TOF_INVALID_MM
        self.fail_count = 0
        self.ok = False
        self.is_real = False           # 실제 하드웨어 드라이버가 붙었는가

        # [신규] 마지막으로 '성공한' 측정값과 그 시각.
        # 측정이 준비되지 않은 틱에서 out-of-range 로 오판하지 않기 위해 사용.
        self._last_valid_mm = None
        self._last_valid_time = 0.0

    # ------------------------------------------------------------------
    def attach_driver(self, driver):
        self._driver = driver
        self.ok = driver is not None
        self.is_real = bool(driver is not None and getattr(driver, "is_real", False))

    def _read_raw_mm(self) -> float:
        """
        센서로부터 단일 측정값(mm)을 읽는다.

        [수정/중요] 반환 규칙이 세 가지로 나뉜다.
          1) 새 측정값이 있으면 그 값
          2) 아직 측정이 끝나지 않았으면(None) **직전 유효값을 유지**한다.
             VL53L1X 는 timing_budget(50ms) 마다 값이 갱신되는데 제어 루프도
             50ms 라서 "아직 준비 안 됨"이 매우 흔하다. 이걸 실패로 처리하면
             TOF_INVALID_MM(=out-of-range) 이 되어 벽이 사라진 것처럼 보이고
             가짜 '고랑 끝' 판정 -> 즉시 유턴/SAFE_HALT 로 이어진다.
          3) TOF_STALE_TIMEOUT_SEC 넘게 새 값이 없으면 그때는 진짜 고장이므로
             TOF_INVALID_MM 을 반환한다.
        """
        if self._driver is None:
            self.fail_count += 1
            return TOF_INVALID_MM

        now = time.monotonic()
        try:
            value = self._driver.read_mm()
        except Exception as exc:
            self.fail_count += 1
            if self.fail_count in (1, 10, 100):
                log.warning("%s ToF 읽기 실패(%d회): %s", self.name, self.fail_count, exc)
            value = None

        if value is not None and value > 0:
            self.fail_count = 0
            self._last_valid_mm = float(value)
            self._last_valid_time = now
            return float(value)

        # 새 값이 없다 -> 유효 기간 안이면 직전 값을 유지 (실패로 세지 않는다)
        if (
            self._last_valid_mm is not None
            and (now - self._last_valid_time) <= TOF_STALE_TIMEOUT_SEC
        ):
            return self._last_valid_mm

        self.fail_count += 1
        if self.fail_count in (1, 20, 100):
            log.warning(
                "%s ToF 가 %.1f초 이상 새 측정값을 내지 못했습니다(%d회). "
                "I2C 배선/주소/XSHUT 를 점검하세요.",
                self.name, TOF_STALE_TIMEOUT_SEC, self.fail_count,
            )
        return TOF_INVALID_MM

    def sample(self) -> float:
        """
        센서를 실제로 1회 측정하고 EMA 필터를 갱신한다.
        **한 제어 틱에 한 번만** 호출할 것 (ToFPair.read() 가 관리).

        [중요] raw 값과 EMA 값을 둘 다 보관한다.
        EMA 는 조향 오차용이고, "벽이 있는가/없는가" 판정은 **raw** 로 한다.
        EMA 로 out-of-range 를 판정하면 임계값에 점근할 뿐 도달하지 못해
        (150mm -> 800mm 전환 시 알파 0.35 기준 약 4초 지연) 고랑 끝 감지가
        치명적으로 늦어진다. 노이즈 방어는 EMA 가 아니라 연속 틱 확인
        (TOF_END_CONFIRM_TICKS) 으로 한다.
        """
        raw = self._read_raw_mm()
        raw = min(raw, TOF_OUT_OF_RANGE_MM)  # 상한 클램프
        self.last_raw_mm = raw
        if self._ema_value is None:
            self._ema_value = raw
        else:
            self._ema_value = TOF_EMA_ALPHA * raw + (1 - TOF_EMA_ALPHA) * self._ema_value
        self._last_mm = self._ema_value
        return self._last_mm

    def wall_visible(self) -> bool:
        """마지막 raw 측정 기준으로 옆에 벽(이랑)이 보이는가."""
        return self.last_raw_mm < TOF_OUT_OF_RANGE_MM

    @property
    def value_mm(self) -> float:
        """마지막으로 샘플링된 필터값 (재측정하지 않음)."""
        return self._last_mm

    def close(self):
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None


# ======================================================================
# 드라이버 백엔드
# ======================================================================
class _SimDriver:
    """하드웨어 없는 개발 환경용. 고랑 안에 있다고 가정한 값을 낸다."""

    is_real = False

    def __init__(self, name):
        self.name = name
        self.override = None  # 테스트에서 값을 강제 주입할 때 사용

    def read_mm(self):
        if self.override is not None:
            return self.override
        return TOF_NOMINAL_WALL_DISTANCE_MM

    def close(self):
        pass


class _DeadDriver:
    """
    [신규] 초기화에 실패한 센서용.

    예전에는 초기화 실패 시 _SimDriver 로 대체했는데, 이건 실기에서 매우
    위험하다. 배선이 끊겨 있어도 "항상 고랑 한가운데(150mm)"라는 **가짜 값**이
    나오므로 로봇은 고랑 끝을 영원히 감지하지 못한 채 계속 직진한다.
    (물을 뿌리면서 밭을 벗어날 수 있다)
    이제는 명시적으로 '측정 불가'를 반환해서 상위 안전 로직
    (FURROW_WALL_ACQUIRE_SEC -> SAFE_HALT)이 정상적으로 걸리게 한다.
    """

    is_real = False

    def read_mm(self):
        return None

    def close(self):
        pass


class _AdafruitDriver:
    """
    adafruit-circuitpython-vl53l1x 백엔드 (주소 재할당 지원).

    [수정/치명적] 이 라이브러리는 반드시 아래 순서를 지켜야 한다.
        if sensor.data_ready:
            d = sensor.distance
            sensor.clear_interrupt()
    clear_interrupt() 를 부르지 않으면 센서가 다음 측정을 시작하지 않아
    **첫 측정 이후 값이 영원히 갱신되지 않는다**. 예전 코드는 이 호출이
    없었기 때문에 실기에서 좌우 ToF 가 곧바로 멈춰버리고, 그 결과
    "벽이 안 보임 -> 고랑 끝" 오판 또는 SAFE_HALT 로 끝났을 것이다.

    또한 data_ready 확인 없이 .distance 를 읽으면 옛 값이나 None 이 나온다.
    측정이 아직 준비되지 않았으면 None 을 반환하고, ToFSensor 가 직전
    유효값을 유지한다(TOF_STALE_TIMEOUT_SEC 참고).
    """

    is_real = True

    def __init__(self, sensor):
        self._sensor = sensor

    def read_mm(self):
        if not self._sensor.data_ready:
            return None  # 아직 측정 중 -> 상위에서 직전 값 유지
        cm = self._sensor.distance  # cm 단위, 측정 실패 시 None
        self._sensor.clear_interrupt()  # [필수] 다음 측정 트리거
        return None if cm is None else cm * 10.0

    def close(self):
        try:
            self._sensor.stop_ranging()
        except Exception:
            pass


class _PimoroniDriver:
    """pimoroni VL53L1X 백엔드."""

    is_real = True

    def __init__(self, sensor):
        self._sensor = sensor

    def read_mm(self):
        return self._sensor.get_distance()

    def close(self):
        try:
            self._sensor.stop_ranging()
            self._sensor.close()
        except Exception:
            pass


# ======================================================================
class ToFPair:
    """
    좌/우 ToF 센서를 한번에 관리.

    [핵심] VL53L1X 는 전원 인가 시 모두 같은 기본 주소(0x29)를 갖는다.
    따라서 반드시 아래 순서로 초기화해야 한다.
        1) 모든 XSHUT 를 LOW 로 내려 전부 끈다
        2) 센서 하나만 XSHUT HIGH -> 0x29 로 접속 -> 새 주소로 변경
        3) 다음 센서에 대해 2) 반복
    """

    def __init__(self, left_cfg: dict, right_cfg: dict, backend: str = None):
        self.backend = (backend or TOF_BACKEND).lower()
        self.left = ToFSensor("left", left_cfg["i2c_address"], left_cfg["xshut_pin"])
        self.right = ToFSensor("right", right_cfg["i2c_address"], right_cfg["xshut_pin"])

        self._end_streak = 0
        self._i2c = None
        self._initialized = False
        self.init_hardware()

    # ------------------------------------------------------------------
    def init_hardware(self):
        # The constructor initializes once. Re-running XSHUT would reset the
        # live sensors back to 0x29 and invalidate their driver objects.
        if self._initialized:
            return

        # PC 개발환경(GPIO 없음) 또는 명시적 sim 백엔드
        if self.backend == "sim" or not _HAS_GPIO:
            if self.backend != "sim":
                log.info("RPi.GPIO 없음 - ToF 를 시뮬레이션 백엔드로 대체합니다.")
            self.left.attach_driver(_SimDriver("left"))
            self.right.attach_driver(_SimDriver("right"))
            self._initialized = True
            return

        try:
            self._init_with_xshut()
            self._initialized = True
            return
        except Exception as exc:
            log.error("ToF 초기화 실패: %s", exc)
            # Do not leave one sensor/driver half-initialized after failure.
            for sensor in (self.left, self.right):
                sensor.close()
            if self._i2c is not None:
                try:
                    self._i2c.deinit()
                except Exception:
                    pass
                self._i2c = None
            for pin in (self.left.xshut_pin, self.right.xshut_pin):
                try:
                    GPIO.output(pin, GPIO.LOW)
                except Exception:
                    pass

        # [수정/중요] 실기에서 초기화에 실패했을 때 가짜 값을 만들지 않는다.
        # 가짜 값(=항상 고랑 한가운데)은 로봇이 고랑 끝을 감지하지 못한 채
        # 계속 직진하게 만든다. 명시적으로 '측정 불가'를 내보내
        # 상위 안전 로직이 SAFE_HALT 를 걸도록 한다.
        if TOF_ALLOW_SIM_FALLBACK:
            log.warning(
                "TOF_ALLOW_SIM_FALLBACK=True 이므로 더미값으로 대체합니다. "
                "실기 주행에서는 절대 이 설정을 쓰지 마세요."
            )
            self.left.attach_driver(_SimDriver("left"))
            self.right.attach_driver(_SimDriver("right"))
        else:
            log.error(
                "ToF 를 사용할 수 없는 상태로 계속합니다. 고랑에 진입해도 벽이 "
                "감지되지 않으므로 %s 안에 SAFE_HALT 로 정지합니다.", "FURROW_WALL_ACQUIRE_SEC"
            )
            self.left.attach_driver(_DeadDriver())
            self.right.attach_driver(_DeadDriver())
        self._initialized = False

    def _init_with_xshut(self):
        sensors = [self.left, self.right]

        # 1) 전부 끄기
        GPIO.setmode(GPIO.BCM)
        for s in sensors:
            GPIO.setup(s.xshut_pin, GPIO.OUT)
            GPIO.output(s.xshut_pin, GPIO.LOW)
        time.sleep(TOF_INIT_SETTLE_SEC * 4)

        # 2) 하나씩 켜면서 주소 재할당
        for s in sensors:
            GPIO.output(s.xshut_pin, GPIO.HIGH)
            time.sleep(TOF_INIT_SETTLE_SEC)
            driver = self._open_one(s)
            s.attach_driver(driver)
            log.info("%s ToF 초기화 완료 (주소 0x%02X)", s.name, s.i2c_address)

    def _open_one(self, s: ToFSensor):
        if self.backend == "adafruit":
            import adafruit_vl53l1x
            import board
            import busio

            if self._i2c is None:
                self._i2c = busio.I2C(board.SCL, board.SDA)

            sensor = adafruit_vl53l1x.VL53L1X(
                self._i2c, address=TOF_DEFAULT_I2C_ADDRESS
            )
            # 기본 주소(0x29)에서 접속한 뒤 곧바로 고유 주소로 옮긴다
            sensor.set_address(s.i2c_address)
            sensor.distance_mode = 1  # 1 = Short (실외 밝은 환경에서 더 안정적)
            sensor.timing_budget = TOF_TIMING_BUDGET_MS
            sensor.start_ranging()

            # 첫 측정이 나올 때까지 잠깐 기다렸다가 인터럽트를 비운다.
            # (이걸 안 하면 첫 read_mm() 이 항상 '준비 안 됨'으로 나온다)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if sensor.data_ready:
                    _ = sensor.distance
                    sensor.clear_interrupt()
                    break
                time.sleep(0.01)
            return _AdafruitDriver(sensor)

        if self.backend == "pimoroni":
            import VL53L1X

            sensor = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=TOF_DEFAULT_I2C_ADDRESS)
            sensor.open()
            sensor.change_address(s.i2c_address)
            sensor.close()

            sensor = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=s.i2c_address)
            sensor.open()
            sensor.start_ranging(1)  # 1 = Short range
            return _PimoroniDriver(sensor)

        raise ValueError(f"알 수 없는 ToF 백엔드: {self.backend}")

    # ------------------------------------------------------------------
    def read(self):
        """
        좌/우를 각각 **1회씩만** 측정하고 (left_mm, right_mm) 반환.
        한 제어 틱에 한 번만 호출할 것.
        """
        left_mm = self.left.sample()
        right_mm = self.right.sample()

        # 벽 유무 판정은 EMA 가 아니라 raw 로 한다 (위 sample() 주석 참고)
        both_out = not self.left.wall_visible() and not self.right.wall_visible()
        if both_out:
            self._end_streak += 1
        else:
            self._end_streak = 0

        return left_mm, right_mm

    def both_out_of_range(self) -> bool:
        """
        좌우 모두 벽을 못 찾음 = 고랑 끝. 재측정하지 않고 read() 결과를 쓴다.
        노이즈 1회로 확정하지 않도록 연속 TOF_END_CONFIRM_TICKS 틱을 요구한다.
        """
        return self._end_streak >= TOF_END_CONFIRM_TICKS

    def walls_visible(self) -> bool:
        """좌우 모두 벽이 보이는가 = ToF 조향 오차를 신뢰할 수 있는가."""
        return self.left.wall_visible() and self.right.wall_visible()

    def reset_end_detection(self):
        self._end_streak = 0

    def healthy(self) -> bool:
        return self.left.fail_count < 20 and self.right.fail_count < 20

    def hardware_ok(self) -> bool:
        """
        [신규] 실제 ToF 하드웨어가 두 개 다 붙어서 초기화까지 끝났는가.
        임무 시작 전 점검(REQUIRE_REAL_SENSORS)에 쓰인다.
        """
        return bool(self._initialized and self.left.is_real and self.right.is_real)

    def close(self):
        self.left.close()
        self.right.close()
        if _HAS_GPIO and self.backend != "sim":
            for pin in (self.left.xshut_pin, self.right.xshut_pin):
                try:
                    GPIO.cleanup(pin)
                except Exception:
                    pass


if __name__ == "__main__":
    from config import TOF_LEFT, TOF_RIGHT

    pair = ToFPair(TOF_LEFT, TOF_RIGHT)
    try:
        while True:
            l, r = pair.read()
            print(
                f"left={l:7.1f}mm right={r:7.1f}mm diff={r - l:+7.1f} "
                f"end={pair.both_out_of_range()}"
            )
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        pair.close()

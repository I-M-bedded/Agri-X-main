# -*- coding: utf-8 -*-
"""VL53L1X x2 XSHUT/address diagnostic using the Raspberry Pi header I2C.

This test separates wiring failures from address-initialization failures:
both off -> left only -> right only -> assign 0x30/0x31 -> read distances.
Stop every other program using the ToF sensors before running it.
"""

from pathlib import Path
import sys
import time

FARM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FARM_ROOT))

from config import (  # noqa: E402
    TOF_DEFAULT_I2C_ADDRESS,
    TOF_INIT_SETTLE_SEC,
    TOF_LEFT,
    TOF_RIGHT,
    TOF_TIMING_BUDGET_MS,
)


def address_list(addresses):
    return " ".join(f"0x{address:02X}" for address in addresses) or "(none)"


def scan(i2c):
    deadline = time.monotonic() + 2.0
    while not i2c.try_lock():
        if time.monotonic() >= deadline:
            raise RuntimeError("I2C bus lock timeout; stop other sensor programs")
        time.sleep(0.01)
    try:
        return i2c.scan()
    finally:
        i2c.unlock()


def expect_default(stage, addresses, should_exist):
    present = TOF_DEFAULT_I2C_ADDRESS in addresses
    expected = "present" if should_exist else "absent"
    result = "PASS" if present == should_exist else "FAIL"
    print(
        f"[{result}] {stage}: {address_list(addresses)} "
        f"(0x{TOF_DEFAULT_I2C_ADDRESS:02X} expected {expected})")
    return present == should_exist


def wait_distance(sensor, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sensor.data_ready:
            value = sensor.distance
            sensor.clear_interrupt()
            return value
        time.sleep(0.01)
    return None


def main():
    try:
        import RPi.GPIO as GPIO
        import adafruit_vl53l1x
        import board
        import busio
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        print(
            "Install: pip install RPi.GPIO adafruit-blinka "
            "adafruit-circuitpython-vl53l1x",
            file=sys.stderr,
        )
        return 2

    left_pin = int(TOF_LEFT["xshut_pin"])
    right_pin = int(TOF_RIGHT["xshut_pin"])
    left_address = int(TOF_LEFT["i2c_address"])
    right_address = int(TOF_RIGHT["i2c_address"])
    settle = max(0.05, float(TOF_INIT_SETTLE_SEC))

    print("Agri-X VL53L1X XSHUT diagnostic")
    print(f"left XSHUT=BCM{left_pin}, right XSHUT=BCM{right_pin}")
    print(f"target addresses=0x{left_address:02X}, 0x{right_address:02X}")

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    i2c = busio.I2C(board.SCL, board.SDA)
    sensors = []
    stages_ok = True

    try:
        GPIO.setup(left_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(right_pin, GPIO.OUT, initial=GPIO.LOW)
        time.sleep(settle * 4)
        stages_ok &= expect_default("both XSHUT LOW", scan(i2c), False)

        GPIO.output(left_pin, GPIO.HIGH)
        time.sleep(settle * 2)
        stages_ok &= expect_default("left only HIGH", scan(i2c), True)

        GPIO.output(left_pin, GPIO.LOW)
        GPIO.output(right_pin, GPIO.HIGH)
        time.sleep(settle * 2)
        stages_ok &= expect_default("right only HIGH", scan(i2c), True)

        GPIO.output(right_pin, GPIO.LOW)
        time.sleep(settle * 2)
        if not stages_ok:
            print("\nXSHUT wiring test failed; address assignment was not attempted.")
            return 1

        GPIO.output(left_pin, GPIO.HIGH)
        time.sleep(settle * 2)
        left = adafruit_vl53l1x.VL53L1X(
            i2c, address=TOF_DEFAULT_I2C_ADDRESS)
        left.set_address(left_address)
        sensors.append(left)
        print(f"[PASS] left address changed to 0x{left_address:02X}")

        GPIO.output(right_pin, GPIO.HIGH)
        time.sleep(settle * 2)
        right = adafruit_vl53l1x.VL53L1X(
            i2c, address=TOF_DEFAULT_I2C_ADDRESS)
        right.set_address(right_address)
        sensors.append(right)
        print(f"[PASS] right address changed to 0x{right_address:02X}")

        addresses = scan(i2c)
        final_ok = left_address in addresses and right_address in addresses
        print(
            f"[{'PASS' if final_ok else 'FAIL'}] final scan: "
            f"{address_list(addresses)}")
        if not final_ok:
            return 1

        for sensor in sensors:
            sensor.distance_mode = 1
            sensor.timing_budget = TOF_TIMING_BUDGET_MS
            sensor.start_ranging()
        left_cm = wait_distance(left)
        right_cm = wait_distance(right)
        print(
            "[{}] distance: left={} cm, right={} cm".format(
                "PASS" if left_cm is not None and right_cm is not None else "FAIL",
                left_cm,
                right_cm,
            ))
        return 0 if left_cm is not None and right_cm is not None else 1
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        for sensor in sensors:
            try:
                sensor.stop_ranging()
            except Exception:
                pass
        try:
            GPIO.output(left_pin, GPIO.LOW)
            GPIO.output(right_pin, GPIO.LOW)
            time.sleep(settle)
        except Exception:
            pass
        try:
            GPIO.cleanup((left_pin, right_pin))
        except Exception:
            pass
        try:
            i2c.deinit()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

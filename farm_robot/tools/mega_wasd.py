#!/usr/bin/env python3
"""Interactive WASD bring-up through the production MegaMotion abstraction.

Unlike the Mega firmware's raw single-byte WASD test, this tool uses the same
``MegaMotion`` transport/protocol layer as autonomous driving and sends explicit
physical wheel-speed targets with ``DRIVE left_rpm right_rpm``.

Terminal input does not provide key-up events. A held key is therefore inferred
from OS key-repeat events. The first key press starts at a moderate RPM; repeated
same-direction events increase the target linearly toward the configured maximum.
The target never ramps by itself after key repeats stop.

Space toggles the existing MOSFET-based ``PumpController``. With the current
LR7843 path, ON means full output (100%); there is no PWM percentage control in
this bring-up tool. The manual test explicitly opens the pump zone interlock while
it is running, but retains the pump continuous-run watchdog and always turns the
pump off during cleanup.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import select
import sys
import termios
import time
import tty

_FARM_ROBOT = str(Path(__file__).resolve().parents[1])
if _FARM_ROBOT not in sys.path:
    sys.path.insert(0, _FARM_ROBOT)

from actuators.pump_controller import PumpController  # noqa: E402
from control.mega_motion import MegaMotion  # noqa: E402


REFRESH_SEC = 0.10
DEFAULT_START_RPM = 60.0
DEFAULT_MAX_RPM = 120.0
DEFAULT_RAMP_SEC = 1.20
DEFAULT_TURN_SCALE = 0.85
INITIAL_REPEAT_GRACE_SEC = 0.70
RELEASE_DEADMAN_SEC = 0.25
MAX_TEST_RPM = 145.0  # firmware clamps at 150 RPM; retain a little margin
RPM_STEP = 10.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def ramp_rpm(start_rpm: float, max_rpm: float, held_sec: float, ramp_sec: float) -> float:
    if ramp_sec <= 0.0:
        return max_rpm
    alpha = clamp(held_sec / ramp_sec, 0.0, 1.0)
    return start_rpm + alpha * (max_rpm - start_rpm)


def wheel_targets(key: str, rpm: float, turn_scale: float) -> tuple[float, float]:
    if key == "w":
        return rpm, rpm
    if key == "s":
        return -rpm, -rpm
    turn_rpm = rpm * turn_scale
    if key == "a":
        return -turn_rpm, turn_rpm
    if key == "d":
        return turn_rpm, -turn_rpm
    raise ValueError(f"unsupported motion key: {key}")


def print_help(start_rpm: float, max_rpm: float, ramp_sec: float, turn_scale: float) -> None:
    print(
        "\nControls: W forward | S reverse | A left | D right | "
        "X motor stop | Space pump ON/OFF | P state | +/- max RPM | Q quit"
    )
    print(
        f"Hold/repeat: {start_rpm:.0f} -> {max_rpm:.0f} RPM over {ramp_sec:.1f}s; "
        f"turn scale={turn_scale:.2f}."
    )
    print(
        f"First-repeat grace={INITIAL_REPEAT_GRACE_SEC:.2f}s, "
        f"release stop={RELEASE_DEADMAN_SEC:.2f}s after repeats begin."
    )
    print("Pump: LR7843 MOSFET output, Space toggles OFF <-> ON (100%).\n")


def print_state(motion: MegaMotion, pump: PumpController) -> None:
    motion.request_status()
    time.sleep(0.06)
    state = motion.state
    print(
        f"\nSTATE mode={state.mode} "
        f"L={state.left_rpm:+.1f}rpm R={state.right_rpm:+.1f}rpm "
        f"Ldeg={state.left_degrees:+.1f} Rdeg={state.right_degrees:+.1f} "
        f"pump={'ON(100%)' if pump.is_on() else 'OFF'}"
    )


def toggle_pump(pump: PumpController) -> None:
    if pump.is_on():
        pump.turn_off()
        print("\nPUMP OFF")
        return

    if not getattr(pump, "_gpio_ready", False):
        print("\nPUMP unavailable: GPIO initialization failed", file=sys.stderr)
        return

    if pump.turn_on():
        print("\nPUMP ON (100%)")
    else:
        print("\nPUMP ON blocked by PumpController interlock", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ramped WASD and pump test through production robot abstractions"
    )
    parser.add_argument("--port", help="Mega serial device; auto-detect when omitted")
    parser.add_argument(
        "--start-rpm",
        type=float,
        default=DEFAULT_START_RPM,
        help=f"RPM on first key press (default: {DEFAULT_START_RPM:.0f})",
    )
    parser.add_argument(
        "--max-rpm",
        type=float,
        default=DEFAULT_MAX_RPM,
        help=f"RPM after holding the key (default: {DEFAULT_MAX_RPM:.0f})",
    )
    parser.add_argument(
        "--ramp-sec",
        type=float,
        default=DEFAULT_RAMP_SEC,
        help=f"hold time to reach max RPM (default: {DEFAULT_RAMP_SEC:.1f}s)",
    )
    parser.add_argument(
        "--turn-scale",
        type=float,
        default=DEFAULT_TURN_SCALE,
        help=f"in-place turn RPM scale (default: {DEFAULT_TURN_SCALE:.2f})",
    )
    args = parser.parse_args()

    if not sys.stdin.isatty():
        print("This controller must be run in an interactive terminal.", file=sys.stderr)
        return 2

    max_rpm = clamp(args.max_rpm, 20.0, MAX_TEST_RPM)
    start_rpm = clamp(args.start_rpm, 10.0, max_rpm)
    ramp_sec = clamp(args.ramp_sec, 0.0, 5.0)
    turn_scale = clamp(args.turn_scale, 0.40, 1.00)

    motion = MegaMotion(port=args.port or None)
    if not motion.available or motion.faulted:
        print(f"Mega Motion connection failed: {motion.last_error}", file=sys.stderr)
        motion.cleanup()
        return 1

    pump = PumpController()
    # This is an explicit manual hardware bring-up tool. Autonomous driving
    # continues to control the real in-furrow zone interlock separately.
    pump.set_zone(True)
    pump.turn_off()

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)

    active_key: str | None = None
    held_since = 0.0
    last_key_event = 0.0
    current_rpm = start_rpm
    next_refresh = 0.0
    repeat_seen = False

    try:
        print("MegaMotion abstraction connected.")
        print(f"refresh={REFRESH_SEC:.2f}s; firmware DRIVE watchdog remains active")
        if getattr(pump, "_gpio_ready", False):
            print("PumpController GPIO ready; Space toggles LR7843 MOSFET at 100% output")
        else:
            print("WARNING: PumpController GPIO is not ready; Space cannot drive the pump")
        print_help(start_rpm, max_rpm, ramp_sec, turn_scale)
        tty.setcbreak(fd)

        while True:
            now = time.monotonic()
            readable, _, _ = select.select([sys.stdin], [], [], 0.02)

            if readable:
                key = sys.stdin.read(1).lower()
                now = time.monotonic()

                if key in "wasd":
                    same_continuous_key = (
                        active_key == key
                        and last_key_event > 0.0
                        and (now - last_key_event) <= INITIAL_REPEAT_GRACE_SEC
                    )
                    if same_continuous_key:
                        repeat_seen = True
                    else:
                        active_key = key
                        held_since = now
                        repeat_seen = False
                        current_rpm = start_rpm

                    active_key = key
                    last_key_event = now
                    current_rpm = ramp_rpm(
                        start_rpm,
                        max_rpm,
                        now - held_since,
                        ramp_sec,
                    )
                    left, right = wheel_targets(active_key, current_rpm, turn_scale)
                    if not motion.set_wheel_rpm(left, right):
                        raise RuntimeError(motion.last_error or "failed to send DRIVE")
                    next_refresh = now + REFRESH_SEC
                    print(
                        f"\rDRIVE L={left:+.0f} R={right:+.0f} RPM "
                        f"hold={now - held_since:.2f}s "
                        f"pump={'ON' if pump.is_on() else 'OFF'}          ",
                        end="",
                        flush=True,
                    )

                elif key == "x":
                    active_key = None
                    repeat_seen = False
                    motion.stop()
                    print("\rMOTOR STOP                                     ", end="", flush=True)

                elif key == " ":
                    toggle_pump(pump)

                elif key == "p":
                    print_state(motion, pump)

                elif key in ("+", "="):
                    max_rpm = clamp(max_rpm + RPM_STEP, start_rpm, MAX_TEST_RPM)
                    print(f"\nmax RPM={max_rpm:.0f}")

                elif key in ("-", "_"):
                    max_rpm = clamp(max_rpm - RPM_STEP, start_rpm, MAX_TEST_RPM)
                    print(f"\nmax RPM={max_rpm:.0f}")

                elif key == "q":
                    break

                elif key == "h":
                    print_help(start_rpm, max_rpm, ramp_sec, turn_scale)

            now = time.monotonic()
            if active_key is not None:
                timeout = RELEASE_DEADMAN_SEC if repeat_seen else INITIAL_REPEAT_GRACE_SEC
                if (now - last_key_event) >= timeout:
                    active_key = None
                    repeat_seen = False
                    motion.stop()
                    print("\rDEADMAN MOTOR STOP                             ", end="", flush=True)
                elif now >= next_refresh:
                    left, right = wheel_targets(active_key, current_rpm, turn_scale)
                    if not motion.set_wheel_rpm(left, right):
                        raise RuntimeError(motion.last_error or "failed to refresh DRIVE")
                    next_refresh = now + REFRESH_SEC

            # Keep PumpController's maximum-continuous-run watchdog active even
            # while this manual teleop loop is otherwise idle.
            pump.tick()

            if motion.faulted:
                raise RuntimeError(motion.last_error or "Mega Motion fault")

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nController stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            pump.turn_off()
            motion.stop()
        finally:
            pump.cleanup()
            motion.cleanup()
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        print("\nStopped. Pump OFF, motors STOP.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

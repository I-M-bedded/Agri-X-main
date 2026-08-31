# -*- coding: utf-8 -*-
"""
tools/sim_entry_experiment.py
------------------------------
**최초 마커 검출 → 고랑 정렬** 구간만 떼어내 실험한다.

왜 따로 만드는가
  맵이 없으므로 "첫 마커를 찾는 것" 이 전체 임무의 성패를 가른다. 그런데
  기존 실험(sim_vision_experiment.py)은 로봇이 이미 고랑 정면을 보고 서 있는
  유리한 초기 조건(x=0, y=-1, theta=90도)에서만 돌렸다. 실제로는 로봇이
  헤드랜드(이랑과 수직인 선) 위에서 출발하므로, 그 조건을 바꿔가며 봐야 한다.

두 가지 진입 시퀀스를 비교한다
  A) mission  : navigation/mission_state_machine.py (main.py 가 쓰는 것)
                시작 헤딩을 '밭 안쪽'으로 **가정**한다(_field_heading = 초기 theta).
  B) entry    : navigation/aruco_entry_fsm.py
                마커 탐색 -> 접근 -> 정지 -> **90도 선회** -> 고랑 추종에 인계.
                로봇이 이랑과 수직으로 접근하는 상황을 정면으로 다룬다.

팻말 각도 규약 (시뮬레이터와 동일)
  0도  = 팻말 면이 고랑 정면(-y)을 향함
  90도 = 팻말 면이 이랑과 평행(옆, ±x)을 향함
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


def _make_world(n_furrows, tilt_deg, start_x, start_y, start_theta_deg):
    import tools.simulation as sim
    from tools.simulation import SimWorld

    saved = sim.MARKER_POST_TILT_DEG
    sim.MARKER_POST_TILT_DEG = tilt_deg
    world = SimWorld(n_furrows=n_furrows)
    world.markers = world._build_markers()
    sim.MARKER_POST_TILT_DEG = saved

    world.x, world.y = start_x, start_y
    world.theta = math.radians(start_theta_deg)
    return world


def run_entry(tilt_deg, start_x, start_y, start_theta_deg,
              n_furrows=3, target_marker=1, max_seconds=60.0):
    """B) aruco_entry_fsm 경로: 마커 찾기 -> 접근 -> 90도 선회."""
    from actuators.motor_driver import MotorDriver
    from navigation.aruco_entry_fsm import ArucoEntryConfig, ArucoEntryFSM, EntryState
    from sensors.odometry import Odometry
    from tools.simulation import FakeClock, SimAruco

    world = _make_world(n_furrows, tilt_deg, start_x, start_y, start_theta_deg)
    clock = FakeClock()
    real_monotonic, real_sleep = time.monotonic, time.sleep
    time.monotonic, time.sleep = clock.monotonic, clock.sleep
    try:
        odom = Odometry()
        motors = MotorDriver(odometry=odom)
        world.motors, world.odom = motors, odom
        clock.on_advance = world.integrate

        # 로봇이 헤드랜드를 +x 로 진행 중이면 고랑은 왼쪽(CCW, +90도)에 있다.
        turn = math.pi / 2.0 if math.cos(math.radians(start_theta_deg)) > 0 else -math.pi / 2.0
        cfg = ArucoEntryConfig(target_marker_id=target_marker, turn_angle_rad=turn)
        fsm = ArucoEntryFSM(motors=motors, odom=odom, aruco=SimAruco(world), config=cfg)

        t0 = clock.now
        while fsm.state not in (EntryState.DONE, EntryState.SAFE_HALT):
            fsm.step()
            time.sleep(cfg.loop_dt_sec)
            if clock.now - t0 > max_seconds:
                break

        # 성공 판정: DONE 이면서 고랑 축(+y)을 향하고, 목표 고랑 중심 근처인가
        cx = (target_marker - 1) * world.FURROW_SPACING_M
        heading_err = abs(math.degrees(math.atan2(
            math.sin(world.theta - math.pi / 2), math.cos(world.theta - math.pi / 2))))
        lateral_err = abs(world.x - cx)
        return {
            "state": fsm.state.name,
            "done": fsm.state == EntryState.DONE,
            "heading_err_deg": heading_err,
            "lateral_err_m": lateral_err,
            # 고랑 반폭 안쪽 + 헤딩 25도 이내면 "고랑 추종에 인계 가능"
            "handoff_ok": (fsm.state == EntryState.DONE
                           and lateral_err <= world.FURROW_HALF_WIDTH_M
                           and heading_err <= 25.0),
        }
    finally:
        time.monotonic, time.sleep = real_monotonic, real_sleep


def run_mission_entry(tilt_deg, start_x, start_y, start_theta_deg,
                      n_furrows=3, seed=0):
    """A) mission_state_machine 경로: 첫 고랑 진입 성공 여부만 본다."""
    from tools.sim_vision_experiment import run_trial

    r = run_trial(seed=seed, n_furrows=n_furrows,
                  start_x=start_x, start_y=start_y,
                  start_theta_deg=start_theta_deg, marker_tilt_deg=tilt_deg)
    return {"state": r["state"], "handoff_ok": r["completed_furrows"] >= 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--furrows", type=int, default=3)
    args = ap.parse_args()

    TILTS = (0, 30, 45, 60, 90)
    STARTS = (-0.6, 0.0, 0.6)
    HEADINGS = (
        (90, "밭 안쪽 응시"),
        (0, "헤드랜드 +x 진행"),
        (180, "헤드랜드 -x 진행"),
    )

    print("=" * 88)
    print("최초 마커 검출 -> 고랑 정렬 실험")
    print("  팻말 각도: 0도=고랑 정면을 봄 / 90도=이랑에 평행(옆을 봄)")
    print("  성공 = 고랑 추종에 인계 가능 (중심 오차 <= 고랑반폭, 헤딩 오차 <= 25도)")
    print("=" * 88)

    for label, runner in (("A) mission_state_machine (main.py 경로)", run_mission_entry),
                          ("B) aruco_entry_fsm (마커->정지->90도 선회)", run_entry)):
        print(f"\n### {label}")
        print(f"    {'시작 헤딩':<16s} |" + "".join(f" {t:3d}도 |" for t in TILTS))
        print("    " + "-" * (18 + 8 * len(TILTS)))
        for th, thname in HEADINGS:
            row = f"    {thname:<16s} |"
            for tilt in TILTS:
                ok = 0
                for sx in STARTS:
                    try:
                        r = runner(tilt, sx, -1.0, th, n_furrows=args.furrows)
                        ok += 1 if r["handoff_ok"] else 0
                    except Exception:
                        pass
                row += f" {ok}/{len(STARTS)}  |"
            print(row)


if __name__ == "__main__":
    main()

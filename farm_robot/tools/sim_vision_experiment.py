# -*- coding: utf-8 -*-
"""
tools/sim_vision_experiment.py
-------------------------------
비전이 **실측 오차 분포대로 틀릴 때** 폐루프(시퀀스)가 견디는지 확인하는 실험.

    python tools/sim_vision_experiment.py
    python tools/sim_vision_experiment.py --trials 60

무엇을 재는가
  완주율          MISSION_COMPLETE 로 끝난 비율
  SAFE_HALT율     스스로 위험을 감지하고 멈춘 비율 (폭주보다 훨씬 낫다)
  최대 횡편차     고랑 중심에서 벗어난 최대 거리(m)
  이랑 밟음       |횡편차| > 여유(고랑 반폭 - 차체 반폭) 인 시행 수
  게이트 통과 오답 conf >= VISION_MIN_CONFIDENCE 인데 오차가 컸던 프레임 수

판정 기준 (차체 폭 20cm, 고랑 폭 40cm 기준)
  여유 = 0.20(고랑 반폭) - 0.10(차체 반폭) = 0.10m
  이 값을 넘으면 이랑을 밟은 것으로 본다.
"""

import argparse
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

BODY_HALF_WIDTH_M = 0.10   # 차체 폭 20cm


# 편차를 재는 구간: 실제로 "고랑 안을 주행 중"인 상태만.
# 입구 정렬(SEARCH_AND_ALIGN)은 비스듬히 접근하며 자세를 잡는 단계라
# 중심선 편차가 크게 나오는 것이 정상이므로 측정에서 제외한다.
DRIVING_STATES = ("TRAVEL_INTO_FURROW", "TRAVEL_BACK_TO_ENTRANCE")


def lateral_deviation(world):
    """가장 가까운 고랑 중심선에서의 횡편차(m).

    [중요] world.lateral_offset_in_furrow() 를 쓰면 안 된다. 그 함수는 로봇이
    고랑 밖으로 나가는 순간 None 을 돌려주므로, 편차가 고랑 반폭에서 **포화**해
    "얼마나 크게 이탈했는가" 를 영원히 측정하지 못한다.
    여기서는 고랑 구간(y) 안에 있는 동안 가장 가까운 중심선과의 거리를
    상한 없이 잰다. 헤드랜드 구간은 측정 대상이 아니므로 None.
    """
    if not (0.0 <= world.y <= world.FURROW_LENGTH_M):
        return None
    k = round(world.x / world.FURROW_SPACING_M)
    return world.x - k * world.FURROW_SPACING_M


def run_trial(seed, n_furrows=3, burst_len=1, blind=False,
              furrow_scale=2.5, max_seconds=400.0,
              start_x=None, start_y=None, start_theta_deg=None,
              marker_tilt_deg=None, use_imu=False, slip=0.0,
              rendered_aruco=False, blur_px=0):
    """비전 오차를 주입한 임무 1회. 결과 dict 반환.

    start_* 를 주면 로봇 시작 자세를 바꾼다(기본: 고랑1 정면, 밭 안쪽 응시).
    marker_tilt_deg 를 주면 팻말 기울기를 바꾼다(기본: config 값).
    """
    from navigation.mission_state_machine import (
        MissionStateMachine, MissionState, TERMINAL_STATES,
    )
    from actuators.motor_driver import MotorDriver
    from actuators.pump_controller import PumpController
    from control.line_follower import LineFollower
    from navigation.furrow_manager import FurrowManager
    from sensors.odometry import Odometry
    from sensors.tof_sensor import ToFPair
    from tools.simulation import (FakeClock, SimAruco, SimCamera, SimImu,
                                  SimWaterSensor, SimWorld)
    from tools.vision_error_model import ReplayVision
    from tools.rendered_aruco import RenderedAruco

    import math as _math
    import tools.simulation as _sim
    _saved_tilt = _sim.MARKER_POST_TILT_DEG
    if marker_tilt_deg is not None:
        _sim.MARKER_POST_TILT_DEG = marker_tilt_deg

    world = SimWorld(n_furrows=n_furrows, slip_left=slip, slip_right=slip)
    if marker_tilt_deg is not None:
        world.markers = world._build_markers()   # 바뀐 기울기로 재생성
    _sim.MARKER_POST_TILT_DEG = _saved_tilt
    if start_x is not None:
        world.x = start_x
    if start_y is not None:
        world.y = start_y
    if start_theta_deg is not None:
        world.theta = _math.radians(start_theta_deg)
    clock = FakeClock()
    real_monotonic, real_sleep = time.monotonic, time.sleep
    time.monotonic, time.sleep = clock.monotonic, clock.sleep
    try:
        if use_imu:
            # 거리=엔코더, 방향=자이로. 미끄러짐이 있을 때 진가가 드러난다.
            from sensors.odometry import ImuFusedOdometry
            odom = ImuFusedOdometry(imu=SimImu(world))
        else:
            odom = Odometry()
        motors = MotorDriver(odometry=odom)
        world.motors, world.odom = motors, odom
        clock.on_advance = world.integrate

        tof = ToFPair(config.TOF_LEFT, config.TOF_RIGHT, backend="sim")
        vision = ReplayVision(world, seed=seed, burst_len=burst_len,
                              furrow_scale=furrow_scale, force_blind=blind)

        def sync_tof():
            left_mm, right_mm = world.tof_readings_mm()
            tof.left._driver.override = left_mm
            tof.right._driver.override = right_mm

        deps = {
            "odom": odom, "motors": motors, "pump": PumpController(),
            "tof_pair": tof, "camera": SimCamera(),
            # rendered_aruco=True 면 '면각 70도 하드컷' 대신 **실제 OpenCV
            # 검출기**로 마커 인식을 판정한다(해상도·원근·겹침이 반영된다).
            "aruco": (RenderedAruco(world, blur_px=blur_px) if rendered_aruco
                      else SimAruco(world)),
            "vision_line": vision, "water_sensor": SimWaterSensor(world),
            "furrow_mgr": FurrowManager(),
        }
        deps["line_follower"] = LineFollower(tof, vision_detector=vision,
                                             odometry=odom)
        fsm = MissionStateMachine(deps=deps)

        margin = world.FURROW_HALF_WIDTH_M - BODY_HALF_WIDTH_M
        max_dev, ridge_hits, inside_ticks = 0.0, 0, 0
        visited = set()
        t0 = clock.now
        while fsm.state not in TERMINAL_STATES:
            visited.add(fsm.state.name)
            sync_tof()
            fsm.step()
            off = (lateral_deviation(world)
                   if fsm.state.name in DRIVING_STATES else None)
            if off is not None:
                inside_ticks += 1
                max_dev = max(max_dev, abs(off))
                if abs(off) > margin:
                    ridge_hits += 1
            time.sleep(config.CONTROL_LOOP_DT)
            if clock.now - t0 > max_seconds:
                break

        return {
            "state": fsm.state.name,
            "complete": fsm.state == MissionState.MISSION_COMPLETE,
            "safe_halt": fsm.state == MissionState.SAFE_HALT,
            "completed_furrows": len(fsm.furrow_mgr.completed),
            "max_dev_m": max_dev,
            "margin_m": margin,
            "ridge_hit_ticks": ridge_hits,
            "inside_ticks": inside_ticks,
            "vision": dict(vision.stats),
            "sim_seconds": clock.now - t0,
            # 유턴 -> 복귀 주행을 실제로 거쳤는가
            "turned_around": "TURN_AROUND" in visited,
            "drove_back": "TRAVEL_BACK_TO_ENTRANCE" in visited,
            # HOME(x=0) 에서 최종적으로 얼마나 벗어났는가
            "home_error_m": abs(world.x),
        }
    finally:
        time.monotonic, time.sleep = real_monotonic, real_sleep


def summarize(results, label):
    n = len(results)
    comp = sum(r["complete"] for r in results)
    halt = sum(r["safe_halt"] for r in results)
    devs = [r["max_dev_m"] for r in results]
    hits = sum(1 for r in results if r["ridge_hit_ticks"] > 0)
    return {
        "label": label, "n": n,
        "complete_rate": comp / n, "safe_halt_rate": halt / n,
        "other_rate": (n - comp - halt) / n,
        "max_dev_med": st.median(devs), "max_dev_max": max(devs),
        "margin_m": results[0]["margin_m"],
        "ridge_hit_trials": hits,
        "vision_accepted": sum(r["vision"]["accepted"] for r in results),
        "vision_accepted_bad": sum(r["vision"]["accepted_bad"] for r in results),
        "vision_gated": sum(r["vision"]["gated_out"] for r in results),
        "furrows_done": sum(r["completed_furrows"] for r in results),
    }


def print_row(s):
    print(f"  {s['label']:<26s} 완주 {s['complete_rate']*100:5.1f}%  "
          f"HALT {s['safe_halt_rate']*100:5.1f}%  "
          f"기타 {s['other_rate']*100:5.1f}%  "
          f"편차중앙 {s['max_dev_med']*100:4.1f}cm 최악 {s['max_dev_max']*100:5.1f}cm  "
          f"이랑밟음 {s['ridge_hit_trials']:2d}/{s['n']}")


def set_gate(value):
    """VISION_MIN_CONFIDENCE 를 런타임에 바꾼다(모듈이 from-import 하므로 둘 다)."""
    import control.line_follower as lf_mod

    config.VISION_MIN_CONFIDENCE = value
    lf_mod.VISION_MIN_CONFIDENCE = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--furrows", type=int, default=3)
    args = ap.parse_args()

    furrow_w = config.TOF_NOMINAL_WALL_DISTANCE_MM * 2 / 1000.0
    margin_cm = (config.TOF_NOMINAL_WALL_DISTANCE_MM / 1000.0 - BODY_HALF_WIDTH_M) * 100

    print("=" * 96)
    print("비전 실측 오차 주입 폐루프 실험")
    print(f"  고랑 폭 {furrow_w:.2f}m / 차체 폭 {BODY_HALF_WIDTH_M*2:.2f}m "
          f"-> 좌우 여유 각 {margin_cm:.0f}cm")
    print(f"  시행 {args.trials}회 x 고랑 {args.furrows}개")
    print(f"  오차원: reports/crdld_furrow_v1/metrics_frames.csv (실측 430장)")
    print("=" * 96)

    base_gate = config.VISION_MIN_CONFIDENCE
    out = []

    print(f"\n[실험 1] 실측 오차 주입 (VISION_MIN_CONFIDENCE={base_gate})")
    res = [run_trial(seed=i, n_furrows=args.furrows) for i in range(args.trials)]
    s = summarize(res, "실측 오차 주입")
    print_row(s)
    print(f"    비전 프레임: 통과 {s['vision_accepted']} "
          f"(그중 큰오차 {s['vision_accepted_bad']}), 차단 {s['vision_gated']}")
    out.append(s)

    print(f"\n[실험 2] VISION_MIN_CONFIDENCE 스윕")
    for gate in (0.25, 0.40, 0.55, 0.70):
        set_gate(gate)
        res = [run_trial(seed=i, n_furrows=args.furrows) for i in range(args.trials)]
        s = summarize(res, f"gate={gate:.2f}")
        print_row(s)
        print(f"    비전 프레임: 통과 {s['vision_accepted']} "
              f"(그중 큰오차 {s['vision_accepted_bad']}), 차단 {s['vision_gated']}")
        out.append(s)
    set_gate(base_gate)

    print(f"\n[실험 3] 연속 오답 스트레스 (측정값 아닌 **가정**)")
    for burst in (1, 4, 10, 20):
        res = [run_trial(seed=i, n_furrows=args.furrows, burst_len=burst)
               for i in range(args.trials)]
        s = summarize(res, f"연속 {burst}프레임 ({burst*0.05:.2f}s)")
        print_row(s)
        out.append(s)

    print(f"\n[실험 4] 비전 완전 실명 -> ToF 폴백 (기존 검증 경로)")
    res = [run_trial(seed=i, n_furrows=args.furrows, blind=True)
           for i in range(args.trials)]
    s = summarize(res, "비전 실명")
    print_row(s)
    out.append(s)

    return out


if __name__ == "__main__":
    main()

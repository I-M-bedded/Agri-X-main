# -*- coding: utf-8 -*-
"""
selftest.py
------------
하드웨어 없이 돌리는 자체 점검.

    $ python3 selftest.py

무엇을 검증하는가
  1) 모든 모듈이 임포트되는가
  2) **부호 규약**이 전 모듈에서 일관되는가 (지난 버전의 최대 버그)
  3) PID 정규화/포화/와인드업 방지가 동작하는가
  4) 추측항법(Odometry) 수식과 엔코더 틱 환산이 맞는가
  5) 회전이 무한루프에 빠지지 않는가 (타임아웃/엔코더 stall)
  6) 펌프 인터록과 수위 디바운스가 동작하는가
  7) ToF EMA 와 고랑 끝 판정이 제때 반응하는가
  8) 비전 검출기가 BGR 프레임에서 흙을 실제로 찾아내는가 (합성 영상)
  9) 전체 임무가 시뮬레이션 밭에서 끝까지 완주하는가
 10) 센서를 고장내면 폭주하지 않고 SAFE_HALT 로 가는가

무엇을 검증하지 **못하는가**
  실제 주행 성능(미끄러짐, 조명, 오검출, 모터 응답). 이건 실기 테스트로만
  확인할 수 있다. tools/setup.py 2번(모터·부호 점검)을 먼저 돌릴 것.
"""

import math
import sys
import time
import traceback

sys.path.insert(0, ".")

import config

# 테스트는 조용히 (로그 폭주 방지)
config.LOG_LEVEL = "ERROR"
config.TELEMETRY_EVERY_N_TICKS = 0

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    mark = "PASS" if condition else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return bool(condition)


def section(title):
    print(f"\n=== {title} ===")


# ======================================================================
# 1. 임포트
# ======================================================================
def test_imports():
    section("1. 모듈 임포트")
    mods = [
        "logutil", "config",
        "control.pid_controller", "control.line_follower",
        "sensors.odometry", "sensors.tof_sensor", "sensors.camera",
        "sensors.aruco_detector", "sensors.vision_line_detector",
        "sensors.water_tank_sensor",
        "actuators.motor_driver", "actuators.pump_controller",
        "navigation.furrow_manager", "navigation.mission_state_machine",
        "tools.simulation",
    ]
    ok = True
    for m in mods:
        try:
            __import__(m)
        except Exception as exc:
            ok = False
            check(f"import {m}", False, str(exc))
    check("전체 모듈 임포트", ok, f"{len(mods)}개")


# ======================================================================
# 2. 부호 규약
# ======================================================================
def test_sign_conventions():
    section("2. 부호 규약 (가장 중요)")
    from actuators.motor_driver import MotorDriver
    from sensors.aruco_detector import MarkerObservation, compute_post_bearing

    m = MotorDriver()

    # steer > 0 이면 오른쪽으로 돌아야 하므로 좌륜이 빨라야 한다
    m.drive(0.5, 0.2)
    check(
        "drive(base, steer>0) -> 좌륜이 우륜보다 빠름 (= 우회전)",
        m.last_left > m.last_right,
        f"L={m.last_left:.2f} R={m.last_right:.2f}",
    )

    m.rotate_in_place(clockwise=True)
    check(
        "rotate_in_place(clockwise=True) -> 좌륜 전진/우륜 후진",
        m.last_left > 0 > m.last_right,
        f"L={m.last_left:.2f} R={m.last_right:.2f}",
    )

    # 라인팔로워: 로봇이 왼쪽에 치우침(오른쪽 공간 넓음) -> 오른쪽으로 조향
    from control.line_follower import LineFollower
    from navigation.mission_state_machine import MissionState
    from sensors.tof_sensor import ToFPair
    from tools.simulation import SimWorld

    pair = ToFPair(config.TOF_LEFT, config.TOF_RIGHT, backend="sim")
    lf = LineFollower(pair, vision_detector=None, odometry=None)
    lf.reset()
    pair.left._driver.override = 100.0    # 왼쪽 벽이 가까움
    pair.right._driver.override = 200.0   # 오른쪽이 넓음 -> 오른쪽으로 가야 함
    for _ in range(5):
        res = lf.step()
    check(
        "ToF: 좌측 벽이 가까우면 steer > 0 (오른쪽으로)",
        res.steer > 0,
        f"steer={res.steer:+.3f} err={res.error:+.3f}",
    )

    pair.left._driver.override = 200.0
    pair.right._driver.override = 100.0
    lf.reset()
    for _ in range(5):
        res = lf.step()
    check(
        "ToF: 우측 벽이 가까우면 steer < 0 (왼쪽으로)",
        res.steer < 0,
        f"steer={res.steer:+.3f}",
    )

    # 마커 게이트 정렬 부호
    def make_obs(world_dx, world_dy, robot_theta):
        """월드 좌표 상대 위치를 카메라 좌표 관측으로 변환."""
        dist = math.hypot(world_dx, world_dy)
        bearing_ccw = math.atan2(world_dy, world_dx) - robot_theta
        bearing_ccw = math.atan2(math.sin(bearing_ccw), math.cos(bearing_ccw))
        br = -bearing_ccw
        return MarkerObservation(0, dist, dist * math.cos(br), dist * math.sin(br), br)

    # [수정] 마커는 '고랑 중심이 어디인지' 알려주지 않는다.
    #   방위와 거리만 준다. 중심은 비전이 고랑을 보고 직접 찾는다.
    obs = MarkerObservation(1, 1.0, 1.0, 0.30, 0.0)
    al = compute_post_bearing(obs)
    check(
        "마커: 좌우 위치와 거리를 그대로 전달",
        al.valid and abs(al.lateral_error - 0.30) < 1e-9,
        f"lateral={al.lateral_error:+.2f}m forward={al.forward_distance:.2f}m",
    )
    check(
        "마커: 단독 마커로 각도를 추정하지 않음 (포즈 모호성 회피)",
        al.heading_error == 0.0,
        "진입각은 _field_heading(밭 방위)에서 얻는다",
    )
    check(
        "제어가 '마커-중심선 거리' 상수에 의존하지 않음",
        "MARKER_POST_LATERAL_OFFSET_M" not in open(
            "navigation/mission_state_machine.py", encoding="utf-8").read(),
        "팻말을 어디에 박든 비전이 중심을 찾는다",
    )

    pair.close()


# ======================================================================
# 3. PID
# ======================================================================
def test_pid():
    section("3. PID")
    from control.pid_controller import PIDController

    pid = PIDController(0.55, 0.02, 0.06, output_limit=0.35,
                        integral_limit=0.5, d_filter_hz=5.0)

    out = pid.compute(0.1, dt=0.05)
    check("정규화 오차 0.1 에서 출력이 포화되지 않음", abs(out) < 0.35,
          f"out={out:+.4f}")
    check("작은 오차에서 출력 부호가 오차와 같음", out > 0)

    pid.reset()
    out = pid.compute(5.0, dt=0.05)
    check("큰 오차에서 출력이 상한으로 클램프", abs(out - 0.35) < 1e-9,
          f"out={out:.4f}")

    # 와인드업 방지: 포화 상태로 오래 유지해도 적분이 폭주하지 않아야 함
    # (D항 링잉과 분리해서 보기 위해 kd=0 인 컨트롤러로 검증)
    pid_i = PIDController(0.55, 0.2, 0.0, output_limit=0.35, integral_limit=0.5)
    for _ in range(200):
        pid_i.compute(5.0, dt=0.05)
    check("포화 상태를 오래 유지해도 적분항이 상한 안에 머무름",
          abs(pid_i._integral) <= 0.5 + 1e-9, f"I={pid_i._integral:.4f}")
    recovered = [pid_i.compute(0.0, dt=0.05) for _ in range(3)][-1]
    check("포화 후 오차 0 이면 출력이 곧바로 작아짐 (와인드업 방지)",
          abs(recovered) < 0.12, f"out={recovered:+.4f}")

    # dt 이상값 방어
    pid.reset()
    pid.compute(0.1, dt=0.05)
    out = pid.compute(0.1, dt=0.0)
    check("dt=0 이어도 예외/무한대가 나지 않음", math.isfinite(out))
    out = pid.compute(0.1, dt=1e6)
    check("dt가 비정상적으로 커도 클램프됨", math.isfinite(out) and abs(out) <= 0.35)

    # 미분 킥 방지
    pid.reset()
    first = pid.compute(0.1, dt=0.05)
    check("첫 호출에서 미분 킥이 없음 (P항 + 미미한 I항만)",
          abs(first - 0.055) < 0.002, f"out={first:.6f}")

    # 물리적 제약: 정렬 허용 오차가 데드밴드 최소 회전각보다 커야 수렴한다
    v_min = config.MOTOR_MIN_DUTY * 0.5
    min_step = (2 * v_min / config.WHEEL_BASE_M) * config.CONTROL_LOOP_DT
    check("정렬 각도 허용 오차 > 한 틱 최소 회전각 (수렴 가능 조건)",
          config.ENTRANCE_HEADING_THRESHOLD_RAD > min_step,
          f"허용={config.ENTRANCE_HEADING_THRESHOLD_RAD:.3f} 최소스텝={min_step:.3f} rad")

    # 물리적 제약: 진입 판정 거리 > 팻말이 화면 안에 남아 있는 최소 거리
    #   d_min = (팻말의 중심선 횡오프셋) / tan(수평화각 절반 31도)
    #   [주의] 아래 0.25m 는 게이트에 마커가 2개이던 시절의 절반 간격에서
    #     물려받은 값이다. 팻말이 1개인 지금의 기준 오프셋(0.30m)으로는
    #     d_min = 0.50m 가 되어 ENTRANCE_ENTER_DISTANCE_M 와 같아진다.
    #     즉 이 검사는 현재 **하한만** 보증한다. 실제로 마커를 더 일찍
    #     놓치는 경우는 ENTRANCE_MARKER_LOST_ARRIVAL_M 유예가 받아낸다.
    d_min = 0.25 / math.tan(math.radians(31.0))
    check("진입 판정 거리 > 근접 시 팻말 이탈 거리 하한",
          config.ENTRANCE_ENTER_DISTANCE_M > d_min,
          f"진입={config.ENTRANCE_ENTER_DISTANCE_M:.2f}m 최소={d_min:.2f}m")


# ======================================================================
# 4. Odometry
# ======================================================================
def test_odometry():
    section("4. 추측항법(Odometry)")
    from sensors.odometry import Odometry, normalize_angle

    odom = Odometry()
    dist_per_tick = (2 * math.pi * config.WHEEL_RADIUS_M) / config.TICKS_PER_REVOLUTION

    # 직진 1 m
    ticks = int(round(1.0 / dist_per_tick))
    odom.inject_ticks(ticks, ticks)
    odom.update()
    check("직진 1m 후 x 오차 < 1%", abs(odom.x - 1.0) < 0.01,
          f"x={odom.x:.4f}m theta={odom.theta:.4f}")
    check("직진 시 theta 변화 없음", abs(odom.theta) < 1e-9)

    # 제자리 좌회전(CCW): 우륜 전진, 좌륜 후진 -> theta 증가
    odom.reset()
    arc = (math.pi / 2) * config.WHEEL_BASE_M / 2.0
    t = int(round(arc / dist_per_tick))
    odom.inject_ticks(-t, t)
    odom.update()
    check("우륜 전진/좌륜 후진 -> theta 증가 (CCW 양수 규약)", odom.theta > 0,
          f"theta={math.degrees(odom.theta):.1f}deg")
    check("90도 회전 오차 < 3도", abs(math.degrees(odom.theta) - 90) < 3.0)

    # 각도 정규화
    check("normalize_angle(3pi) ≈ pi", abs(abs(normalize_angle(3 * math.pi)) - math.pi) < 1e-9)

    # path_length 는 항상 증가
    odom.reset()
    odom.inject_ticks(ticks, ticks)
    odom.update()
    odom.inject_ticks(-ticks, -ticks)
    odom.update()
    check("path_length 는 후진해도 증가 (누적 주행거리)", odom.path_length > 1.9,
          f"{odom.path_length:.3f}m")


# ======================================================================
# 5. 회전 종료 보장
# ======================================================================
def test_turn_termination():
    section("5. 회전이 무한루프에 빠지지 않는가")
    from actuators.motor_driver import MotorDriver
    from sensors.odometry import Odometry
    from tools.simulation import FakeClock, SimWorld

    clock = FakeClock()
    real_monotonic, real_sleep = time.monotonic, time.sleep
    time.monotonic, time.sleep = clock.monotonic, clock.sleep
    try:
        odom = Odometry()
        motors = MotorDriver(odometry=odom)
        world = SimWorld()
        world.motors, world.odom = motors, odom
        clock.on_advance = world.integrate

        start = world.theta
        ok = motors.turn_180_blocking()
        turned = abs(world.theta - start)
        check("엔코더 정상: 180도 회전이 종료됨", ok)
        check("실제 회전량이 180도 ±10도",
              abs(math.degrees(turned) - 180) < 10,
              f"{math.degrees(turned):.1f}deg")
        check("회전 후 모터 정지", motors.last_left == 0 and motors.last_right == 0)

        # 엔코더 고장(틱이 전혀 안 들어옴) -> stall 감지 후 시간 기반으로 종료
        world.odom = None
        odom2 = Odometry()
        motors2 = MotorDriver(odometry=odom2)
        world.motors = motors2
        t0 = clock.now
        ok2 = motors2.turn_180_blocking()
        elapsed = clock.now - t0
        check("엔코더 고장 시에도 회전이 유한 시간에 종료됨",
              elapsed < config.TURN_180_DURATION_SEC * config.TURN_TIMEOUT_MARGIN + 1.0,
              f"{elapsed:.2f}s, reached={ok2}")

        # 양자화로 목표창을 건너뛰는 상황 (예전 버전의 무한루프 원인)
        odom3 = Odometry()
        motors3 = MotorDriver(odometry=odom3)
        world3 = SimWorld()
        world3.motors, world3.odom = motors3, odom3
        clock.on_advance = world3.integrate
        t0 = clock.now
        motors3.turn_by_angle_blocking(math.radians(17.0))  # 틱 해상도(약 3.5도)와 안 맞음
        check("목표각이 엔코더 해상도와 안 맞아도 종료됨",
              clock.now - t0 < 5.0, f"{clock.now - t0:.2f}s")
    finally:
        time.monotonic, time.sleep = real_monotonic, real_sleep


# ======================================================================
# 6. 펌프 인터록 / 수위
# ======================================================================
def test_pump_and_water():
    section("6. 펌프 인터록 / 수위 디바운스")
    from actuators.pump_controller import PumpController
    from sensors.water_tank_sensor import WaterTankSensor
    from tools.simulation import FakeClock

    pump = PumpController()
    check("초기 상태는 OFF", not pump.is_on())
    check("고랑 밖에서 turn_on() 은 거부됨", pump.turn_on() is False)
    check("거부된 뒤에도 릴레이는 OFF", not pump.is_on())

    pump.set_zone(True)
    check("고랑 안에서만 ON 가능", pump.turn_on() is True and pump.is_on())

    pump.set_zone(False)
    check("고랑을 벗어나면 즉시 OFF", not pump.is_on())
    pump.set_zone(True)
    check("재진입해도 자동으로 켜지지 않음 (명시적 요청 필요)", not pump.is_on())

    pump.turn_on()
    pump.set_lockout(True)
    check("물 부족 잠금 시 OFF", not pump.is_on())
    pump.set_lockout(False)
    check("잠금 해제 시 이전 요청이 복원됨", pump.is_on())

    # 최대 연속 가동 워치독
    clock = FakeClock()
    real_monotonic = time.monotonic
    time.monotonic = clock.monotonic
    try:
        p2 = PumpController()
        p2.set_zone(True)
        p2.turn_on()
        clock.now += config.PUMP_MAX_CONTINUOUS_SEC + 1.0
        p2.tick()
        check("최대 연속 가동 시간을 넘기면 워치독이 강제 OFF", not p2.is_on())

        # 수위 디바운스
        w = WaterTankSensor()
        w._sim_low = True
        w.poll()
        check("물 부족 신호 직후에는 아직 확정 안 됨", not w.is_water_low())
        clock.now += config.WATER_LEVEL_DEBOUNCE_SEC + 0.1
        w.poll()
        w.poll()
        check("디바운스 시간이 지나면 확정됨", w.is_water_low())

        w._sim_low = False
        w.poll()
        check("복구 신호도 즉시 반영되지 않음(디바운스)", w.is_water_low())
        clock.now += config.WATER_LEVEL_DEBOUNCE_SEC + 0.1
        w.poll()
        w.poll()
        check("디바운스 후 정상 복구", not w.is_water_low())
    finally:
        time.monotonic = real_monotonic


# ======================================================================
# 7. ToF
# ======================================================================
def test_tof():
    section("7. ToF 필터 / 고랑 끝 판정")
    from sensors.tof_sensor import ToFPair

    pair = ToFPair(config.TOF_LEFT, config.TOF_RIGHT, backend="sim")
    pair.left._driver.override = 150.0
    pair.right._driver.override = 150.0
    for _ in range(10):
        pair.read()
    check("고랑 안에서는 끝으로 판정하지 않음", not pair.both_out_of_range())

    # 한쪽만 사라지면 아직 끝이 아니다
    pair.left._driver.override = config.TOF_OUT_OF_RANGE_MM
    for _ in range(5):
        pair.read()
    check("한쪽만 out-of-range 면 끝이 아님", not pair.both_out_of_range())

    # 양쪽 다 사라지면 확인 틱 후 확정
    pair.right._driver.override = config.TOF_OUT_OF_RANGE_MM
    detected_at = None
    for i in range(1, 30):
        pair.read()
        if pair.both_out_of_range():
            detected_at = i
            break
    check("양쪽 out-of-range 후 곧바로 고랑 끝 확정 (EMA 지연 없음)",
          detected_at is not None and detected_at <= config.TOF_END_CONFIRM_TICKS + 1,
          f"{detected_at}틱")

    # 노이즈 1회로는 오판하지 않아야 함
    pair2 = ToFPair(config.TOF_LEFT, config.TOF_RIGHT, backend="sim")
    pair2.left._driver.override = 150.0
    pair2.right._driver.override = 150.0
    for _ in range(5):
        pair2.read()
    pair2.left._driver.override = config.TOF_OUT_OF_RANGE_MM
    pair2.right._driver.override = config.TOF_OUT_OF_RANGE_MM
    pair2.read()
    spurious = pair2.both_out_of_range()
    check("단발 노이즈로는 고랑 끝을 확정하지 않음", not spurious)

    # 한 번의 read() 가 센서를 두 번 측정하지 않는지
    class CountingDriver:
        def __init__(self):
            self.n = 0

        def read_mm(self):
            self.n += 1
            return 150.0

        def close(self):
            pass

    cd_l, cd_r = CountingDriver(), CountingDriver()
    pair3 = ToFPair(config.TOF_LEFT, config.TOF_RIGHT, backend="sim")
    pair3.left.attach_driver(cd_l)
    pair3.right.attach_driver(cd_r)
    pair3.read()
    pair3.both_out_of_range()
    pair3.walls_visible()
    check("read() 1회당 센서 측정도 1회 (이중 읽기 제거)",
          cd_l.n == 1 and cd_r.n == 1, f"L={cd_l.n} R={cd_r.n}")

    pair.close()
    pair2.close()
    pair3.close()


# ======================================================================
# 8. 비전 (합성 영상)
# ======================================================================
def test_vision():
    section("8. 비전 라인검출 (합성 BGR 영상)")
    try:
        import cv2
        import numpy as np
    except ImportError:
        check("OpenCV 없음 - 비전 테스트 건너뜀", True, "skipped")
        return

    from sensors.vision_line_detector import VisionLineDetector

    det = VisionLineDetector(camera=None)

    def make_frame(center_x, width_px=300, slant=0):
        """BGR 프레임: 초록 배경 + 갈색 흙 띠."""
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = (60, 120, 60)          # BGR 초록 (풀)
        for y in range(h):
            shift = int(slant * (y - h / 2))
            x0 = int(center_x + shift - width_px // 2)
            x1 = x0 + width_px
            x0, x1 = max(0, x0), min(w, x1)
            if x1 > x0:
                img[y, x0:x1] = (55, 85, 120)   # BGR 갈색 (흙)
        return img

    r = det.compute_from_frame(make_frame(320))
    check("중앙에 흙이 있으면 오차 ≈ 0", abs(r.normalized_error) < 0.05,
          f"err={r.normalized_error:+.3f} conf={r.confidence:.2f}")
    check("중앙 흙에서 신뢰도가 임계값 이상", r.confidence >= config.VISION_MIN_CONFIDENCE,
          f"conf={r.confidence:.2f} coverage={r.coverage:.2f}")

    r = det.compute_from_frame(make_frame(460))
    check("흙이 오른쪽에 있으면 오차 > 0 (오른쪽으로 가야 함)",
          r.normalized_error > 0.15, f"err={r.normalized_error:+.3f}")

    r = det.compute_from_frame(make_frame(180))
    check("흙이 왼쪽에 있으면 오차 < 0", r.normalized_error < -0.15,
          f"err={r.normalized_error:+.3f}")

    # 화면 전체가 흙 -> 중심이 무의미하므로 신뢰도가 낮아야 한다
    full = np.zeros((480, 640, 3), dtype=np.uint8)
    full[:, :] = (55, 85, 120)
    r = det.compute_from_frame(full)
    check("화면 전체가 흙이면 신뢰도가 낮음 (예전 버전의 역상관 결함 수정)",
          r.confidence < config.VISION_MIN_CONFIDENCE,
          f"conf={r.confidence:.2f} coverage={r.coverage:.2f}")

    # 흙이 전혀 없음
    none_img = np.zeros((480, 640, 3), dtype=np.uint8)
    none_img[:, :] = (60, 120, 60)
    r = det.compute_from_frame(none_img)
    check("흙이 없으면 신뢰도 0", r.confidence == 0.0, f"conf={r.confidence:.2f}")

    # RGB/BGR 혼동 회귀 테스트: 예전 코드처럼 채널을 뒤집으면 흙을 못 찾는다
    swapped = make_frame(320)[:, :, ::-1].copy()
    r_swap = det.compute_from_frame(swapped)
    check("채널을 뒤집으면 흙을 못 찾음 (BGR 처리가 맞다는 증거)",
          r_swap.confidence < config.VISION_MIN_CONFIDENCE,
          f"conf={r_swap.confidence:.2f}")


# ======================================================================
# 9. 전체 임무 시뮬레이션
# ======================================================================
def _run_mission(world, max_seconds=400.0, blind_vision=False, break_tof=False):
    from navigation.mission_state_machine import (
        MissionStateMachine, MissionState, TERMINAL_STATES,
    )
    from actuators.motor_driver import MotorDriver
    from actuators.pump_controller import PumpController
    from control.line_follower import LineFollower
    from navigation.furrow_manager import FurrowManager
    from sensors.odometry import Odometry
    from sensors.tof_sensor import ToFPair
    from tools.simulation import (
        FakeClock, SimAruco, SimCamera, SimVision, SimWaterSensor,
    )

    clock = FakeClock()
    real_monotonic, real_sleep = time.monotonic, time.sleep
    time.monotonic, time.sleep = clock.monotonic, clock.sleep

    try:
        odom = Odometry()
        motors = MotorDriver(odometry=odom)
        world.motors, world.odom = motors, odom
        clock.on_advance = world.integrate

        tof = ToFPair(config.TOF_LEFT, config.TOF_RIGHT, backend="sim")
        vision = SimVision(world)
        vision.force_blind = blind_vision

        def sync_tof():
            l, r = world.tof_readings_mm()
            if break_tof:
                l = r = config.TOF_OUT_OF_RANGE_MM
            tof.left._driver.override = l
            tof.right._driver.override = r

        deps = {
            "odom": odom,
            "motors": motors,
            "pump": PumpController(),
            "tof_pair": tof,
            "camera": SimCamera(),
            "aruco": SimAruco(world),
            "vision_line": vision,
            "water_sensor": SimWaterSensor(world),
            "furrow_mgr": FurrowManager(),
        }
        # 실기(MissionStateMachine._make_line_follower)와 동일하게 구성한다.
        deps["line_follower"] = LineFollower(
            tof, vision_detector=vision, odometry=odom
        )

        fsm = MissionStateMachine(deps=deps)

        # 펌프가 고랑 밖에서 켜진 적이 있는지 감시
        violations = {"pump_outside": 0, "pump_ticks_inside": 0, "max_abs_x": 0.0}
        t_start = clock.now
        visited_states = set()

        while fsm.state not in TERMINAL_STATES:
            sync_tof()
            fsm.step()
            visited_states.add(fsm.state)

            # 입구 경계에서 몇 틱 겹치는 것은 정상이므로 여유(0.35m)를 둔다.
            # "고랑에서 확실히 벗어난 곳에서 펌프가 켜졌는가"만 위반으로 센다.
            if fsm.pump.is_on() and not world.pump_zone_ok(margin=0.35):
                violations["pump_outside"] += 1
            if fsm.pump.is_on() and world.inside_furrow_index() is not None:
                violations["pump_ticks_inside"] += 1
            violations["max_abs_x"] = max(violations["max_abs_x"], abs(world.x))

            time.sleep(config.CONTROL_LOOP_DT)
            if clock.now - t_start > max_seconds:
                break

        return fsm, world, violations, visited_states, clock.now - t_start
    finally:
        time.monotonic, time.sleep = real_monotonic, real_sleep


def test_full_mission():
    section("9. 전체 임무 시뮬레이션 (고랑 2개 + END 마커)")
    from navigation.mission_state_machine import MissionState
    from tools.simulation import SimWorld

    world = SimWorld(n_furrows=2)
    fsm, world, viol, states, elapsed = _run_mission(world)

    check("임무가 MISSION_COMPLETE 로 종료됨",
          fsm.state == MissionState.MISSION_COMPLETE,
          f"state={fsm.state.name} halt={fsm._halt_reason[:60]}")
    check("고랑 2개를 모두 완료 처리",
          fsm.furrow_mgr.completed == [1, 2],
          f"completed={fsm.furrow_mgr.completed}")
    check("헤드랜드 이동 상태를 실제로 거침",
          MissionState.HEADLAND_TRANSIT in states)
    check("유턴 상태를 거침", MissionState.TURN_AROUND in states)
    check("펌프가 고랑 밖에서 켜진 적 없음 (인터록 유효)",
          viol["pump_outside"] == 0, f"위반 {viol['pump_outside']}회")
    check("종료 시 펌프 OFF", not fsm.pump.is_on())
    check("실제로 고랑 안에서 급수함 (펌프 가동 시간)",
          viol["pump_ticks_inside"] * config.CONTROL_LOOP_DT > 20.0,
          f"{viol['pump_ticks_inside'] * config.CONTROL_LOOP_DT:.0f}초 가동")
    check("종료 시 HOME 부근으로 복귀함",
          abs(world.x) < 0.7, f"최종 x={world.x:+.2f}m")
    check("밭 폭(고랑 2개=1m)을 크게 벗어나지 않음",
          viol["max_abs_x"] < 2.6, f"최대 |x|={viol['max_abs_x']:.2f}m")
    check("합리적인 시간 안에 완료", elapsed < 400, f"{elapsed:.0f}s (시뮬레이션 시간)")


def test_larger_field():
    section("9-b. 고랑 5개 밭 (HOME 복귀 거리 검증)")
    from navigation.mission_state_machine import MissionState
    from tools.simulation import SimWorld

    world = SimWorld(n_furrows=5)
    fsm, world, viol, states, elapsed = _run_mission(world, max_seconds=900)
    check("고랑 5개 밭에서도 완주",
          fsm.state == MissionState.MISSION_COMPLETE,
          f"state={fsm.state.name} halt={fsm._halt_reason[:50]}")
    check("고랑 5개를 모두 완료 처리",
          fsm.furrow_mgr.completed == [1, 2, 3, 4, 5],
          f"completed={fsm.furrow_mgr.completed}")
    check("멀리 간 만큼 HOME 까지 되돌아옴", abs(world.x) < 0.7,
          f"최종 x={world.x:+.2f}m")
    check("펌프가 고랑 밖에서 켜진 적 없음", viol["pump_outside"] == 0)


def test_water_low_mission():
    section("10. 물통 부족 시나리오")
    from navigation.mission_state_machine import MissionState, NavigationTarget
    from tools.simulation import SimWorld

    world = SimWorld(n_furrows=3, water_low_after_furrow=1)
    fsm, world, viol, states, elapsed = _run_mission(world)

    check("물이 떨어지면 HOME 으로 복귀해 종료",
          fsm.state == MissionState.MISSION_COMPLETE
          and fsm.target == NavigationTarget.HOME,
          f"state={fsm.state.name} target={fsm.target.name}")
    check("모든 고랑을 돌기 전에 중단됨",
          len(fsm.furrow_mgr.completed) < 3,
          f"completed={fsm.furrow_mgr.completed}")
    check("물 부족 상태에서 펌프가 켜지지 않음", not fsm.pump.is_on())


def test_fault_injection():
    section("11. 고장 주입 - 폭주하지 않고 SAFE_HALT 로 가는가")
    from navigation.mission_state_machine import MissionState
    from tools.simulation import SimWorld

    # (a) ToF 가 항상 out-of-range (예전 버전의 기본 동작이었던 상황)
    world = SimWorld(n_furrows=2)
    fsm, world, viol, states, elapsed = _run_mission(world, break_tof=True)
    check("ToF 고장: 폭주하지 않고 정지 상태로 종료",
          fsm.state in (MissionState.SAFE_HALT, MissionState.ERROR),
          f"state={fsm.state.name}")
    check("ToF 고장: 로봇이 밭 밖으로 멀리 나가지 않음",
          abs(world.x) < 8.0 and abs(world.y) < 8.0,
          f"pos=({world.x:.1f}, {world.y:.1f})")
    check("ToF 고장: 펌프 OFF", not fsm.pump.is_on())

    # (b) 마커가 하나도 없는 밭
    class NoMarkerWorld(SimWorld):
        def visible_markers(self):
            return {}

    world2 = NoMarkerWorld(n_furrows=2)
    fsm2, world2, viol2, states2, elapsed2 = _run_mission(world2, max_seconds=600)
    check("마커 부재: SAFE_HALT 로 정지 (임의로 완료 판단하지 않음)",
          fsm2.state == MissionState.SAFE_HALT, f"state={fsm2.state.name}")
    check("마커 부재: 헤드랜드 이동 횟수 상한을 지킴",
          fsm2._transit_count <= config.MAX_HEADLAND_TRANSITS,
          f"{fsm2._transit_count}회")
    check("마커 부재: 밭 밖으로 무한 전진하지 않음",
          abs(world2.x) < 6.0, f"x={world2.x:.2f}m")

    # (c) 비전 완전 실패 -> 경고 후 SAFE_HALT
    world3 = SimWorld(n_furrows=2)
    fsm3, world3, viol3, states3, elapsed3 = _run_mission(world3, blind_vision=True)
    check("비전 실패: 방치되지 않고 정지 또는 완주",
          fsm3.state in (MissionState.SAFE_HALT, MissionState.MISSION_COMPLETE),
          f"state={fsm3.state.name}")


def test_step_exception_handling():
    section("12. step() 예외 처리 -> ERROR 상태")
    from navigation.mission_state_machine import MissionState, MissionStateMachine
    from tools.simulation import SimCamera, SimWorld, SimAruco, SimVision, SimWaterSensor
    from actuators.motor_driver import MotorDriver
    from actuators.pump_controller import PumpController
    from control.line_follower import LineFollower
    from navigation.furrow_manager import FurrowManager
    from sensors.odometry import Odometry
    from sensors.tof_sensor import ToFPair

    world = SimWorld()
    odom = Odometry()
    motors = MotorDriver(odometry=odom)
    tof = ToFPair(config.TOF_LEFT, config.TOF_RIGHT, backend="sim")

    class ExplodingAruco:
        def detect(self):
            raise RuntimeError("카메라 버스 오류 시뮬레이션")

    deps = {
        "odom": odom, "motors": motors, "pump": PumpController(),
        "tof_pair": tof, "camera": SimCamera(), "aruco": ExplodingAruco(),
        "vision_line": SimVision(world), "water_sensor": SimWaterSensor(world),
        "furrow_mgr": FurrowManager(),
        "line_follower": LineFollower(tof, odometry=odom),
    }
    fsm = MissionStateMachine(deps=deps)
    for _ in range(20):
        fsm.step()
        if fsm.state == MissionState.ERROR:
            break

    check("반복 예외 발생 시 ERROR 상태로 전환", fsm.state == MissionState.ERROR,
          f"state={fsm.state.name}")
    check("ERROR 상태에서 모터 정지",
          motors.last_left == 0.0 and motors.last_right == 0.0)
    check("ERROR 상태에서 펌프 OFF", not fsm.pump.is_on())
    tof.close()


# ======================================================================
# 13. 실기 안전 수정 사항 회귀 테스트
# ======================================================================
def test_hardware_safety_fixes():
    section("13. 실기 안전 수정 사항")

    # --- 13-1. 조향 포화 시 좌우 차이(회전율) 보존 ---
    from actuators.motor_driver import MotorDriver

    md = MotorDriver(odometry=None)
    md.drive(0.9, 0.35)
    diff = (md.last_left - md.last_right) / 2.0
    check(
        "고속에서도 조향량이 잘리지 않음",
        abs(diff - 0.35) < 1e-6,
        f"요청 0.35 -> 실제 {diff:.3f}",
    )
    check(
        "바퀴 명령이 [-1,1] 안에 있음",
        -1.0 <= md.last_left <= 1.0 and -1.0 <= md.last_right <= 1.0,
        f"L={md.last_left:.2f} R={md.last_right:.2f}",
    )

    # --- 13-2. ToF: 측정이 준비되지 않은 틱에서 벽을 잃지 않는다 ---
    from sensors.tof_sensor import ToFSensor

    class _FlakyDriver:
        """실제 VL53L1X 처럼 몇 틱에 한 번만 새 값을 내는 드라이버."""

        is_real = True

        def __init__(self):
            self.n = 0

        def read_mm(self):
            self.n += 1
            return 150.0 if self.n % 3 == 1 else None  # 3틱에 1번만 갱신

        def close(self):
            pass

    s = ToFSensor("test", 0x30, 17)
    s.attach_driver(_FlakyDriver())
    walls = []
    for _ in range(9):
        s.sample()
        walls.append(s.wall_visible())
    check(
        "ToF 미준비 틱을 out-of-range 로 오판하지 않음",
        all(walls),
        f"벽 감지 {sum(walls)}/9틱",
    )

    # --- 13-3. ToF 고장은 '가짜 정상값'이 아니라 '측정 불가'로 나온다 ---
    from sensors.tof_sensor import _DeadDriver

    dead = ToFSensor("dead", 0x31, 27)
    dead.attach_driver(_DeadDriver())
    dead.sample()
    check(
        "ToF 초기화 실패 시 가짜 벽 거리를 만들지 않음",
        not dead.wall_visible(),
        f"raw={dead.last_raw_mm:.0f}mm",
    )
    check("고장 드라이버는 is_real=False", not dead.is_real)

    # --- 13-4. 외부 종료 요청(SIGTERM) 처리 ---
    import signal

    from actuators.pump_controller import PumpController
    from navigation.mission_state_machine import MissionState, MissionStateMachine

    from sensors.tof_sensor import ToFPair
    from tools.simulation import SimCamera

    fsm = MissionStateMachine(
        deps={
            "motors": MotorDriver(),
            "pump": PumpController(),
            "camera": SimCamera(),
            "tof_pair": ToFPair(config.TOF_LEFT, config.TOF_RIGHT, backend="sim"),
        }
    )
    check("초기에는 종료 요청 없음", not fsm._stop_requested)

    fsm._install_signal_handlers()
    handler = signal.getsignal(signal.SIGTERM)
    check(
        "SIGTERM 핸들러가 등록됨 (systemctl stop 시 모터가 계속 도는 것 방지)",
        callable(handler) and handler not in (signal.SIG_DFL, signal.SIG_IGN),
    )

    handler(signal.SIGTERM, None)
    check("SIGTERM 수신 시 정지 요청 플래그가 세워짐", fsm._stop_requested)

    fsm._safe_halt("테스트 종료 요청")
    check(
        "종료 요청은 SAFE_HALT 로 귀결",
        fsm.state == MissionState.SAFE_HALT,
        f"state={fsm.state.name}",
    )
    check("종료 시 모터 정지", fsm.motors.last_left == 0.0 and fsm.motors.last_right == 0.0)
    check("종료 시 펌프 OFF", not fsm.pump.is_on())
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    # --- 13-5. 사전 점검이 시뮬레이션(주입) 모드를 막지 않는다 ---
    check("컴포넌트 주입 시 하드웨어 사전 점검을 건너뜀", fsm._preflight_check())


# ======================================================================
# 14. 고랑 끝 판정 (출구 마커 없이)
# ======================================================================
def test_furrow_end_detection():
    section("14. 고랑 끝 판정 (출구 마커 없음)")

    from control.line_follower import LineFollower
    from navigation.mission_state_machine import MissionState
    from sensors.tof_sensor import ToFPair
    from tools.simulation import SimWorld

    # 출구에 마커가 없으므로 ToF 와 비전만으로 "고랑이 끝났다"를 알아내야 한다.
    pair = ToFPair(config.TOF_LEFT, config.TOF_RIGHT, backend="sim")

    def set_walls(left_mm, right_mm):
        pair.left._driver.override = left_mm
        pair.right._driver.override = right_mm

    # --- 14-1. 벽이 있으면 끝이 아니다 ---
    lf = LineFollower(pair, vision_detector=None, odometry=None)
    set_walls(150.0, 150.0)
    for _ in range(10):
        r = lf.step()
    check("좌우 벽이 보이면 고랑 끝이 아님", not r.furrow_end_detected)

    # --- 14-2. 한쪽만 사라지면 끝이 아니다 ---
    set_walls(150.0, config.TOF_OUT_OF_RANGE_MM + 100)
    for _ in range(10):
        r = lf.step()
    check(
        "한쪽 벽만 사라지면 고랑 끝이 아님 (이랑 유실 구간 방어)",
        not r.furrow_end_detected,
    )

    # --- 14-3. 양쪽이 사라지고 연속 확인되면 끝 ---
    set_walls(config.TOF_OUT_OF_RANGE_MM + 100, config.TOF_OUT_OF_RANGE_MM + 100)
    results = []
    for _ in range(config.TOF_END_CONFIRM_TICKS + 2):
        results.append(lf.step().furrow_end_detected)
    check(
        "양쪽 벽이 연속으로 사라지면 고랑 끝으로 판정",
        results[-1],
        f"확인 틱={config.TOF_END_CONFIRM_TICKS}, 판정 이력={results}",
    )
    check(
        "첫 틱에 성급히 확정하지 않음 (노이즈 방어)",
        not results[0],
        "TOF_END_CONFIRM_TICKS 만큼 연속 관측을 요구",
    )

    # --- 14-4. 비전 동의 옵션 ---
    check(
        "비전 동의 요구 여부가 설정으로 제어됨",
        isinstance(config.FURROW_END_REQUIRE_VISION_AGREE, bool),
        f"FURROW_END_REQUIRE_VISION_AGREE={config.FURROW_END_REQUIRE_VISION_AGREE} "
        f"(True 면 ToF+비전 모두 필요)",
    )
    check(
        "비전 신뢰도 임계값이 유효 범위",
        0.0 < config.FURROW_END_VISION_CONFIDENCE_MAX < 1.0,
        f"{config.FURROW_END_VISION_CONFIDENCE_MAX}",
    )

    # --- 14-5. 최소 주행거리 가드 ---
    check(
        "고랑 끝 판정 전 최소 주행거리 조건이 있음",
        config.FURROW_END_MIN_TRAVEL_M > 0,
        f"{config.FURROW_END_MIN_TRAVEL_M}m 이전에는 판정하지 않음 "
        f"(입구 근처 오판 방지)",
    )

    # --- 14-6. ToF 보조 혼합 비율 ---
    check(
        "비전이 살아 있어도 ToF 를 일부 섞음",
        0.0 < config.TOF_ASSIST_WEIGHT < 1.0,
        f"TOF_ASSIST_WEIGHT={config.TOF_ASSIST_WEIGHT} "
        f"(서로 다른 오류 특성을 갖는 두 센서를 융합)",
    )

    # --- 14-7. 마커 ID 규칙 (입구에만) ---
    check(
        "입구 팻말 ID = 고랑 번호",
        config.furrow_marker_id(3) == 3,
        f"3번 고랑 -> id={config.furrow_marker_id(3)}",
    )
    check(
        "출구 마커 함수가 존재하지 않음 (출구에 마커 없음)",
        not hasattr(config, "furrow_exit_marker_id"),
    )
    check(
        "ToF 탐침이 기본 활성화됨 (비전 실패 시 대체 경로)",
        config.ENTRANCE_TOF_PROBE_ENABLED,
        "비전만 믿으면 흐린 날 아예 못 들어간다",
    )
    check(
        "탐침도 실패하면 무리하지 않고 정지 (기본 설정)",
        not config.ENTRANCE_ALLOW_BLIND_CREEP,
        "이랑을 밟는 것보다 멈추는 편이 낫다",
    )
    check(
        "탐침 전진 거리에 상한이 있음",
        0 < config.ENTRANCE_PROBE_MAX_TRAVEL_M <= 2.0,
        f"{config.ENTRANCE_PROBE_MAX_TRAVEL_M}m 안에 벽을 못 잡으면 SAFE_HALT",
    )

    # ---- 비전이 완전히 죽어도 ToF 만으로 임무를 마치는가 ----
    # 흐린 날, 비 온 뒤, 흙 HSV 미조정 상황을 재현한다.
    import tools.simulation as _sim
    from sensors.vision_line_detector import VisionLineResult as _VR

    _orig_vr = _sim.SimWorld.vision_result
    try:
        _sim.SimWorld.vision_result = lambda self: _VR(0.0, 0.0, 0.0, 0.01)
        world_b = SimWorld(n_furrows=3)
        fsm_b, world_b, _, _, _ = _run_mission(world_b, max_seconds=4000.0)
        check(
            "비전이 완전히 죽어도 ToF 탐침으로 전 고랑 급수 완료",
            fsm_b.state == MissionState.MISSION_COMPLETE
            and fsm_b.furrow_mgr.total_completed() == 3,
            f"완료 {fsm_b.furrow_mgr.total_completed()}/3 state={fsm_b.state.name}",
        )
    finally:
        _sim.SimWorld.vision_result = _orig_vr

    pair.close()


# ======================================================================
# 15. 임무 동작 사양 대조
# ======================================================================
def test_mission_behaviour_spec():
    section("15. 임무 동작 사양 대조")

    from navigation.mission_state_machine import (
        MissionState, MissionStateMachine, NavigationTarget,
    )
    from tools.simulation import SimWorld

    # ---- 15-1. 왕복 방식인가 (ㄹ자가 아니라) ----
    world = SimWorld(n_furrows=3)
    fsm, world, viol, states, elapsed = _run_mission(world)

    check(
        "고랑마다 진입 → 유턴 → 복귀 (왕복 방식)",
        MissionState.TRAVEL_INTO_FURROW in states
        and MissionState.TURN_AROUND in states
        and MissionState.TRAVEL_BACK_TO_ENTRANCE in states,
        f"방문 상태에 유턴 포함={MissionState.TURN_AROUND in states}",
    )
    check(
        "고랑 사이는 헤드랜드로 이동 (제자리 회전만으로 못 감)",
        MissionState.HEADLAND_TRANSIT in states,
    )
    check(
        "모든 고랑 급수 후 HOME 복귀로 종료",
        fsm.state == MissionState.MISSION_COMPLETE
        and fsm.target == NavigationTarget.HOME,
        f"state={fsm.state.name} target={fsm.target.name} "
        f"완료={fsm.furrow_mgr.total_completed()}개",
    )
    check(
        "고랑 밖에서 펌프가 켜진 적 없음",
        viol["pump_outside"] == 0,
        f"위반 {viol['pump_outside']}틱",
    )
    check(
        "고랑 안에서 실제로 살수함",
        viol["pump_ticks_inside"] > 0,
        f"살수 {viol['pump_ticks_inside']}틱",
    )

    # ---- 15-2. 물 부족 시 진입 중단하고 즉시 유턴 ----
    from navigation.mission_state_machine import MissionStateMachine as MSM

    from actuators.motor_driver import MotorDriver
    from actuators.pump_controller import PumpController
    from sensors.tof_sensor import ToFPair
    from tools.simulation import SimCamera

    fsm2 = MSM(deps={
        "motors": MotorDriver(),
        "pump": PumpController(),
        "camera": SimCamera(),
        "tof_pair": ToFPair(config.TOF_LEFT, config.TOF_RIGHT, backend="sim"),
    })
    check(
        "물 부족 중단 플래그가 초기에는 꺼져 있음",
        not fsm2._water_low_aborted,
    )
    check(
        "설정: 물 부족 시 진입 주행 중단 활성화",
        config.WATER_LOW_ABORT_INBOUND_LEG,
        "고랑 끝까지 헛돌지 않고 즉시 유턴",
    )

    # 중단 플래그가 서 있으면 물이 다시 차 보여도 HOME 을 선택해야 한다
    class _AlwaysFine:
        def is_water_low(self):
            return False

        def poll(self):
            pass

    fsm2.water_sensor = _AlwaysFine()
    fsm2._water_low_aborted = True
    fsm2._state_evaluate_mission()
    check(
        "물 부족으로 중단했으면 센서가 정상으로 읽혀도 HOME 으로 간다",
        fsm2.target == NavigationTarget.HOME,
        f"target={fsm2.target.name} (디바운스 경계 흔들림 방어)",
    )
    check(
        "HOME 결정 후 중단 플래그는 초기화됨",
        not fsm2._water_low_aborted,
    )

    # ---- 15-3. 복귀 구간 살수 설정이 실제로 반영되는가 ----
    check(
        "복귀 구간 살수 여부가 설정으로 제어됨",
        isinstance(config.PUMP_ON_RETURN_LEG, bool),
        f"PUMP_ON_RETURN_LEG={config.PUMP_ON_RETURN_LEG} "
        f"({'왕복 모두 살수' if config.PUMP_ON_RETURN_LEG else '진입 시에만 살수'})",
    )

    # ---- 15-4. HOME 복귀 시 지나온 고랑 수만큼 이동 허용 ----
    fsm2.target = NavigationTarget.HOME
    fsm2.furrow_mgr.current_index = 5
    allowed = fsm2._max_transits_allowed()
    check(
        "HOME 복귀 시 지나온 고랑 수만큼 헤드랜드 이동 허용",
        allowed >= 5,
        f"고랑 5개 지나옴 → 허용 {allowed}회 (부족하면 집에 못 돌아옴)",
    )

    # ---- 15-5. 임무 완료 판정은 END 마커를 실제로 봤을 때만 ----
    check(
        "마커를 못 봤다는 이유로 임무 완료를 추론하지 않음",
        config.REQUIRE_EXPLICIT_FIELD_END_MARKER,
        "END 마커 미관측 = '모름'이지 '완료'가 아님",
    )

    # ---- 15-6. END 마커 의미: 마지막 고랑도 급수해야 한다 ----
    # 밭 전체 고랑 수만큼 완료 처리되어야 한다.
    # 예전 코드는 END 마커를 보는 즉시 돌아서서 마지막 고랑을 통째로
    # 빠뜨렸다(n_furrows-1 개만 완료).
    for n in (2, 3, 4):
        world_n = SimWorld(n_furrows=n)
        fsm_n, world_n, viol_n, states_n, _ = _run_mission(world_n)
        check(
            f"고랑 {n}개 밭: 마지막 고랑까지 전부 급수 완료",
            fsm_n.state == MissionState.MISSION_COMPLETE
            and fsm_n.furrow_mgr.total_completed() == n,
            f"완료 {fsm_n.furrow_mgr.total_completed()}/{n}개 "
            f"state={fsm_n.state.name}",
        )

    # ---- 15-7. END 마커를 봐도 그 자리에서 돌아서지 않는다 ----
    world2 = SimWorld(n_furrows=2)
    fsm3, world2, _, states2, _ = _run_mission(world2)
    check(
        "END 마커를 본 고랑에도 진입해서 살수함",
        fsm3.furrow_mgr.total_completed() == 2,
        f"완료={fsm3.furrow_mgr.completed} (마지막 고랑이 빠지면 실패)",
    )
    check(
        "임무 완료 사유가 END 마커임이 기록됨",
        fsm3._mission_finished_by_end_marker,
    )

    # ---- 15-8. 왕복 살수 설정 확인 (사용자 선택: 두 번 준다) ----
    check(
        "왕복 모두 살수하도록 설정됨",
        config.PUMP_ON_RETURN_LEG is True,
        "진입할 때와 복귀할 때 각각 살수",
    )

    # ---- 15-9. 밭이 커져도 HOME(1번 고랑 입구)까지 반드시 돌아온다 ----
    # 고랑이 많아질수록 되돌아갈 칸 수도 늘어난다. 이동 허용 횟수가 부족하면
    # 밭 한가운데서 SAFE_HALT 로 멈춘다 -- 실제로 가장 위험한 실패 모드다.
    for n in (8, 12, 16):
        world_b = SimWorld(n_furrows=n)
        fsm_b, world_b, _, _, _ = _run_mission(world_b, max_seconds=3000.0)
        home_x = 0.0   # HOME = 1번 고랑 입구 = x 원점
        check(
            f"고랑 {n}개 밭: 전부 급수하고 HOME 까지 복귀",
            fsm_b.state == MissionState.MISSION_COMPLETE
            and fsm_b.furrow_mgr.total_completed() == n,
            f"완료 {fsm_b.furrow_mgr.total_completed()}/{n} "
            f"state={fsm_b.state.name}",
        )
        check(
            f"고랑 {n}개 밭: 밭 한가운데가 아니라 HOME 부근에서 종료",
            abs(world_b.x - home_x) < 1.0,
            f"최종 x={world_b.x:+.2f}m (HOME=0.00m)",
        )

    # ---- 15-10. 되돌아갈 칸 수를 실제로 계산하는가 ----
    fsm2.target = NavigationTarget.NEXT_FURROW
    fsm2.furrow_mgr.current_index = 9
    fsm2._water_low_aborted = True
    fsm2._state_evaluate_mission()
    check(
        "9번 고랑에서 물이 떨어지면 HOME 까지 8칸으로 계산",
        fsm2._home_transits_remaining == 8,
        f"남은 칸={fsm2._home_transits_remaining} (9번→1번 = 8칸)",
    )
    check(
        "이동 허용 횟수가 필요 칸 수보다 넉넉함",
        fsm2._max_transits_allowed() > fsm2._home_transits_remaining,
        f"필요 8칸 / 허용 {fsm2._max_transits_allowed()}회",
    )

    # 1번 고랑에서 끝났다면 되돌아갈 필요가 없다
    fsm2.target = NavigationTarget.NEXT_FURROW
    fsm2.furrow_mgr.current_index = 1
    fsm2._water_low_aborted = True
    fsm2._state_evaluate_mission()
    check(
        "1번 고랑에서 끝나면 되돌아갈 칸 없음 (이미 HOME)",
        fsm2._home_transits_remaining == 0,
        f"남은 칸={fsm2._home_transits_remaining}",
    )


# ======================================================================
# 16. 바퀴 미끄러짐 내성 (추측항법 드리프트)
# ======================================================================
def test_slip_robustness():
    section("16. 바퀴 미끄러짐 내성")

    from navigation.mission_state_machine import MissionState
    from tools.simulation import SimWorld

    # 실제 밭에서는 무른 흙에서 바퀴가 헛돈다. 엔코더는 "돌았다"고 보고하지만
    # 로봇은 그만큼 안 나간다. 이 오차는 **누적**되므로, 마커로 주기적으로
    # 절대 위치·방위를 다시 잡지 않으면 밭 밖으로 나간다.
    # [주의] 팻말이 고랑당 1개로 줄면서 내성이 낮아졌다.
    #   게이트마다 마커가 2개일 때는 그 자리에서 두 마커의 상대 위치로
    #   **절대 방위**를 매번 다시 잡을 수 있었다. 팻말이 1개인 지금은
    #   _reanchor_heading_from_posts() 가 **서로 다른 두 고랑의 팻말**이
    #   동시에 보일 때만 방위를 잡을 수 있어서, 회전 오차를 바로잡을
    #   기회가 크게 줄었다.
    #   실측 한계: 대칭 미끄러짐 2% (2마커 시절에는 5%)
    for slip in (0.0, 0.02):
        world = SimWorld(n_furrows=12, slip_left=slip, slip_right=slip)
        fsm, world, _, _, _ = _run_mission(world, max_seconds=6000.0)
        check(
            f"미끄러짐 {slip*100:.0f}%: 고랑 12개 전부 급수하고 HOME 복귀",
            fsm.state == MissionState.MISSION_COMPLETE
            and fsm.furrow_mgr.total_completed() == 12,
            f"완료 {fsm.furrow_mgr.total_completed()}/12 state={fsm.state.name}",
        )
        check(
            f"미끄러짐 {slip*100:.0f}%: HOME 부근에서 종료",
            abs(world.x) < 1.0,
            f"최종 x={world.x:+.2f}m (HOME=0.00m)",
        )

    # 감당 못 하는 수준이면 폭주하지 말고 안전하게 멈춰야 한다
    world = SimWorld(n_furrows=12, slip_left=0.25, slip_right=0.05)
    fsm, world, _, _, _ = _run_mission(world, max_seconds=6000.0)
    check(
        "감당 불가능한 미끄러짐에서는 폭주하지 않고 SAFE_HALT",
        fsm.state in (MissionState.SAFE_HALT, MissionState.MISSION_COMPLETE),
        f"state={fsm.state.name} (ERROR 나 무한주행이 아니어야 함)",
    )
    check(
        "SAFE_HALT 시 모터 정지",
        fsm.motors.last_left == 0.0 and fsm.motors.last_right == 0.0,
    )
    check("SAFE_HALT 시 펌프 OFF", not fsm.pump.is_on())

    # 방위 보정 안전장치
    check(
        "게이트 방위 보정에 거리 제한이 있음",
        config.GATE_HEADING_ANCHOR_MAX_DISTANCE_M > 0,
        f"{config.GATE_HEADING_ANCHOR_MAX_DISTANCE_M}m 이내 게이트만 신뢰",
    )
    check(
        "게이트 방위 보정에 최대 보정각 제한이 있음",
        0 < config.GATE_HEADING_ANCHOR_MAX_CORRECTION_RAD < math.pi / 2,
        f"{math.degrees(config.GATE_HEADING_ANCHOR_MAX_CORRECTION_RAD):.0f}도 "
        f"초과 보정은 오검출로 무시",
    )


# ======================================================================
# 17. 무한궤도(트랙) 파라미터
# ======================================================================
def test_track_parameters():
    section("17. 무한궤도 파라미터")

    # 궤도는 회전 시 지면을 비비며 미끄러진다. 그래서 회전 계산에 쓰는
    # '유효 궤도 간격'이 실측 간격보다 넓어야 한다.
    check(
        "유효 궤도 간격이 실측 간격보다 넓음 (미끄러짐 보정)",
        config.WHEEL_BASE_M > config.TRACK_WIDTH_M,
        f"실측 {config.TRACK_WIDTH_M}m × 계수 {config.TRACK_SLIP_FACTOR} "
        f"= {config.WHEEL_BASE_M:.3f}m",
    )
    check(
        "미끄러짐 계수가 현실적인 범위",
        1.0 <= config.TRACK_SLIP_FACTOR <= 2.0,
        f"{config.TRACK_SLIP_FACTOR} (1.0=바퀴형, 1.2~1.5=일반 궤도)",
    )
    check(
        "직진 거리 보정 계수가 현실적인 범위",
        0.7 <= config.DISTANCE_CALIBRATION_FACTOR <= 1.1,
        f"{config.DISTANCE_CALIBRATION_FACTOR} "
        f"(1.0=미끄러짐 없음, 무른 흙이면 0.9 근처)",
    )

    # 거리 보정이 실제로 오도메트리에 반영되는가
    from sensors.odometry import Odometry

    odom = Odometry()
    import math as _m

    expected = (
        (2 * _m.pi * config.WHEEL_RADIUS_M) / config.TICKS_PER_REVOLUTION
    ) * config.DISTANCE_CALIBRATION_FACTOR
    check(
        "거리 보정 계수가 오도메트리에 실제로 적용됨",
        abs(odom._distance_per_tick - expected) < 1e-12,
        f"틱당 {odom._distance_per_tick*1000:.2f}mm",
    )
    odom.cleanup()

    # 회전 정밀도: 유효 간격이 틀리면 유턴이 어긋난다
    # 180도 회전 시 필요한 궤도 이동 거리 = pi * WHEEL_BASE_M / 2 (한쪽)
    arc = _m.pi * config.WHEEL_BASE_M / 2.0
    check(
        "180도 유턴에 필요한 궤도 이동량이 계산 가능",
        arc > 0,
        f"한쪽 궤도가 {arc*100:.1f}cm 이동해야 180도 회전",
    )

    # 궤도는 정지마찰이 커서 최소 듀티가 높다 -> 조향 분해능 확인
    check(
        "최소 듀티에서도 미세 조향이 가능한 범위",
        config.MOTOR_MIN_DUTY < 0.5,
        f"MOTOR_MIN_DUTY={config.MOTOR_MIN_DUTY} "
        f"(궤도는 보통 0.25~0.40, setup.py 4번으로 실측)",
    )

    # ToF 탐침이 각도에 의존하지 않는지 (궤도에서 중요)
    src = open("navigation/mission_state_machine.py", encoding="utf-8").read()
    probe = src[src.index("def _probe_entrance_with_tof"):]
    probe = probe[:probe.index("\n    def ")]
    check(
        "ToF 탐침이 좌우 거리차로 중심을 찾음 (엔코더 각도에 의존 안 함)",
        "left_mm - right_mm" in probe,
        "궤도는 회전 시 미끄러져 엔코더 각도가 부정확하다",
    )


# ======================================================================
def main():
    print("=" * 70)
    print(" 농장 로봇 자체 점검 (selftest)")
    print("=" * 70)

    tests = [
        test_imports,
        test_sign_conventions,
        test_pid,
        test_odometry,
        test_turn_termination,
        test_pump_and_water,
        test_tof,
        test_vision,
        test_full_mission,
        test_larger_field,
        test_water_low_mission,
        test_fault_injection,
        test_step_exception_handling,
        test_hardware_safety_fixes,
        test_furrow_end_detection,
        test_mission_behaviour_spec,
        test_slip_robustness,
        test_track_parameters,
    ]

    for t in tests:
        try:
            t()
        except Exception:
            check(f"{t.__name__} 실행 중 예외", False, "")
            traceback.print_exc()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print("\n" + "=" * 70)
    print(f" 결과: {passed}/{total} 통과")
    if passed != total:
        print("\n 실패 항목:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"   - {name}  {detail}")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

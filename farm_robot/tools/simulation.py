# -*- coding: utf-8 -*-
"""
tools/simulation.py
--------------------
하드웨어 없이 FSM 전체를 돌려보기 위한 간단한 밭/로봇 시뮬레이터.

한계를 분명히 밝힘:
  이 시뮬레이터는 **상태 전이, 부호 규약, 워치독, 인터록** 을 검증하기 위한
  것이지 실제 주행 성능을 예측하지 않는다. 바퀴 미끄러짐, 흙의 요철, 조명
  변화, 마커 오검출, 모터 응답 지연은 모델링되어 있지 않다.
  즉 "여기서 통과 = 실기에서 잘 달린다" 가 아니라
     "여기서 실패 = 실기에서는 확실히 실패한다" 로 읽어야 한다.

좌표계 (config 의 규약과 동일)
  x = 밭을 가로지르는 방향(고랑이 늘어선 방향), y = 고랑이 뻗는 방향
  theta = CCW 양수, theta = pi/2 이면 +y(밭 안쪽)를 바라봄
"""

import math

from config import (
    MARKER_CONFIRM_FRAMES,
    TOF_NOMINAL_WALL_DISTANCE_MM,
    TOF_OUT_OF_RANGE_MM,
    WHEEL_BASE_M,
    WHEEL_RADIUS_M,
    TICKS_PER_REVOLUTION,
    furrow_marker_id,
    FIELD_END_MARKER_ID,
    MARKER_POST_LATERAL_OFFSET_M,
    MARKER_POST_TILT_DEG,
    HOME_MARKER_ID,
)
from sensors.aruco_detector import MarkerObservation
from sensors.vision_line_detector import VisionLineResult


# ======================================================================
class FakeClock:
    """time.monotonic / time.sleep 을 대체하는 가상 시계."""

    def __init__(self, start: float = 1000.0):
        self.now = start
        self.on_advance = None  # 시간이 흐를 때 호출할 콜백 (물리 적분)

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float):
        seconds = max(0.0, float(seconds))
        # 물리 적분이 너무 큰 스텝으로 튀지 않도록 잘라서 진행
        remaining = seconds
        max_step = 0.02
        while remaining > 1e-9:
            step = min(max_step, remaining)
            self.now += step
            if self.on_advance is not None:
                self.on_advance(step)
            remaining -= step


# ======================================================================
class SimWorld:
    """밭 배치 + 로봇 운동학."""

    # 물리 파라미터
    MAX_WHEEL_SPEED_MPS = 0.5      # 속도 명령 1.0 일 때의 바퀴 선속도
    FURROW_LENGTH_M = 3.0
    FURROW_SPACING_M = 1.0
    FURROW_HALF_WIDTH_M = TOF_NOMINAL_WALL_DISTANCE_MM / 1000.0
    ENTRANCE_MARGIN_M = 0.4        # 입구 앞 이 범위까지는 이랑 벽이 있다고 본다
    MARKER_RANGE_M = 3.0
    CAMERA_HFOV_RAD = math.radians(31.0)   # Pi Camera v2 수평화각 62도의 절반
    # 마커 면의 법선에서 이 각도 안쪽에서만 검출된다(실제 ArUco 특성).
    # 실측 기준 ArUco 는 약 70도 기울기까지 '검출'은 되고(자세 추정 정확도는
    # 50도 부근부터 떨어진다), 그 너머는 사각형이 무너져 인식되지 않는다.
    MARKER_FACE_HALF_ANGLE_RAD = math.radians(70.0)
    # 입구 앞 이 거리까지는 카메라가 고랑 안쪽을 내다볼 수 있다
    VISION_LOOKAHEAD_M = 1.2
    # 고랑 중심에서 좌우로 이만큼 안쪽이어야 그 고랑이 화면에 잡힌다
    # [참고] 이 값이 곧 **팻말 횡오프셋의 비전 쪽 한계**다.
    #   팻말을 이보다 더 옆에 박으면, 로봇이 팻말 앞에 도착해도 정렬할
    #   고랑이 화면에 없어서 SAFE_HALT 된다(실측: 0.65m 에서 실패).
    #   ToF 탐침 쪽 한계는 더 빡빡하다 — 고랑 폭(≈0.32m) 이내.
    #   자세한 표는 config.MARKER_POST_LATERAL_OFFSET_M 주석 참고.
    VISION_ENTRANCE_CAPTURE_M = 0.6

    def __init__(
        self,
        n_furrows: int = 2,
        water_low_after_furrow: int = None,
        slip_left: float = 0.0,
        slip_right: float = 0.0,
    ):
        self.n_furrows = n_furrows
        self.water_low_after_furrow = water_low_after_furrow
        # 바퀴 미끄러짐 비율 (0.0 = 이상적, 0.15 = 15% 헛돎)
        self.slip_left = slip_left
        self.slip_right = slip_right

        # 로봇 실제 자세 (ground truth)
        self.x = 0.0
        self.y = -1.0
        self.theta = math.pi / 2.0     # 밭 안쪽(+y)을 바라봄

        self.motors = None             # MotorDriver (명령 읽기용)
        self.odom = None               # Odometry (틱 주입용)

        self._tick_residual_l = 0.0
        self._tick_residual_r = 0.0
        self._dist_per_tick = (2 * math.pi * WHEEL_RADIUS_M) / TICKS_PER_REVOLUTION

        self.total_time = 0.0
        self.furrows_watered = 0
        self._was_inside = False

        self.markers = self._build_markers()

    # ------------------------------------------------------------------
    def _build_markers(self):
        """
        [수정] 마커는 **고랑 입구에만** 있다. 출구에는 없다.
          - 팻말은 고랑 중심선이 아니라 옆 이랑에 박혀 있다
            (중심선에 있으면 로봇이 진입하다 들이받는다)
          - END 쪽으로 MARKER_POST_TILT_DEG 만큼 기울여 박는다
            (정면 진입할 때도, 헤드랜드를 따라 복귀할 때도 보이도록)
          - 고랑 끝은 마커가 아니라 **ToF 벽 소실 + 비전**으로 판정한다
        """
        m = {}
        off = MARKER_POST_LATERAL_OFFSET_M
        facing = -math.pi / 2 + math.radians(MARKER_POST_TILT_DEG)

        for k in range(1, self.n_furrows + 1):
            cx = (k - 1) * self.FURROW_SPACING_M
            m[furrow_marker_id(k)] = (cx + off, 0.0, facing)

        # HOME = 1번 고랑 입구 팻말과 같은 자리
        m[HOME_MARKER_ID] = (off, 0.0, facing)

        # END 마커는 **마지막 고랑의 입구 팻말**에 함께 붙인다.
        last_cx = (self.n_furrows - 1) * self.FURROW_SPACING_M
        m[FIELD_END_MARKER_ID] = (last_cx + off, 0.0, facing)
        return m

    # ------------------------------------------------------------------
    def integrate(self, dt: float):
        """모터 명령을 읽어 로봇을 dt 만큼 움직이고 엔코더 틱을 주입한다."""
        if self.motors is None or dt <= 0:
            return
        self.total_time += dt

        # MotorDriver 와 동일한 데드밴드 변환을 적용해야 실제와 일치한다
        from actuators.motor_driver import MotorDriver

        def eff(cmd):
            mag = MotorDriver._apply_deadband(cmd)
            return 0.0 if mag == 0.0 else math.copysign(mag, cmd)

        vl = eff(self.motors.last_left) * self.MAX_WHEEL_SPEED_MPS
        vr = eff(self.motors.last_right) * self.MAX_WHEEL_SPEED_MPS

        # 엔코더가 '믿는' 이동량
        dl_cmd = vl * dt
        dr_cmd = vr * dt

        # [신규] 바퀴 미끄러짐(slip) 모델.
        #   실제 밭에서는 무른 흙/젖은 흙에서 바퀴가 헛돈다.
        #   엔코더는 도는데 로봇은 그만큼 안 나간다.
        #   -> 엔코더에는 dl_cmd 를 주입하되, 실제 위치는 (1-slip) 만큼만 움직인다.
        #   이것이 추측항법 오차가 **누적**되는 실제 메커니즘이다.
        dl = dl_cmd * (1.0 - self.slip_left)
        dr = dr_cmd * (1.0 - self.slip_right)
        d_center = (dl + dr) / 2.0
        d_theta = (dr - dl) / WHEEL_BASE_M

        mid = self.theta + d_theta / 2.0
        self.x += d_center * math.cos(mid)
        self.y += d_center * math.sin(mid)
        self.theta += d_theta

        # 엔코더 틱 주입 (소수부는 다음 스텝으로 이월)
        if self.odom is not None:
            # 엔코더는 미끄러짐을 모른다. 명령한 만큼 돌았다고 보고한다.
            tl = dl_cmd / self._dist_per_tick + self._tick_residual_l
            tr = dr_cmd / self._dist_per_tick + self._tick_residual_r
            il, ir = int(tl), int(tr)
            self._tick_residual_l = tl - il
            self._tick_residual_r = tr - ir
            if il or ir:
                self.odom.inject_ticks(il, ir)

        # 급수한 고랑 수 집계 (통계용)
        inside = self.inside_furrow_index() is not None
        if inside and not self._was_inside:
            self.furrows_watered += 1
        self._was_inside = inside

    # ------------------------------------------------------------------
    def inside_furrow_index(self):
        """지금 몇 번 고랑 안에 있는지. 밖이면 None."""
        if not (-self.ENTRANCE_MARGIN_M <= self.y <= self.FURROW_LENGTH_M):
            return None
        for k in range(1, self.n_furrows + 1):
            cx = (k - 1) * self.FURROW_SPACING_M
            if abs(self.x - cx) <= self.FURROW_HALF_WIDTH_M:
                return k
        return None

    def pump_zone_ok(self, margin: float = 0.0) -> bool:
        """펌프가 켜져 있어도 되는 위치인가 (여유 margin 포함)."""
        if not (-self.ENTRANCE_MARGIN_M - margin <= self.y
                <= self.FURROW_LENGTH_M + margin):
            return False
        for k in range(1, self.n_furrows + 1):
            cx = (k - 1) * self.FURROW_SPACING_M
            if abs(self.x - cx) <= self.FURROW_HALF_WIDTH_M + margin:
                return True
        return False

    def lateral_offset_in_furrow(self):
        """고랑 중심 기준 횡오프셋(m). 양수 = 로봇이 오른쪽(+x)에 치우침."""
        k = self.inside_furrow_index()
        if k is None:
            return None
        cx = (k - 1) * self.FURROW_SPACING_M
        return self.x - cx

    # ------------------------------------------------------------------
    @staticmethod
    def _ray_to_wall_m(dir_x: float, off: float, half: float):
        """
        로봇 옆면에서 쏜 빔이 이랑 벽(x = cx ± half 평면)에 닿는 거리.
        dir_x = 빔 방향의 월드 x 성분. 빔이 벽과 나란하면 무한대(측정 불가).
        """
        if abs(dir_x) < 1e-6:
            return None
        if dir_x > 0:
            d = (half - off) / dir_x
        else:
            d = (half + off) / (-dir_x)
        return d if d > 0 else None

    def tof_readings_mm(self):
        """
        (left_mm, right_mm) 실제 거리.

        [중요] 로봇의 좌/우는 **진행 방향에 따라 뒤바뀐다**.
        복귀 주행(로봇이 -y 를 향할 때)에는 로봇의 왼쪽이 월드 +x 쪽이 된다.
        처음 시뮬레이터를 짤 때 이걸 빠뜨려서 복귀 구간에서만 조향이
        양의 피드백이 되어 고랑을 이탈했다.
        (실기에서는 센서가 로봇에 붙어 있으므로 자동으로 뒤바뀐다)
        """
        off = self.lateral_offset_in_furrow()
        if off is None:
            return TOF_OUT_OF_RANGE_MM, TOF_OUT_OF_RANGE_MM
        half = self.FURROW_HALF_WIDTH_M
        # 로봇 왼쪽 방향 = theta + 90도,  오른쪽 = theta - 90도
        left_dir_x = -math.sin(self.theta)
        right_dir_x = math.sin(self.theta)

        def to_mm(d):
            if d is None:
                return TOF_OUT_OF_RANGE_MM
            return min(TOF_OUT_OF_RANGE_MM, max(10.0, d * 1000.0))

        return (
            to_mm(self._ray_to_wall_m(left_dir_x, off, half)),
            to_mm(self._ray_to_wall_m(right_dir_x, off, half)),
        )

    def visible_markers(self):
        """카메라에 보이는 마커를 {id: MarkerObservation} 로 반환."""
        out = {}
        for mid, (mx, my, facing) in self.markers.items():
            dx, dy = mx - self.x, my - self.y
            dist = math.hypot(dx, dy)
            if dist > self.MARKER_RANGE_M or dist < 1e-6:
                continue
            # [신규] 마커 면이 로봇 쪽을 향하고 있어야 검출된다.
            # 비스듬히 볼수록 사각형이 찌그러져 인식률이 급격히 떨어지고,
            # 뒤쪽에서는 아예 보이지 않는다.
            to_robot = math.atan2(-dy, -dx)
            face_off = math.atan2(
                math.sin(to_robot - facing), math.cos(to_robot - facing)
            )
            if abs(face_off) > self.MARKER_FACE_HALF_ANGLE_RAD:
                continue
            bearing_ccw = math.atan2(dy, dx) - self.theta
            bearing_ccw = math.atan2(math.sin(bearing_ccw), math.cos(bearing_ccw))
            if abs(bearing_ccw) > self.CAMERA_HFOV_RAD:
                continue
            bearing_right = -bearing_ccw
            cam_x = dist * math.sin(bearing_right)
            cam_z = dist * math.cos(bearing_right)
            out[mid] = MarkerObservation(
                marker_id=mid,
                distance_m=dist,
                forward_m=cam_z,
                lateral_offset_m=cam_x,
                yaw_error_rad=bearing_right,
            )
        return out

    def vision_result(self):
        """
        카메라가 보는 고랑 중앙선.

        [수정/중요] 예전에는 **로봇 몸통이 고랑 안에 있을 때만** 신뢰도를 줬다.
          그래서 입구 앞에서는 항상 신뢰도 0 이었고, "마커와 중앙선 거리를
          미리 알려주는" 방식 외에는 정렬할 방법이 없는 것처럼 보였다.
          실제 카메라는 **앞을 보므로**, 고랑 입구 앞에 서 있어도 고랑
          안쪽이 화면에 들어온다. 그걸 반영한다.

        조건
          - 로봇이 밭 안쪽을 향하고 있을 것 (뒤돌아 있으면 안 보인다)
          - 가장 가까운 고랑 중심에서 좌우로 크게 벗어나지 않을 것
          - 고랑 입구 앞 VISION_LOOKAHEAD_M 안쪽이거나 고랑 내부일 것
        """
        # 1) 몸통이 고랑 안이면 기존대로
        off = self.lateral_offset_in_furrow()
        if off is not None:
            lateral = (-off) * math.sin(self.theta)
            norm = max(-1.0, min(1.0, lateral / self.FURROW_HALF_WIDTH_M))
            return VisionLineResult(
                normalized_error=norm, heading_error=0.0,
                confidence=0.8, coverage=0.45,
            )

        # 2) 입구 앞: 카메라가 고랑 안쪽을 내다보는 상황
        if not (-self.VISION_LOOKAHEAD_M <= self.y < 0.0):
            return VisionLineResult(0.0, 0.0, 0.0, 0.02)

        # 밭 안쪽(+y)을 향해야 고랑이 보인다
        facing = math.sin(self.theta)          # +y 성분
        if facing < math.cos(math.radians(50.0)):
            return VisionLineResult(0.0, 0.0, 0.0, 0.02)

        # 가장 가까운 고랑 중심
        k = round(self.x / self.FURROW_SPACING_M)
        if not (0 <= k < self.n_furrows):
            return VisionLineResult(0.0, 0.0, 0.0, 0.02)
        cx = k * self.FURROW_SPACING_M
        dx = cx - self.x
        if abs(dx) > self.VISION_ENTRANCE_CAPTURE_M:
            return VisionLineResult(0.0, 0.0, 0.0, 0.02)

        lateral = dx * math.sin(self.theta)
        norm = max(-1.0, min(1.0, lateral / self.FURROW_HALF_WIDTH_M))
        # 멀수록·비스듬할수록 신뢰도가 떨어진다
        conf = 0.8 * facing * (1.0 - abs(self.y) / (self.VISION_LOOKAHEAD_M * 1.5))
        return VisionLineResult(
            normalized_error=norm, heading_error=0.0,
            confidence=max(0.0, min(0.8, conf)), coverage=0.4,
        )

    def is_water_low(self):
        if self.water_low_after_furrow is None:
            return False
        return self.furrows_watered > self.water_low_after_furrow


# ======================================================================
# 시뮬레이터에 물린 가짜 센서들
# ======================================================================
class SimAruco:
    """실제 ArucoDetector 와 같은 인터페이스. 다중 프레임 확인도 흉내낸다."""

    def __init__(self, world: SimWorld):
        self.world = world
        self._streak = {}

    def detect(self):
        present = self.world.visible_markers()
        for mid in list(self._streak):
            if mid not in present:
                del self._streak[mid]
        for mid in present:
            self._streak[mid] = self._streak.get(mid, 0) + 1
        return {
            mid: obs
            for mid, obs in present.items()
            if self._streak[mid] >= MARKER_CONFIRM_FRAMES
        }


class SimVision:
    def __init__(self, world: SimWorld):
        self.world = world
        self.force_blind = False

    def compute(self):
        if self.force_blind:
            return VisionLineResult(0.0, 0.0, 0.0, 0.01)
        return self.world.vision_result()


class SimCamera:
    def __init__(self):
        self.available = True
        self.fail_count = 0

    def capture_frame(self):
        return None

    def healthy(self):
        return True

    def close(self):
        pass


class SimWaterSensor:
    """실제 WaterTankSensor 를 그대로 쓰되 원시 신호만 시뮬레이터에서 받는다."""

    def __init__(self, world: SimWorld):
        from sensors.water_tank_sensor import WaterTankSensor

        self._inner = WaterTankSensor()
        self.world = world

    def poll(self):
        self._inner._sim_low = self.world.is_water_low()
        self._inner.poll()

    def is_water_low(self):
        return self._inner.is_water_low()

    def cleanup(self):
        self._inner.cleanup()

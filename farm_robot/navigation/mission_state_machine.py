# -*- coding: utf-8 -*-
"""
navigation/mission_state_machine.py
------------------------------------
전체 임무 로직을 담당하는 유한상태머신(FSM).

임무 흐름
  HOME 대기
    -> N번 고랑 입구 마커 탐색/정렬
    -> 고랑 진입 (펌프 ON) + 라인트래킹 직진
    -> 고랑 끝 감지 (좌우 ToF 모두 out-of-range, 연속 N틱 확인)
    -> 엔코더 기반 제자리 180도 유턴
    -> 복귀 주행 -> 입구 도달 -> 고랑 이탈 (펌프 OFF)
    -> [신규] 헤드랜드 이동: 90도 선회 -> 옆으로 이동 -> 90도 선회
    -> N+1번 고랑 탐색 반복
    -> 물통 부족 또는 밭 끝(END) 마커 확인 -> HOME 복귀 후 종료

------------------------------------------------------------------------
안전(Fail-safe) 설계 원칙
------------------------------------------------------------------------
1) 확실한 근거 없이는 다음 단계로 넘어가지 않는다.
   "모든 고랑 완료"는 전용 END 마커(FIELD_END_MARKER_ID)를 실제로 봤을 때만
   확정한다. 애매한 상황은 SAFE_HALT.
2) 모든 "전진" 동작에는 시간/거리 상한(watchdog)이 있다.

------------------------------------------------------------------------
이번 버전에서 고친 것
------------------------------------------------------------------------
1) [치명적] 헤드랜드 이동 상태가 없었다. 고랑을 빠져나온 뒤 제자리 회전만으로
   다음 고랑 입구를 찾으려 했지만 다음 입구는 옆으로 수 미터 떨어져 있어
   물리적으로 도달이 불가능했고, 결국 항상 SAFE_HALT 로 끝났다.
2) 수위 센서를 매 틱 폴링한다. 예전에는 고랑 1개당 1회만 호출해서
   디바운스 첫 호출이 항상 False -> 물이 떨어져도 고랑 하나를 더 돌았다.
3) MissionState.ERROR 가 아무 데서도 설정되지 않는 죽은 상태였다.
   이제 step() 을 예외 처리로 감싸고 연속 실패 시 ERROR 로 보낸다.
4) 조향을 MotorDriver.drive(base, steer) 로 일원화 (부호 규약 단일화).
5) Odometry.update() 를 매 틱 정확히 1회, 여기서만 호출한다.
6) 비전이 오래 죽어 있으면 경고에 그치지 않고 SAFE_HALT 로 전환한다.
"""

import math
import time
from enum import Enum, auto

from config import (
    ALIGN_PID_D_FILTER_HZ,
    ALIGN_PID_INTEGRAL_LIMIT,
    ALIGN_PID_KD,
    ALIGN_PID_KI,
    ALIGN_PID_KP,
    ALIGN_SPEED,
    BASE_SPEED,
    CONTROL_LOOP_DT,
    ENTRANCE_ENTER_DISTANCE_M,
    ENTRANCE_REQUIRE_WALLS_BEFORE_ENTER,
    ENTRANCE_ALLOW_BLIND_CREEP,
    ENTRANCE_HEADING_THRESHOLD_RAD,
    ENTRANCE_VISION_CENTER_THRESHOLD,
    ENTRANCE_VISION_MIN_CONFIDENCE,
    ENTRANCE_VISION_WAIT_TICKS,
    ENTRANCE_TOF_PROBE_ENABLED,
    SIGN_TOF_ERROR,
    TOF_NOMINAL_WALL_DISTANCE_MM,
    ENTRANCE_PROBE_SPEED_SCALE,
    ENTRANCE_PROBE_MAX_TRAVEL_M,
    ENTRANCE_PROBE_TOO_CLOSE_MM,
    ENTRANCE_PROBE_CENTER_TOLERANCE_MM,
    ENTRANCE_LATERAL_THRESHOLD_M,
    ENTRANCE_MARKER_LOST_ARRIVAL_M,
    ENTRANCE_MARKER_LOST_GRACE_SEC,
    ENTRANCE_APPROACH_HEADING_BLEND,
    ENTRANCE_ROTATE_IN_PLACE_RAD,
    ENTRANCE_MAX_APPROACH_HEADING_RAD,
    GATE_REJECT_TICKS_BEFORE_TRANSIT,
    FIELD_END_MARKER_MAX_BEARING_RAD,
    FIELD_END_MARKER_MAX_DISTANCE_M,
    FIELD_END_NEAR_PASS_MAX_DISTANCE_M,
    FIELD_END_NEAR_PASS_SEC,
    FURROW_END_MIN_TRAVEL_M,
    RETURN_ENTRANCE_MARKER_DISTANCE_M,
    GATE_HEADING_ANCHOR_MAX_CORRECTION_RAD,
    GATE_HEADING_ANCHOR_MAX_DISTANCE_M,
    GATE_HEADING_ANCHOR_MIN_BASELINE_M,
    HOME_RETURN_MARKER_LOCALIZATION,
    FURROW_WALL_ACQUIRE_MAX_TRAVEL_M,
    FURROW_WALL_ACQUIRE_SEC,
    MIN_FURROW_TRAVEL_DISTANCE_M,
    EXIT_FORWARD_DURATION_SEC,
    FIELD_END_MARKER_ID,
    MARKER_ON_RIDGE_CENTER,
    MARKER_POST_LATERAL_OFFSET_M,
    FIELD_HEADING_REANCHOR_MAX_DRIFT_DEG,
    HEADLAND_DIRECTION,
    HEADLAND_ANCHOR_HEADING,
    HEADLAND_ENABLED,
    HEADLAND_FALLBACK_DURATION_SEC,
    HEADLAND_SCAN_EVERY_N_TICKS,
    HEADLAND_SPEED_RATIO,
    HEADLAND_STEP_DISTANCE_M,
    HEADLAND_TURN_RAD,
    HOME_MARKER_ID,
    START_LOCALIZE_FROM_MARKER,
    SIGN_MARKER_LATERAL,
    furrow_marker_id,
    furrow_index_from_marker,
    MAX_APPROACH_DURATION_SEC,
    MAX_CONSECUTIVE_STEP_ERRORS,
    MAX_FURROW_TRAVEL_DURATION_SEC,
    PUMP_ON_RETURN_LEG,
    WATER_LOW_ABORT_INBOUND_LEG,
    WATER_LOW_ABORT_MIN_TRAVEL_M,
    MAX_HEADLAND_DURATION_SEC,
    MAX_HEADLAND_TRANSITS,
    MARKER_POST_TILT_DEG,
    MARKER_TILT_HEADING_MAX_CORRECTION_DEG,
    MARKER_TILT_HEADING_MAX_DISTANCE_M,
    USE_MARKER_TILT_FOR_FIELD_HEADING,
    MAX_SEARCH_RETRIES,
    MIN_INSIDE_FURROW_BEFORE_END_CHECK_SEC,
    REQUIRE_EXPLICIT_FIELD_END_MARKER,
    MAX_STEER_CORRECTION,
    SEARCH_CREEP_HEADING_GAIN,
    SEARCH_CREEP_MAX_DISTANCE_M,
    SEARCH_CREEP_SPEED,
    SEARCH_MODE,
    SEARCH_ROTATE_SPEED,
    SEARCH_MAX_REVOLUTIONS,
    SEARCH_PULSE_ON_TICKS,
    SEARCH_PULSE_OFF_TICKS,
    SEARCH_TIMEOUT_SEC,
    TELEMETRY_EVERY_N_TICKS,
    VISION_FALLBACK_HALT_SEC,
    VISION_FALLBACK_WARN_SEC,
)
from logutil import get_logger
from sensors.odometry import normalize_angle

log = get_logger("fsm")

# 데드밴드(MOTOR_DEADBAND_EPS) 위로 확실히 올려주기 위한 최소 명령값
ALIGN_TURN_MIN_COMMAND = 0.06


def _at_least(value: float, floor: float) -> float:
    """0이 아닌 명령은 최소 floor 이상의 크기를 갖게 한다(부호 유지)."""
    if value == 0.0:
        return 0.0
    return math.copysign(max(abs(value), floor), value)


class MissionState(Enum):
    INIT = auto()
    SEARCH_AND_ALIGN = auto()
    TRAVEL_INTO_FURROW = auto()
    TURN_AROUND = auto()
    TRAVEL_BACK_TO_ENTRANCE = auto()
    EXIT_FURROW = auto()
    HEADLAND_TRANSIT = auto()       # [신규] 고랑 사이 이동
    EVALUATE_MISSION = auto()
    HOME_ARRIVED = auto()
    MISSION_COMPLETE = auto()
    SAFE_HALT = auto()
    ERROR = auto()


class NavigationTarget(Enum):
    NEXT_FURROW = auto()
    HOME = auto()


class TransitPhase(Enum):
    TURN_OUT = auto()
    DRIVE = auto()
    TURN_IN = auto()


TERMINAL_STATES = (
    MissionState.MISSION_COMPLETE,
    MissionState.SAFE_HALT,
    MissionState.ERROR,
)


class MissionStateMachine:
    def __init__(self, deps=None):
        """
        deps: 테스트/시뮬레이션에서 하위 컴포넌트를 주입하기 위한 dict.
              키: odom, motors, pump, tof_pair, camera, aruco, vision_line,
                  water_sensor, furrow_mgr, line_follower
              (지정하지 않은 것만 실제 하드웨어 객체로 생성한다)
        """
        deps = deps or {}

        if deps:
            log.info("주입된 컴포넌트로 FSM 을 구성합니다: %s", sorted(deps.keys()))

        self.odom = deps.get("odom") or self._make_odometry()
        self.motors = deps.get("motors") or self._make_motors()
        self.pump = deps.get("pump") or self._make_pump()
        self.tof_pair = deps.get("tof_pair") or self._make_tof()
        self.camera = deps.get("camera") or self._make_camera()
        self.aruco = deps.get("aruco") or self._make_aruco()
        self.vision_line = deps.get("vision_line") or self._make_vision()
        self.water_sensor = deps.get("water_sensor") or self._make_water()
        self.furrow_mgr = deps.get("furrow_mgr") or self._make_furrow_mgr()
        self.line_follower = deps.get("line_follower") or self._make_line_follower()

        from control.pid_controller import PIDController

        self._align_pid_heading = PIDController(
            ALIGN_PID_KP, ALIGN_PID_KI, ALIGN_PID_KD,
            output_limit=1.0,
            integral_limit=ALIGN_PID_INTEGRAL_LIMIT,
            d_filter_hz=ALIGN_PID_D_FILTER_HZ,
        )
        self._align_pid_lateral = PIDController(
            ALIGN_PID_KP, ALIGN_PID_KI, ALIGN_PID_KD,
            output_limit=1.0,
            integral_limit=ALIGN_PID_INTEGRAL_LIMIT,
            d_filter_hz=ALIGN_PID_D_FILTER_HZ,
        )

        self.state = MissionState.INIT
        self.target = NavigationTarget.NEXT_FURROW
        self._state_enter_time = time.monotonic()
        self._entry_pending = True

        self._search_retry_count = 0
        self._approach_start_time = None
        self._vision_wait_ticks = 0
        self._probe_start_len = None
        self._halt_reason = ""
        self._vision_fallback_since = None
        self._consecutive_errors = 0
        self._tick = 0

        self._last_align_distance = None
        self._last_align_time = None
        self._walls_seen = False
        self._leg_start_len = 0.0
        self._gate_rejected_ticks = 0
        self._search_start_theta = 0.0
        self._search_start_len = 0.0
        # 탐색 2단계 진행 상태 (회전 -> 직선 전진 순서)
        self._search_rotate_done = False
        self._search_creep_exhausted = False
        # 시작 고랑 번호를 팻말로 확정했는가 (임무 시작 시 1회)
        self._start_localized = False
        # 밭 안쪽을 향하는 기준 방위. 입구 정렬을 마칠 때마다 갱신된다.
        self._field_heading = 0.0

        self._transit_phase = None
        self._transit_start_len = 0.0
        self._transit_start_time = 0.0
        self._transit_count = 0
        self._transit_start_ticks = 0

        # 외부 신호(SIGTERM 등)로 정지를 요청받았는가
        self._stop_requested = False
        # 물 부족으로 진입 주행을 중단했는가 (복귀 후 곧바로 HOME 으로)
        self._water_low_aborted = False
        # END 마커를 봤는가 = "지금 주행 중인 고랑이 마지막"
        # 이 고랑을 정상적으로 급수한 뒤 EVALUATE_MISSION 에서 소비된다.
        self._last_furrow_flagged = False
        # END 마커 고랑까지 급수를 마쳤는가 (= 임무 정상 완료)
        self._mission_finished_by_end_marker = False
        # HOME 까지 남은 헤드랜드 이동 칸 수.
        # 로봇은 자기가 몇 번째 고랑에 있는지 알고 있으므로, 돌아갈 칸 수를
        # 계산할 수 있다. 이 값이 0 이 될 때까지는 중간 정거장에서 전체
        # 탐색을 하지 않는다(아래 _state_headland_transit 주석 참고).
        self._home_transits_remaining = 0
        # 컴포넌트를 주입받았다면 시뮬레이션/테스트이므로 하드웨어 점검을 건너뛴다
        self._injected = bool(deps)

    # ------------------------------------------------------------------
    def _preflight_check(self) -> bool:
        """
        [신규] 바퀴가 구르기 **전에** 하드웨어 상태를 확인한다.

        예전에는 센서가 죽어 있어도 일단 출발한 뒤 주행 중에야 SAFE_HALT 가
        걸렸다. 밭 한가운데서 멈추는 것보다 출발 전에 거부하는 편이 안전하고
        원인 파악도 쉽다.
        """
        from config import REQUIRE_REAL_SENSORS

        if self._injected or not REQUIRE_REAL_SENSORS:
            return True

        problems = []

        if hasattr(self.tof_pair, "hardware_ok") and not self.tof_pair.hardware_ok():
            problems.append(
                "ToF 센서 2개가 정상 초기화되지 않았습니다 "
                "(I2C 활성화 여부, XSHUT 배선, 주소 재할당을 확인하세요). "
                "`i2cdetect -y 1` 로 0x30/0x31 이 보여야 합니다."
            )

        if hasattr(self.camera, "available") and not self.camera.available:
            problems.append(
                "카메라를 열지 못했습니다 (libcamera/picamera2 또는 /dev/video0 확인)."
            )

        if hasattr(self.odom, "is_available") and not self.odom.is_available():
            problems.append("오도메트리 입력 초기화에 실패했습니다.")

        if hasattr(self.motors, "available") and not self.motors.available:
            problems.append(
                "Arduino Mega Motion USB 연결에 실패했습니다: "
                + str(getattr(self.motors, "last_error", "원인 불명"))
            )

        from config import VISION_BACKEND
        if (
            VISION_BACKEND == "onnx_boundary"
            and hasattr(self.vision_line, "available")
            and not self.vision_line.available
        ):
            problems.append(
                "ONNX 고랑선 모델을 열지 못했습니다: "
                + str(getattr(self.vision_line, "last_error", "원인 불명"))
            )

        if not problems:
            log.info("하드웨어 사전 점검 통과.")
            return True

        for p in problems:
            log.critical("사전 점검 실패: %s", p)
        log.critical(
            "주행을 시작하지 않습니다. 문제를 고친 뒤 다시 실행하거나, "
            "의도적으로 무시하려면 config.REQUIRE_REAL_SENSORS=False 로 두세요."
        )
        self._halt_reason = "하드웨어 사전 점검 실패"
        self.state = MissionState.SAFE_HALT
        return False

    # ------------------------------------------------------------------
    # 하드웨어 컴포넌트 생성 (주입되지 않은 경우에만 호출됨)
    # ------------------------------------------------------------------
    def _make_odometry(self):
        # ODOMETRY_BACKEND: "encoder" | "encoder_imu" (config 13-1 참고)
        from sensors.odometry import create_odometry

        return create_odometry()

    def _make_motors(self):
        from control.mega_motion import MegaMotion

        return MegaMotion(odometry=self.odom)

    def _make_pump(self):
        from actuators.pump_controller import PumpController

        return PumpController()

    def _make_tof(self):
        from config import TOF_LEFT, TOF_RIGHT
        from sensors.tof_sensor import ToFPair

        return ToFPair(TOF_LEFT, TOF_RIGHT)

    def _make_camera(self):
        from sensors.camera import Camera

        return Camera()

    def _make_aruco(self):
        from sensors.aruco_detector import ArucoDetector

        return ArucoDetector(self.camera)

    def _make_vision(self):
        from config import VISION_BACKEND

        if VISION_BACKEND == "onnx_boundary":
            from sensors.onnx_furrow_line_detector import ONNXFurrowLineDetector

            return ONNXFurrowLineDetector(self.camera)
        if VISION_BACKEND == "ccrdnet":
            from sensors.ccrdnet_line_detector import CCRDNetLineDetector

            return CCRDNetLineDetector(self.camera)
        if VISION_BACKEND != "hsv":
            raise ValueError(f"지원하지 않는 VISION_BACKEND: {VISION_BACKEND}")
        from sensors.vision_line_detector import VisionLineDetector

        return VisionLineDetector(self.camera)

    def _make_water(self):
        # WATER_SOURCE: "gpio"(IR 센서) | "nano_usb"(Nano 시리얼) — 인터페이스 동일
        from config import WATER_SOURCE

        if WATER_SOURCE == "nano_usb":
            from sensors.nano_link import NanoWaterLink

            return NanoWaterLink()
        if WATER_SOURCE != "gpio":
            raise ValueError(f"지원하지 않는 WATER_SOURCE: {WATER_SOURCE}")
        from sensors.water_tank_sensor import WaterTankSensor

        return WaterTankSensor()

    def _make_furrow_mgr(self):
        from navigation.furrow_manager import FurrowManager

        return FurrowManager()

    def _make_line_follower(self):
        from control.line_follower import LineFollower

        return LineFollower(
            self.tof_pair,
            vision_detector=self.vision_line,
            odometry=self.odom,
        )

    # ------------------------------------------------------------------
    # 상태 전환 유틸
    # ------------------------------------------------------------------
    def _goto(self, new_state: MissionState):
        if new_state != self.state:
            log.info("상태 전환: %s -> %s", self.state.name, new_state.name)
        self.state = new_state
        self._state_enter_time = time.monotonic()
        self._entry_pending = True

    def _consume_entry(self) -> bool:
        """이 틱이 해당 상태의 첫 틱이면 True 를 반환하고 플래그를 소비한다."""
        if self._entry_pending:
            self._entry_pending = False
            return True
        return False

    def _elapsed_in_state(self) -> float:
        return time.monotonic() - self._state_enter_time

    def _safe_halt(self, reason: str):
        self.motors.stop()
        self.pump.set_zone(False)
        self.pump.turn_off()
        self._halt_reason = reason
        log.error("SAFE_HALT: %s", reason)
        self._goto(MissionState.SAFE_HALT)

    def _fatal(self, reason: str):
        self.motors.stop()
        self.pump.turn_off()
        self._halt_reason = reason
        log.critical("ERROR: %s", reason)
        self._goto(MissionState.ERROR)

    # ------------------------------------------------------------------
    def request_stop(self, reason: str = "외부 종료 요청"):
        """
        [신규] SIGTERM 등 외부 신호로 안전하게 멈추라는 요청.
        루프가 다음 틱에서 빠져나오고 _shutdown() 이 정리를 수행한다.
        """
        if not self._stop_requested:
            log.warning("%s -> 안전하게 정지합니다.", reason)
        self._stop_requested = True

    def _install_signal_handlers(self):
        """
        [신규/중요] systemd 로 자동 실행할 때 `systemctl stop` 은 SIGTERM 을 보낸다.
        예전 코드는 KeyboardInterrupt(SIGINT)만 처리했기 때문에, SIGTERM 을 받으면
        파이썬이 곧바로 죽으면서 **finally 블록이 실행되지 않고 PWM 이 마지막
        듀티에 그대로 남는다** -> 모터가 계속 돌고 펌프가 계속 물을 뿌린다.
        전원을 뽑기 전까지 멈추지 않는 매우 위험한 상태다.
        """
        try:
            import signal

            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                try:
                    signal.signal(
                        sig,
                        lambda s, f: self.request_stop(f"신호 {s} 수신"),
                    )
                except (ValueError, OSError, AttributeError):
                    # 메인 스레드가 아니거나 해당 신호가 없는 플랫폼
                    pass
        except Exception as exc:
            log.warning("신호 핸들러 설치 실패(무시하고 계속): %s", exc)

    def run_forever(self):
        self._install_signal_handlers()

        if not self._preflight_check():
            self._shutdown()
            return

        # [수정] 예전에는 step() 이 끝난 뒤 무조건 CONTROL_LOOP_DT 만큼 잤다.
        # 실제 주기 = step 소요시간 + 50ms 가 되어, 라즈베리파이4에서
        # ArUco 검출(30~50ms)이 도는 상태에서는 20Hz 가 아니라 10Hz 근처로
        # 떨어진다. 틱 수를 세는 상수들(TOF_END_CONFIRM_TICKS,
        # SEARCH_PULSE_*, GATE_REJECT_TICKS_BEFORE_TRANSIT 등)이 전부
        # 의도한 시간의 2배가 되어버린다.
        # 이제 '마감시각(deadline) 기준'으로 남은 시간만 자고, 주기를 못 지키면
        # 경고를 남긴다.
        next_deadline = time.monotonic()
        overrun_streak = 0

        try:
            while self.state not in TERMINAL_STATES:
                if self._stop_requested:
                    self._safe_halt("외부 종료 요청으로 임무를 중단했습니다.")
                    break

                self.step()

                next_deadline += CONTROL_LOOP_DT
                slack = next_deadline - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
                    overrun_streak = 0
                else:
                    overrun_streak += 1
                    # 너무 밀렸으면 따라잡으려 하지 말고 기준을 현재로 리셋
                    next_deadline = time.monotonic()
                    if overrun_streak in (20, 100, 500):
                        log.warning(
                            "제어 루프가 목표 주기(%.0fms)를 %d틱 연속 초과했습니다. "
                            "CAMERA_RESOLUTION 을 낮추거나 CONTROL_LOOP_HZ 를 "
                            "줄이세요.",
                            CONTROL_LOOP_DT * 1000.0, overrun_streak,
                        )
        except KeyboardInterrupt:
            log.info("사용자 중단(Ctrl+C).")
        finally:
            self._shutdown()

        if self.state == MissionState.SAFE_HALT:
            log.warning("SAFE_HALT 로 종료: %s", self._halt_reason)
            log.warning("원인을 점검한 뒤 다시 실행하세요 (자동 재시도 안 함).")
        elif self.state == MissionState.ERROR:
            log.error("임무가 ERROR 상태로 종료되었습니다: %s", self._halt_reason)
        else:
            log.info("임무 완료. %s", self.furrow_mgr.summary())

    def _shutdown(self):
        log.info("종료 정리 중...")
        for action in (
            lambda: self.motors.stop(),
            lambda: self.pump.turn_off(),
            lambda: self.vision_line.cleanup()
            if hasattr(self.vision_line, "cleanup")
            else None,
            lambda: self.motors.cleanup(),
            lambda: self.pump.cleanup(),
            lambda: self.tof_pair.close(),
            lambda: self.camera.close(),
            lambda: self.water_sensor.cleanup(),
            lambda: self.odom.cleanup(),
        ):
            try:
                action()
            except Exception as exc:
                log.warning("정리 중 예외 무시: %s", exc)

    # ------------------------------------------------------------------
    def step(self):
        """한 제어 틱. 예외가 나도 로봇이 폭주하지 않도록 방어한다."""
        self._tick += 1

        try:
            if (
                not self._injected
                and self._tick > 10
                and hasattr(self.motors, "link_ok")
                and not self.motors.link_ok()
            ):
                self._safe_halt(
                    "Arduino Mega Motion STATE가 0.5초 이상 끊겼습니다. "
                    "USB 케이블과 /dev/ttyACM 장치를 확인하십시오."
                )
                return

            # 매 틱 정확히 1회 - 위치 추정과 센서 폴링은 여기서만
            self.odom.update()
            self.water_sensor.poll()
            # No background keepalive in DRIVE mode: if this 20 Hz loop stalls,
            # the Mega's 400 ms command watchdog must stop the wheels.
            self.pump.tick()

            handler = {
                MissionState.INIT: self._state_init,
                MissionState.SEARCH_AND_ALIGN: self._state_search_and_align,
                MissionState.TRAVEL_INTO_FURROW: self._state_travel_into_furrow,
                MissionState.TURN_AROUND: self._state_turn_around,
                MissionState.TRAVEL_BACK_TO_ENTRANCE: self._state_travel_back,
                MissionState.EXIT_FURROW: self._state_exit_furrow,
                MissionState.HEADLAND_TRANSIT: self._state_headland_transit,
                MissionState.EVALUATE_MISSION: self._state_evaluate_mission,
                MissionState.HOME_ARRIVED: self._state_home_arrived,
            }.get(self.state)

            if handler is not None:
                handler()

            self._consecutive_errors = 0

        except Exception as exc:
            self._consecutive_errors += 1
            log.exception("step() 예외 (%d회 연속): %s", self._consecutive_errors, exc)
            try:
                self.motors.stop()
                self.pump.turn_off()
            except Exception:
                pass
            if self._consecutive_errors >= MAX_CONSECUTIVE_STEP_ERRORS:
                self._fatal(f"연속 {self._consecutive_errors}회 예외 발생: {exc}")

    # ------------------------------------------------------------------
    def _state_init(self):
        self.motors.stop()
        self.pump.set_zone(False)
        # IMU 오도메트리면 출발 전 정지 상태에서 자이로 바이어스를 실측한다.
        if hasattr(self.odom, "calibrate"):
            self.odom.calibrate()
        self.furrow_mgr.reset()
        self.target = NavigationTarget.NEXT_FURROW
        self._search_retry_count = 0
        self._search_rotate_done = False
        self._search_creep_exhausted = False
        self._approach_start_time = None
        self._vision_fallback_since = None
        self._transit_count = 0
        self._gate_rejected_ticks = 0
        # 로봇은 1번 고랑 입구(HOME)에서 밭 안쪽을 향해 놓인 상태로 시작한다고
        # 가정한다. 이 방위가 헤드랜드 이동의 기준이 된다.
        self._field_heading = self.odom.theta
        # 시작 고랑 번호는 첫 탐색 중 보이는 팻말로 정한다(_maybe_localize_start).
        # 마커는 연속 프레임 확인이 필요해서 INIT 단발 호출로는 잡히지 않는다.
        self._start_localized = not START_LOCALIZE_FROM_MARKER
        log.info("임무 시작. 첫 고랑 입구를 탐색합니다.")
        self._goto(MissionState.SEARCH_AND_ALIGN)

    def _furrow_center_target(self, obs):
        """팻말 관측 -> **고랑 중심 입구**의 (횡오차, 전방거리).

        팻말을 이랑 **중심**에 꽂고 이랑 간격을 측량값으로 알고 있으므로
        (config 'FIELD SURVEY' 블록), 고랑 중심은 계산할 수 있다:

            고랑 N 중심 = 이랑 N 팻말 - (간격 / 2)

        이 보정이 없으면 로봇은 **팻말(=이랑 위)을 향해** 접근하게 되어
        시작 위치가 조금만 벗어나도 고랑이 아니라 이랑으로 진입한다
        (실측: 실패 14건 중 6건이 이 원인이고 완료 고랑 0개였다).

        카메라 좌표계(x=오른쪽, z=전방) 기준으로 되돌려 준다.
        시뮬 참값 대조에서 78/78 관측, 오차 0.00cm 로 검증했다.
        """
        heading_err = normalize_angle(self.odom.theta - self._field_heading)
        # 월드에서 '고랑 쪽'(고랑 번호가 작아지는 쪽) 방향의 로봇 기준 베어링
        bearing = math.pi / 2.0 - heading_err
        # [중요] 옮기는 거리는 '간격/2'가 아니라 **측량된 실제 팻말 오프셋**이다.
        #   팻말을 이랑 중심이 아닌 곳에 꽂았다면 간격/2 를 쓰면 그만큼 빗나간다.
        shift = MARKER_POST_LATERAL_OFFSET_M
        return (
            obs.lateral_offset_m - math.sin(bearing) * shift,
            obs.forward_m + math.cos(bearing) * shift,
        )

    def _field_heading_from_post(self, observed):
        """팻말 **하나**로 밭 안쪽 방위를 유도한다 (설치 각도를 안다는 전제).

            F = theta + b - yaw - tilt

        팻말 2개가 동시에 보여야 하는 _reanchor_heading_from_posts 와 달리
        1개면 되므로 훨씬 자주 쓸 수 있다. 대신 마커 yaw 추정에 의존하므로
        거리/보정각 가드를 둔다(카메라 캘리브레이션 전제).
        반환: 밭 안쪽 방위(rad) 또는 None.
        """
        if not USE_MARKER_TILT_FOR_FIELD_HEADING:
            return None
        best = None
        for mid, obs in observed.items():
            if mid < 1 or mid == FIELD_END_MARKER_ID:
                continue
            if obs.forward_m <= 0 or obs.distance_m > MARKER_TILT_HEADING_MAX_DISTANCE_M:
                continue
            if best is None or obs.distance_m < best.distance_m:
                best = obs
        if best is None:
            return None
        bearing_ccw = -math.atan2(best.lateral_offset_m, best.forward_m)
        return normalize_angle(
            self.odom.theta + bearing_ccw - best.yaw_error_rad
            - math.radians(MARKER_POST_TILT_DEG)
        )

    def _maybe_localize_start(self, observed):
        """임무 시작 직후, 보이는 팻말로 '지금 몇 번 고랑 앞인가'를 확정한다.

        왜 필요한가
          시작 위치는 현장에서 매번 달라진다. 이 보정이 없으면 로봇은 자기
          위치와 무관하게 **무조건 1번 팻말**을 찾으므로, 조금만 옆에 놓아도
          1번이 화각 밖이라 탐색에 실패하고 눈앞의 2번 팻말은 무시한다.
          맵이 없는 시스템에서 팻말은 '지금 어디인가'를 아는 유일한 절대 기준이다.

        1회만 수행한다(아직 고랑을 하나도 완료하지 않은 시점).
        """
        if self._start_localized or self.target != NavigationTarget.NEXT_FURROW:
            return
        if self.furrow_mgr.current_index != 0 or self.furrow_mgr.completed:
            self._start_localized = True
            return
        located = self._localize_from_markers(observed)
        if located is None:
            return
        self._start_localized = True

        # [신규] 팻말 설치 각도를 알고 있으므로, 출발 자세를 가정하는 대신
        #   팻말에서 밭 안쪽 방위를 직접 유도한다. 시작 헤딩이 조금 틀어져
        #   있어도 이후 헤드랜드 이동이 어긋나지 않는다.
        derived = self._field_heading_from_post(observed)
        if derived is not None:
            drift = normalize_angle(derived - self._field_heading)
            if abs(drift) <= math.radians(MARKER_TILT_HEADING_MAX_CORRECTION_DEG):
                self._field_heading = derived
                if abs(drift) > math.radians(2.0):
                    log.info("팻말 각도로 밭 방위를 %+.1f도 보정했습니다.",
                             math.degrees(drift))
            else:
                log.warning("팻말 각도 기반 밭 방위 보정 %+.1f도는 상한(%.0f도)을 "
                            "넘어 무시합니다.", math.degrees(drift),
                            MARKER_TILT_HEADING_MAX_CORRECTION_DEG)
        if located > 1:
            # 1번보다 뒤에서 출발하면 그 사이 고랑은 급수 대상에서 빠진다.
            log.warning(
                "시작 위치가 %d번 고랑 앞입니다. 1~%d번 고랑은 이번 임무에서 "
                "제외됩니다. 의도한 것이 아니면 0번 팻말 근처에서 출발하세요.",
                located, located - 1,
            )
        else:
            log.info("시작 위치 확인: %d번 고랑 앞.", located)
        # current_index 는 '완료한 고랑 번호' 이므로 시작 고랑의 직전 값을 넣는다.
        self.furrow_mgr.current_index = located - 1

    def _current_target_marker_id(self):
        """이번에 찾을 팻말 마커 ID (고랑당 입구 팻말 1개)."""
        if self.target == NavigationTarget.HOME:
            return HOME_MARKER_ID
        return furrow_marker_id(self.furrow_mgr.next_index())

    # ------------------------------------------------------------------
    def _state_search_and_align(self):
        from sensors.aruco_detector import compute_post_bearing

        if self._consume_entry():
            self._search_start_theta = self.odom.theta
            self._search_start_len = self.odom.path_length

        observed = self.aruco.detect()
        self._maybe_localize_start(observed)
        # [신규] 팻말 2개가 동시에 보이면 그 두 점을 잇는 선이 곧 헤드랜드
        #   방향이므로, 밭 기준 방위를 **절대값으로** 다시 잡을 수 있다.
        #   IMU 가 없는 구성에서 엔코더 회전 오차를 지우는 유일한 수단인데,
        #   예전에는 HOME 복귀 경로에서만 호출되어 급수 주행 내내 한 번도
        #   보정되지 않았다. 여기서도 매 탐색 틱마다 시도한다.
        #   (함수 안에 베이스라인/거리/최대보정각 가드가 이미 있다)
        self._reanchor_heading_from_posts(observed)
        target_id = self._current_target_marker_id()

        # ---------------- END 마커 ----------------
        # [수정/치명적] END 마커의 의미를 바로잡는다.
        #
        #   END 마커는 **마지막 고랑의 팻말**에 붙어 있고, 그 뜻은
        #   "이 고랑이 마지막이다" 이다. "여기서 즉시 돌아가라"가 아니다.
        #
        #   예전 코드는 END 마커를 보는 즉시 target=HOME 으로 바꾸고 돌아섰다.
        #   그 결과 **마지막 고랑에는 물을 한 방울도 주지 않고** 임무를
        #   "완료"로 종료했다. 밭 하나를 통째로 빠뜨리는 심각한 버그였다.
        #
        #   올바른 동작:
        #     1) END 마커를 보면 "이번이 마지막 고랑"이라고 기억만 해 둔다
        #     2) 평소처럼 진입 -> 살수 -> 유턴 -> 복귀 -> 고랑 완료
        #     3) 그 다음(EVALUATE_MISSION)에 비로소 HOME 으로 향한다
        if (
            self.target == NavigationTarget.NEXT_FURROW
            and not self._last_furrow_flagged
            and self._end_marker_confirmed(observed)
        ):
            self._last_furrow_flagged = True
            log.info(
                "END 마커 확인 -> %d번 고랑이 마지막입니다. "
                "이 고랑에 물을 준 뒤 HOME 으로 복귀합니다.",
                self.furrow_mgr.next_index(),
            )
            # target 은 바꾸지 않는다. 이 고랑에 정상적으로 진입해야 한다.

        post_obs = observed.get(target_id)

        # 팻말이 보여도 "들어갈 수 있는 상태"인지 따로 검증한다.
        align = None
        reject_reason = None
        if post_obs is not None:
            candidate = compute_post_bearing(post_obs)
            # [중요] 진입 각도는 마커 자세(yaw)가 아니라 **밭 방위**로 잰다.
            #   팻말이 하나뿐이면 두 마커의 상대 위치를 쓸 수 없고, 단독 yaw 는
            #   포즈 모호성 때문에 프레임마다 수십 도씩 튄다.
            #   _field_heading 은 헤드랜드에서 정확히 90도 선회한 방위이고
            #   마커로 주기적으로 재보정되므로 훨씬 믿을 만하다.
            candidate.heading_error = normalize_angle(
                self.odom.theta - self._field_heading
            )
            # [신규] 팻말은 이랑 중심에 있다. 접근 목표를 팻말이 아니라
            #   **고랑 중심**으로 옮긴다(밭 측량값을 알고 있으므로 계산 가능).
            #   [주의] forward_distance 는 건드리지 않는다. 그 값은 '도착
            #   판정'(ENTRANCE_ENTER_DISTANCE_M)에 쓰이는데, 함께 옮기면
            #   정렬이 엉뚱한 지점에서 끝나 **펌프가 고랑 밖에서 켜진다**
            #   (실측: 인터록 위반 6~7회). 조향에 쓰는 횡오차만 옮긴다.
            if MARKER_ON_RIDGE_CENTER:
                candidate.lateral_error = self._furrow_center_target(post_obs)[0]
            if not candidate.valid:
                reject_reason = "팻말을 너무 가까이/비스듬히 보고 있음"
            elif abs(candidate.heading_error) > ENTRANCE_MAX_APPROACH_HEADING_RAD:
                # 입구는 보이지만 정면이 아니다. 이대로 직진하면 고랑을 가로질러
                # 작물을 밟는다. 위치를 옮겨서 다시 찾아야 한다.
                reject_reason = (
                    f"진입각 {math.degrees(candidate.heading_error):+.0f}도로 정면이 아님"
                )
            else:
                align = candidate

        if align is None and reject_reason is not None:
            self._gate_rejected_ticks += 1
            self.motors.stop()
            if self._gate_rejected_ticks == 1:
                log.warning(
                    "입구 마커(id=%s)는 보이지만 접근할 수 없습니다: %s",
                    target_id, reject_reason,
                )
            # 제자리에서 더 찾아봐야 소용없다. 헤드랜드로 위치를 옮긴다.
            if (
                self._gate_rejected_ticks >= GATE_REJECT_TICKS_BEFORE_TRANSIT
                and HEADLAND_ENABLED
                and self._transit_count < self._max_transits_allowed()
            ):
                log.info("위치를 바꾸기 위해 헤드랜드로 이동합니다.")
                self._gate_rejected_ticks = 0
                self._search_retry_count = 0
                self._approach_start_time = None
                self._start_headland_transit()
                return
        else:
            self._gate_rejected_ticks = 0

        # ---------------- 입구를 확보하지 못한 경우 ----------------
        if align is None:
            # 다 와서 마커가 화면 밖으로 나가는 것은 정상이다.
            # 팻말은 중심선에서 옆으로 치우쳐 서 있으므로, 가까워질수록
            # 화면 가장자리로 밀려난다. 오프셋 0.25m + 수평화각 62도이면
            # 0.42m 이내에서는 화면에 담기지 않는다. 직전까지 정상 접근
            # 중이었고 마지막 거리가 충분히 가까웠다면 "도착"으로 인정한다.
            if (
                self._last_align_distance is not None
                and self._last_align_time is not None
                and self._last_align_distance <= ENTRANCE_MARKER_LOST_ARRIVAL_M
                and time.monotonic() - self._last_align_time
                <= ENTRANCE_MARKER_LOST_GRACE_SEC
            ):
                log.info(
                    "근접(%.2fm)으로 마커가 화면을 벗어남 -> 입구 도착으로 판정.",
                    self._last_align_distance,
                )
                self.motors.stop()
                self._finish_alignment()
                return

            self._approach_start_time = None
            self._pulse_scan()

            # 시간 상한과 "충분히 훑었는가" 중 먼저 걸리는 쪽으로 종료한다.
            #   직선 탐색: 전진 거리 상한,  회전 탐색: 회전량 상한
            if self._search_phase() == "creep":
                swept_enough = (
                    self._search_creep_distance() >= SEARCH_CREEP_MAX_DISTANCE_M
                )
            else:
                rotated = abs(self.odom.theta - self._search_start_theta)
                swept_enough = rotated >= 2.0 * math.pi * SEARCH_MAX_REVOLUTIONS
            if self._elapsed_in_state() <= SEARCH_TIMEOUT_SEC and not swept_enough:
                return

            self.motors.stop()
            self._search_start_theta = self.odom.theta
            self._search_start_len = self.odom.path_length

            # 단계 전환: 회전으로 못 찾으면 직선 전진, 그것도 실패하면 재시도.
            # (단계 전환은 재시도 카운트를 소비하지 않는다)
            phase = self._search_phase()
            if phase == "rotate" and SEARCH_MODE == "rotate_then_creep"                     and not self._search_rotate_done:
                self._search_rotate_done = True
                log.info("제자리 회전으로 팻말을 찾지 못했습니다. "
                         "직선 %.1fm 전진 탐색으로 전환합니다.",
                         SEARCH_CREEP_MAX_DISTANCE_M)
                self._goto(MissionState.SEARCH_AND_ALIGN)
                return
            if phase == "creep":
                self._search_creep_exhausted = True
                log.info("직선 탐색 %.1fm 동안 팻말을 찾지 못했습니다.",
                         SEARCH_CREEP_MAX_DISTANCE_M)
                self._goto(MissionState.SEARCH_AND_ALIGN)
                return

            self._search_retry_count += 1

            if self._search_retry_count <= MAX_SEARCH_RETRIES:
                log.warning(
                    "마커(id=%s) 탐색 실패 %d회. 재탐색합니다.",
                    target_id, self._search_retry_count,
                )
                self._goto(MissionState.SEARCH_AND_ALIGN)
                return

            # 제자리 탐색으로는 못 찾음 -> 헤드랜드로 한 칸 이동해서 다시 탐색
            if HEADLAND_ENABLED and self._transit_count < self._max_transits_allowed():
                log.warning(
                    "제자리 탐색 실패. 헤드랜드로 한 칸 이동 후 재탐색합니다 (%d/%d).",
                    self._transit_count + 1, self._max_transits_allowed(),
                )
                self._search_retry_count = 0
                self._start_headland_transit()
                return

            if (
                self.target == NavigationTarget.NEXT_FURROW
                and not REQUIRE_EXPLICIT_FIELD_END_MARKER
            ):
                log.warning(
                    "END 마커 없이 부재 기반으로 홈 복귀를 시도합니다 "
                    "(REQUIRE_EXPLICIT_FIELD_END_MARKER=False)."
                )
                self.target = NavigationTarget.HOME
                self._search_retry_count = 0
                self._transit_count = 0
                self._goto(MissionState.SEARCH_AND_ALIGN)
                return

            self._safe_halt(
                f"목표 팻말(id={target_id})을 제자리 탐색 "
                f"{MAX_SEARCH_RETRIES + 1}회와 헤드랜드 이동 {self._transit_count}회 "
                f"이후에도 찾지 못했습니다. 마커 오염/가림/조명 문제 또는 "
                f"END 마커 미부착이 의심됩니다. 사람의 확인이 필요합니다."
            )
            return

        # ---------------- 입구 확보: 정렬/접근 ----------------
        if self._approach_start_time is None:
            self._approach_start_time = time.monotonic()
            self.furrow_mgr.mark_attempt()

        if time.monotonic() - self._approach_start_time > MAX_APPROACH_DURATION_SEC:
            self.motors.stop()
            self._safe_halt(
                f"입구 팻말(id={target_id})은 보이지만 "
                f"{MAX_APPROACH_DURATION_SEC:.0f}초 동안 정렬/접근을 완료하지 "
                f"못했습니다. 거리/자세 추정 오류 또는 모터 데드밴드가 "
                f"의심되어 더 이상 전진하지 않습니다."
            )
            return

        self._last_align_distance = align.forward_distance
        self._last_align_time = time.monotonic()

        # ================= 중심 정렬 =================
        # [수정/핵심] 마커는 "여기 고랑 입구가 있다"까지만 알려준다.
        #   고랑 중심이 마커에서 몇 cm 옆인지는 마커 안에 없는 정보다.
        #   그걸 상수로 넣어두면(예전 방식) 현장에서 팻말을 조금만 옮겨
        #   박아도 로봇이 이랑으로 돌진한다.
        #
        #   그래서 역할을 나눈다.
        #     마커 : 어느 고랑인지 + 그 입구가 어느 방향인지 (거친 접근)
        #     비전 : 고랑 자체를 보고 **중심선을 직접 찾는다** (정밀 정렬)
        #     ToF  : 고랑에 코를 들이민 뒤 좌우 벽 대칭으로 확인
        vision_result = None
        if self.vision_line is not None:
            try:
                vision_result = self.vision_line.compute()
            except Exception as exc:
                log.warning("입구 정렬 중 비전 실패: %s", exc)

        vision_ready = (
            vision_result is not None
            and vision_result.confidence >= ENTRANCE_VISION_MIN_CONFIDENCE
        )

        heading_error = normalize_angle(self.odom.theta - self._field_heading)

        # ---- 1단계: 비전이 아직 고랑을 못 봄 -> 마커 방위로 거친 접근 ----
        if not vision_ready:
            if align.forward_distance < ENTRANCE_ENTER_DISTANCE_M:
                # 마커 코앞까지 왔는데 비전이 고랑을 못 본다.
                # 비전 하나만 믿으면 흐린 날 로봇이 아예 못 들어간다.
                # -> ToF 로 벽을 더듬어 중심을 찾는다.
                if ENTRANCE_TOF_PROBE_ENABLED:
                    return self._probe_entrance_with_tof(target_id)

                self.motors.stop()
                self._vision_wait_ticks += 1
                if self._vision_wait_ticks > ENTRANCE_VISION_WAIT_TICKS:
                    self._safe_halt(
                        f"입구 팻말(id={target_id})까지 접근했지만 비전이 고랑을 "
                        f"확인하지 못했습니다(신뢰도 "
                        f"{vision_result.confidence if vision_result else 0.0:.2f}). "
                        f"흙 HSV 임계값 또는 카메라 각도를 점검하세요."
                    )
                return

            # 마커를 향해 조향하며 접근 (순수추종)
            bearing = math.atan2(
                align.lateral_error, max(align.forward_distance, 0.2)
            )
            if abs(bearing) > ENTRANCE_ROTATE_IN_PLACE_RAD:
                cmd = _at_least(
                    ALIGN_SPEED * self._align_pid_heading.compute(bearing),
                    ALIGN_TURN_MIN_COMMAND,
                )
                self.motors.set_speeds(cmd, -cmd)
                return
            steer = self._align_pid_lateral.compute(bearing)
            steer = max(-ALIGN_SPEED * 0.8, min(ALIGN_SPEED * 0.8, steer))
            self.motors.drive(ALIGN_SPEED, steer)
            return

        # ---- 2단계: 비전이 고랑을 봄 -> 중심선에 정렬 ----
        self._vision_wait_ticks = 0
        centre_error = vision_result.normalized_error   # 양수 = 오른쪽으로 가야 함

        if abs(centre_error) > ENTRANCE_VISION_CENTER_THRESHOLD:
            # 옆으로 이동해야 하는데 제자리 회전으로는 위치가 안 바뀐다.
            # 중심선을 향해 호를 그리며 전진해 수렴시킨다.
            steer = self._align_pid_lateral.compute(centre_error)
            steer = max(-ALIGN_SPEED * 0.8, min(ALIGN_SPEED * 0.8, steer))
            self.motors.drive(ALIGN_SPEED, steer)
            return

        # ---- 2.5단계: 중심에 섰지만 아직 입구에서 멀면 계속 전진 ----
        # 여기서 멈추면 고랑 밖에서 "정렬 완료"가 되어, 펌프가 밭이 아닌
        # 헤드랜드에서 켜진다. 중심을 유지한 채 입구까지 다가간다.
        # (거리 조건일 뿐, '마커-중심선 거리' 같은 사전 지식이 아니다)
        walls_seen = self.tof_pair.walls_visible()
        # [수정] 팻말이 고랑 중심선에서 멀리(이랑 중심) 있으면 '마커까지의
        #   거리'만으로 도착을 판정할 수 없다. ToF 로 좌우 벽을 실제로 본
        #   뒤에야 진입을 확정한다(그 전까지는 중심을 유지하며 계속 전진).
        #   벽을 끝내 못 보면 MAX_APPROACH_DURATION_SEC 로 SAFE_HALT 된다.
        too_far = align.forward_distance > ENTRANCE_ENTER_DISTANCE_M
        if ENTRANCE_REQUIRE_WALLS_BEFORE_ENTER:
            keep_approaching = not walls_seen
        else:
            keep_approaching = too_far and not walls_seen
        if keep_approaching:
            steer = self._align_pid_lateral.compute(centre_error)
            steer = max(-ALIGN_SPEED * 0.8, min(ALIGN_SPEED * 0.8, steer))
            self.motors.drive(ALIGN_SPEED, steer)
            return

        # ---- 3단계: 중심에 섰으면 진입각만 정밀 조정 ----
        # 제자리 회전은 위치를 바꾸지 않으므로 방금 맞춘 중심이 흐트러지지 않는다.
        if abs(heading_error) > ENTRANCE_HEADING_THRESHOLD_RAD:
            cmd = _at_least(
                ALIGN_SPEED * self._align_pid_heading.compute(heading_error),
                ALIGN_TURN_MIN_COMMAND,
            )
            self.motors.set_speeds(cmd, -cmd)
            return

        # ---- 정렬 완료 ----
        self.motors.stop()
        log.info(
            "입구 정렬 완료 (비전 중심오차 %+.3f, 진입각 %+.1f도).",
            centre_error, math.degrees(heading_error),
        )
        self._finish_alignment()

    def _probe_entrance_with_tof(self, target_id):
        """
        [신규] 비전이 고랑을 못 볼 때, ToF 로 벽을 더듬어 중심을 찾는다.

        왜 필요한가
        -----------
        비전(흙 색상)만으로 진입을 판단하면, 흐린 날이나 비 온 뒤처럼 흙
        색이 달라지는 날 로봇이 **아예 고랑에 못 들어간다.** 임무 전체가
        멈추는 셈이라 실용상 치명적이다.

        원리
        ----
        고랑 입구에 코를 조금씩 들이밀면 좌우 ToF 가 이랑 벽을 잡기 시작한다.

            양쪽 다 벽 보임        -> 고랑 안. 좌우 거리차로 중심을 맞춘다.
            한쪽만, 아주 가까이    -> 이랑에 붙었다. 반대쪽으로 조향.
            양쪽 다 안 보임        -> 아직 고랑 앞. 조금 더 전진.

        비전과 하는 일은 같지만 **근거가 다르다**(색상 vs 거리).
        그래서 조명이 나빠도 동작한다.

        [궤도(무한궤도) 주의] 궤도는 회전 시 지면을 비비며 미끄러지므로
        엔코더 각도가 부정확하다. 여기서는 각도를 쓰지 않고 **좌우 거리차**
        만으로 판단하므로 그 영향을 받지 않는다.
        """
        if self._probe_start_len is None:
            log.info(
                "비전이 고랑을 확인하지 못해 ToF 탐침으로 진입을 시도합니다 "
                "(팻말 id=%d).", target_id,
            )

        # [중요] 먼저 밭 안쪽을 향하도록 자세를 잡는다.
        #   그러지 않으면 마커를 향하던 헤딩 그대로 직진해서 팻말 쪽으로
        #   밀려난다(실제로 고랑 중심에서 28cm 벗어나는 것을 재현했다).
        #   탐침은 '곧게 들어가서 좌우 벽을 더듬는' 동작이므로 방향이 먼저다.
        heading_error = normalize_angle(self.odom.theta - self._field_heading)
        if abs(heading_error) > ENTRANCE_HEADING_THRESHOLD_RAD:
            cmd = _at_least(
                ALIGN_SPEED * self._align_pid_heading.compute(heading_error),
                ALIGN_TURN_MIN_COMMAND,
            )
            self.motors.set_speeds(cmd, -cmd)
            return

        if self._probe_start_len is None:
            self._probe_start_len = self.odom.path_length

        travelled = self.odom.path_length - self._probe_start_len
        if travelled > ENTRANCE_PROBE_MAX_TRAVEL_M:
            self.motors.stop()
            self._safe_halt(
                f"ToF 탐침으로 {ENTRANCE_PROBE_MAX_TRAVEL_M:.1f}m 전진했지만 "
                f"좌우 이랑 벽을 잡지 못했습니다. 고랑이 아닌 곳이거나 "
                f"ToF 장착 높이/각도가 잘못되었을 수 있습니다."
            )
            return

        left_mm, right_mm = self.tof_pair.read()
        left_ok = self.tof_pair.left.wall_visible()
        right_ok = self.tof_pair.right.wall_visible()
        speed = ALIGN_SPEED * ENTRANCE_PROBE_SPEED_SCALE

        # (a) 한쪽 벽에 너무 붙었다 -> 반대쪽으로 밀어낸다
        if left_ok and left_mm < ENTRANCE_PROBE_TOO_CLOSE_MM:
            self.motors.drive(speed, +ALIGN_SPEED * 0.5)   # 오른쪽으로
            return
        if right_ok and right_mm < ENTRANCE_PROBE_TOO_CLOSE_MM:
            self.motors.drive(speed, -ALIGN_SPEED * 0.5)   # 왼쪽으로
            return

        # (b) 양쪽 다 보인다 -> 좌우 거리차로 중심을 맞춘다
        if left_ok and right_ok:
            diff = left_mm - right_mm      # 양수 = 왼쪽이 멀다 = 왼쪽으로 가야
            if abs(diff) <= ENTRANCE_PROBE_CENTER_TOLERANCE_MM:
                self.motors.stop()
                log.info(
                    "ToF 탐침으로 고랑 중심을 찾았습니다 "
                    "(좌 %.0fmm / 우 %.0fmm). 진입합니다.", left_mm, right_mm,
                )
                self._finish_alignment()
                return
            # SIGN_TOF_ERROR 규약: 양수 오차 = 오른쪽으로 가야 함
            err = SIGN_TOF_ERROR * (-diff) / TOF_NOMINAL_WALL_DISTANCE_MM
            steer = max(-0.5, min(0.5, err))
            self.motors.drive(speed, steer)
            return

        # (c) 한쪽만 보인다 -> 보이는 쪽에서 살짝 멀어지며 전진
        if left_ok:
            self.motors.drive(speed, +ALIGN_SPEED * 0.25)
            return
        if right_ok:
            self.motors.drive(speed, -ALIGN_SPEED * 0.25)
            return

        # (d) 아직 아무것도 안 보인다 -> 곧게 조금 더 전진
        self.motors.drive(speed, 0.0)

    def _search_phase(self) -> str:
        """지금 탐색 단계가 '회전'인가 '직선 전진'인가.

        기본 정책(rotate_then_creep): **회전을 먼저** 한다. 팻말은 전부
        헤드랜드선에 있고 로봇은 밭쪽을 보고 출발하므로, 직선 전진은
        마커선에서 멀어진다. 회전으로 화각을 훑어도 못 찾을 때만 이동한다.
        """
        if SEARCH_MODE == "creep":
            return "rotate" if self._search_creep_exhausted else "creep"
        if SEARCH_MODE == "rotate_then_creep":
            if self._search_rotate_done and not self._search_creep_exhausted:
                return "creep"
            return "rotate"
        return "rotate"

    def _moving_pulse(self) -> bool:
        """이번 틱이 '움직이는 구간'인가.

        계속 움직이면 모션 블러로 마커가 잘 안 잡힌다(데드밴드 때문에 속도를
        더 낮출 수도 없다). "조금 움직이고 -> 멈춰서 촬영"을 반복해서
        멈춘 틱의 프레임으로 마커를 검출한다.
        """
        period = max(1, SEARCH_PULSE_ON_TICKS + SEARCH_PULSE_OFF_TICKS)
        return self._tick % period < SEARCH_PULSE_ON_TICKS

    def _search_creep_distance(self) -> float:
        """이번 탐색에서 직선으로 전진한 거리(m)."""
        return self.odom.path_length - self._search_start_len

    def _pulse_scan(self):
        """탐색 동작 1틱.

        SEARCH_MODE="creep" (기본, 운용 시퀀스):
            밭 방위(_field_heading)를 유지하며 **직선으로 조금씩 전진**하면서
            마커를 찾는다. 팻말은 진행 방향 앞쪽에 있으므로 제자리에서 도는
            것보다 이쪽이 먼저 찾는다. SEARCH_CREEP_MAX_DISTANCE_M 만큼
            전진하면 종료되고, 재시도 때는 아래 회전 탐색으로 넘어간다.
        SEARCH_MODE="rotate" (또는 직선 탐색 소진 후):
            제자리 회전 탐색. 옆/뒤쪽 팻말까지 훑는다.
        """
        if self._search_phase() == "creep":
            if not self._moving_pulse():
                self.motors.stop()
                return
            # 직진 중 헤딩이 틀어지면 고랑을 가로지르게 되므로 방위를 잡아준다.
            heading_error = normalize_angle(self.odom.theta - self._field_heading)
            steer = max(-MAX_STEER_CORRECTION, min(
                MAX_STEER_CORRECTION, SEARCH_CREEP_HEADING_GAIN * heading_error
            ))
            self.motors.drive(SEARCH_CREEP_SPEED, steer)
            return

        if self._moving_pulse():
            self.motors.rotate_in_place(clockwise=True, speed=SEARCH_ROTATE_SPEED)
        else:
            self.motors.stop()

    def _end_marker_confirmed(self, observed) -> bool:
        """
        END 마커를 '확실히' 봤는가.
        옆에서 비스듬히/멀리 보이는 것은 인정하지 않는다.
        """
        obs = observed.get(FIELD_END_MARKER_ID)
        if obs is None:
            return False
        if obs.forward_m <= 0 or obs.forward_m > FIELD_END_MARKER_MAX_DISTANCE_M:
            return False
        # [수정] 팻말 설치 위치를 코드가 모르므로, '정면 여부'는 마커 자체의
        #   방위로만 본다. END 마커는 입구 팻말에 함께 붙으므로, 그 팻말을
        #   똑바로 마주보는 시점에 확인된다.
        #   임계값은 팻말이 옆으로 치우쳐 있어도 잡히도록 넉넉히 잡는다.
        bearing = math.atan2(obs.lateral_offset_m, max(obs.forward_m, 0.05))
        return abs(bearing) <= FIELD_END_MARKER_MAX_BEARING_RAD

    def _finish_alignment(self):
        """정렬/접근 완료 후 다음 상태로 넘어간다."""
        self._align_pid_heading.reset()
        self._align_pid_lateral.reset()
        self._search_retry_count = 0
        self._approach_start_time = None
        self._vision_wait_ticks = 0
        self._probe_start_len = None
        self._transit_count = 0
        self._last_align_distance = None
        self._last_align_time = None
        self._gate_rejected_ticks = 0
        # 지금 로봇은 입구를 정면으로 바라보고 있다 = 밭 안쪽 방위.
        # 헤드랜드 이동은 이 방위를 기준으로 절대 각도로 계산한다.
        self._field_heading = self.odom.theta

        if self.target == NavigationTarget.HOME:
            self._goto(MissionState.HOME_ARRIVED)
        else:
            log.info("%d번 고랑 입구 정렬 완료. 진입합니다.", self.furrow_mgr.next_index())
            self._goto(MissionState.TRAVEL_INTO_FURROW)

    # ------------------------------------------------------------------
    def _begin_furrow_leg(self, returning: bool = False):
        """
        고랑 내부 주행 구간 시작 시 타당성 검사 상태를 초기화한다.

        [수정] 마커가 입구에만 있으므로 이 구간에서 쓸 마커 정보가 없다.
        고랑 안에서는 비전(흙 중앙선)과 ToF(좌우 벽)만으로 달린다.
        """
        self._walls_seen = False
        self._leg_start_len = self.odom.path_length
        self._returning_leg = returning

    def _drive_in_furrow(self) -> bool:
        """
        고랑 내부 주행 공통 로직.
        반환: 고랑 끝(또는 입구)에 도달했으면 True.

        ToF 타당성 검사
          - ToF 는 **진입 첫 틱부터** 매 틱 읽는다(line_follower.step 안에서).
            아래 유예값은 '언제부터 보는가'가 아니라 '언제 포기하는가'다.
          - 좌우 벽을 한 번도 못 본 채 FURROW_WALL_ACQUIRE_SEC(시간) 또는
            FURROW_WALL_ACQUIRE_MAX_TRAVEL_M(거리) 중 **먼저 걸리는 쪽**에
            도달하면 SAFE_HALT (예전에는 이 상황에서 짧은 왕복을 반복하며
            물도 안 주고 "임무 완료"로 끝났다).
          - 거리 기준을 함께 두는 이유: 시간만 보면 "천천히 정렬하며 진입하는
            정상 동작"과 "벽 없는 곳을 계속 달리는 고장"을 구분하지 못한다.
          - 최소 주행 거리(MIN_FURROW_TRAVEL_DISTANCE_M) 전의 '고랑 끝'
            신호는 받아들이지 않는다.
        """
        result = self.line_follower.step()
        self.motors.drive(result.base_speed, result.steer)
        self._monitor_vision_fallback(result)
        self._telemetry(result)

        if self.tof_pair.walls_visible():
            self._walls_seen = True

        elapsed = self._elapsed_in_state()
        travelled = self.odom.path_length - self._leg_start_len

        # [수정] 시간뿐 아니라 **주행 거리**로도 끊는다. 시간만 보면
        #   "천천히 정렬하며 진입하는 정상 동작"과 "벽 없는 곳을 계속 달리는
        #   고장"을 구분하지 못한다. 둘 중 먼저 걸리는 쪽에서 정지.
        if not self._walls_seen and (
            elapsed > FURROW_WALL_ACQUIRE_SEC
            or travelled > FURROW_WALL_ACQUIRE_MAX_TRAVEL_M
        ):
            self.motors.stop()
            self._safe_halt(
                f"고랑 주행 시작 후 {elapsed:.1f}초 / {travelled:.2f}m 동안 "
                f"좌우 이랑 벽을 한 번도 감지하지 못했습니다. ToF 배선/주소 설정 "
                f"오류이거나 고랑이 아닌 곳에 진입했을 수 있습니다."
            )
            return False

        if travelled < MIN_FURROW_TRAVEL_DISTANCE_M:
            return False

        if travelled < FURROW_END_MIN_TRAVEL_M:
            return False

        # [신규] 복귀 구간에서는 **입구 마커가 다시 보이는 것**이 추가 근거다.
        #   출구에는 마커가 없지만 입구에는 있으므로, 돌아올 때는 마커가
        #   정면에 나타난다. ToF 벽 소실보다 이르고 확실한 신호다.
        if self._returning_leg:
            entry_id = furrow_marker_id(self.furrow_mgr.next_index())
            obs = self._safe_detect().get(entry_id)
            if obs is not None and 0 < obs.distance_m <= RETURN_ENTRANCE_MARKER_DISTANCE_M:
                log.info(
                    "복귀 중 입구 마커(id=%d)를 %.2fm 앞에서 확인 -> 고랑을 빠져나왔습니다.",
                    entry_id, obs.distance_m,
                )
                return True

        return (
            elapsed > MIN_INSIDE_FURROW_BEFORE_END_CHECK_SEC
            and result.furrow_end_detected
        )

    def _state_travel_into_furrow(self):
        if self._consume_entry():
            self.pump.set_zone(True)
            self.line_follower.reset()
            self._begin_furrow_leg()
            if not self.water_sensor.is_water_low():
                self.pump.turn_on()
            else:
                log.warning("물통이 비어 있어 펌프를 켜지 않고 주행만 합니다.")

        # 진입 직후 팻말 옆을 지나칠 때 END 마커를 한 번 더 확인 (보험)
        self._check_end_marker_near_pass()

        # 주행 중 물이 떨어지면 즉시 펌프 잠금
        self.pump.set_lockout(self.water_sensor.is_water_low())

        # [신규] 물이 떨어졌으면 고랑 끝까지 갈 이유가 없다.
        # 살수가 불가능한 상태로 남은 진입 거리를 마저 달리는 것은
        # 시간과 배터리 낭비일 뿐이다. 즉시 유턴해서 입구로 복귀한다.
        # (고랑 밖으로 나가는 유일한 경로가 '입구로 되돌아가기'이므로
        #  복귀 주행 자체는 생략할 수 없다)
        if WATER_LOW_ABORT_INBOUND_LEG and self.water_sensor.is_water_low():
            travelled = self.odom.path_length - self._leg_start_len
            if travelled >= WATER_LOW_ABORT_MIN_TRAVEL_M:
                self.motors.stop()
                log.warning(
                    "물통 부족 감지(진입 %.2fm 지점). 고랑 끝까지 가지 않고 "
                    "즉시 유턴하여 입구로 복귀합니다.", travelled,
                )
                self._water_low_aborted = True
                self._goto(MissionState.TURN_AROUND)
                return

        if self._elapsed_in_state() > MAX_FURROW_TRAVEL_DURATION_SEC:
            self.motors.stop()
            self._safe_halt(
                f"고랑 진입 후 {MAX_FURROW_TRAVEL_DURATION_SEC:.0f}초 동안 "
                f"ToF 가 고랑 끝을 감지하지 못했습니다. 센서 오작동 또는 "
                f"예상치 못한 장애물이 의심되어 정지합니다."
            )
            return

        if self._drive_in_furrow():
            self.motors.stop()
            log.info("고랑 끝 도달. 유턴합니다.")
            self._goto(MissionState.TURN_AROUND)

    def _check_end_marker_near_pass(self):
        """
        진입 직후 팻말 옆을 지나칠 때 END 마커를 한 번 더 확인한다(보험).

        정면 접근 중(SEARCH_AND_ALIGN)에 확인하는 것이 1순위지만, 햇빛
        반사나 순간적인 가림으로 놓칠 수 있다. 진입 직후에는 팻말 바로 옆을
        스쳐 지나가므로 거리가 매우 가깝다. 이때는 마커가 옆으로 보이므로
        정면 판정이 성립하지 않아 **거리 조건만** 본다.
        """
        if self._last_furrow_flagged:
            return
        if self._elapsed_in_state() > FIELD_END_NEAR_PASS_SEC:
            return

        obs = self._safe_detect().get(FIELD_END_MARKER_ID)
        if obs is None:
            return
        if 0 < obs.distance_m <= FIELD_END_NEAR_PASS_MAX_DISTANCE_M:
            self._last_furrow_flagged = True
            log.info(
                "진입 중 근접(%.2fm)에서 END 마커 확인 -> %d번 고랑이 마지막입니다.",
                obs.distance_m, self.furrow_mgr.next_index(),
            )

    def _state_turn_around(self):
        self.pump.set_zone(False)      # 회전 중에는 물을 뿌리지 않는다
        self.motors.stop()
        ok = self.motors.turn_180_blocking()
        if not ok:
            log.warning("180도 유턴이 각도 도달 없이 종료되었습니다(타임아웃/엔코더).")
        self.line_follower.reset()
        self._goto(MissionState.TRAVEL_BACK_TO_ENTRANCE)

    def _state_travel_back(self):
        if self._consume_entry():
            # [수정] 복귀 구간 살수 여부는 설정으로 정한다.
            #   PUMP_ON_RETURN_LEG=False 이면 같은 고랑에 물을 두 번 주지 않아
            #   물통 한 통으로 커버하는 고랑 수가 2배가 된다.
            water_on_return = PUMP_ON_RETURN_LEG and not self._water_low_aborted
            self.pump.set_zone(water_on_return)
            self.line_follower.reset()
            self._begin_furrow_leg(returning=True)
            if water_on_return and not self.water_sensor.is_water_low():
                self.pump.turn_on()
            elif not water_on_return:
                self.pump.turn_off()
                log.info("복귀 구간은 살수하지 않고 이동만 합니다.")

        self.pump.set_lockout(self.water_sensor.is_water_low())

        if self._elapsed_in_state() > MAX_FURROW_TRAVEL_DURATION_SEC:
            self.motors.stop()
            self._safe_halt(
                f"복귀 주행 중 {MAX_FURROW_TRAVEL_DURATION_SEC:.0f}초 동안 "
                f"ToF 가 입구를 감지하지 못했습니다. 센서 오작동이 의심되어 정지합니다."
            )
            return

        if self._drive_in_furrow():
            log.info("고랑 입구 도달. 이탈합니다.")
            self._goto(MissionState.EXIT_FURROW)

    def _state_exit_furrow(self):
        if self._consume_entry():
            self.pump.set_zone(False)
            self.pump.turn_off()

        if self._elapsed_in_state() < EXIT_FORWARD_DURATION_SEC:
            self.motors.forward(BASE_SPEED)
            return

        self.motors.stop()
        self._reanchor_field_heading()
        self.furrow_mgr.mark_current_done()
        log.info("%d번 고랑 완료. %s", self.furrow_mgr.current_index, self.furrow_mgr.summary())
        self._goto(MissionState.EVALUATE_MISSION)

    # ------------------------------------------------------------------
    def _reanchor_heading_from_posts(self, observed):
        """
        [신규/중요] 보이는 게이트로 '밭 안쪽 방위'를 다시 잡는다.

        왜 필요한가
        -----------
        모든 회전은 엔코더로 각도를 잰다. 바퀴가 미끄러지면 엔코더는
        "돌았다"고 보고하지만 실제로는 덜 돈다. 미끄러짐이 5%면 90도
        선회가 85.5도에서 끝난다. 헤드랜드 이동 한 번에 선회가 두 번이니
        매번 9도씩 어긋난다.

        급수 중에는 고랑마다 게이트에 정렬하므로 이 오차가 리셋된다.
        그런데 **HOME 복귀 중에는 정렬을 하지 않는다**(빠르게 지나가려고
        탐색을 건너뛰기 때문). 그래서 오차를 바로잡을 기회가 없고,
        로봇이 헤드랜드에서 점점 밀려난다.

        시뮬레이션으로 재현한 결과(고랑 12개, 미끄러짐 5%):
            복귀 중 y 가 -0.85m -> -3.32m 로 밀림
            -> 마커 사거리(3m) 밖으로 나가 HOME 을 영영 못 찾음
            (x 는 정확했다. 밀린 방향이 문제였다)

        해결
        ----
        팻말이 고랑당 1개뿐이라 게이트 하나로는 각도를 구할 수 없다.
        하지만 **서로 다른 두 고랑의 입구 팻말**이 동시에 보이면,
        그 두 점을 잇는 방향이 곧 **헤드랜드 방향**이다.
        (모든 입구 팻말이 같은 오프셋으로 한 줄에 서 있으므로)

        헤드랜드 방향을 알면 밭 안쪽 방위는 90도 돌린 값이다.
        마커 두 점의 **위치**만 쓰므로 포즈 모호성의 영향을 받지 않는다.
        """
        posts = {}
        for mid, obs in observed.items():
            if 1 <= mid and mid != FIELD_END_MARKER_ID:
                posts[mid] = obs          # 입구 팻말 ID = 고랑 번호
        if HOME_MARKER_ID in observed:
            posts[1] = observed[HOME_MARKER_ID]   # HOME = 1번 고랑 입구

        if len(posts) < 2:
            return          # 한 개만 보이면 방향을 알 수 없다

        idxs = sorted(posts)
        lo, hi = idxs[0], idxs[-1]
        a, b = posts[lo], posts[hi]

        # [가드1] 두 팻말이 너무 가까이 붙어 보이면 각도 오차가 크게 증폭된다.
        dx = b.lateral_offset_m - a.lateral_offset_m
        dz = b.forward_m - a.forward_m
        if math.hypot(dx, dz) < GATE_HEADING_ANCHOR_MIN_BASELINE_M:
            return
        if min(a.forward_m, b.forward_m) > GATE_HEADING_ANCHOR_MAX_DISTANCE_M:
            return

        # 카메라 좌표(x=오른쪽, z=전방) -> 로봇 기준 방위각.
        # 고랑 번호가 커지는 방향 = 다음 고랑으로 가는 헤드랜드 방향.
        bearing_to_higher = math.atan2(dx, dz) * SIGN_MARKER_LATERAL
        heading_toward_next = normalize_angle(self.odom.theta + bearing_to_higher)

        # _headland_heading() 의 규약과 반대로 풀면 밭 안쪽 방위가 나온다.
        #   heading_toward_next = field_heading + HEADLAND_TURN_RAD
        best = (min(a.forward_m, b.forward_m),
                normalize_angle(
                    self.odom.theta
                    - normalize_angle(heading_toward_next - HEADLAND_TURN_RAD)
                ))

        new_heading = normalize_angle(self.odom.theta - best[1])
        drift = normalize_angle(new_heading - self._field_heading)

        # [가드2] 한 번에 크게 튀는 보정은 마커 오검출이거나 엉뚱한 게이트를
        #   본 것일 가능성이 크다. 무시하고 기존 값을 유지한다.
        if abs(drift) > GATE_HEADING_ANCHOR_MAX_CORRECTION_RAD:
            log.warning(
                "게이트 기반 방위 보정이 %+.0f도로 과도해 무시합니다 "
                "(허용 %.0f도). 마커 오검출이 의심됩니다.",
                math.degrees(drift),
                math.degrees(GATE_HEADING_ANCHOR_MAX_CORRECTION_RAD),
            )
            return

        self._field_heading = new_heading
        if abs(drift) > math.radians(2.0):
            log.info(
                "게이트 마커로 밭 방위를 %+.1f도 보정했습니다 "
                "(엔코더 회전 오차 제거).", math.degrees(drift),
            )

    def _reanchor_field_heading(self):
        """
        헤드랜드 이동 직전에 '밭 안쪽 방위'를 다시 잡는다.

        왜 필요한가
        -----------
        모든 회전은 엔코더로 각도를 잰다. 바퀴가 미끄러지면 엔코더는
        "돌았다"고 보고하지만 실제로는 덜 돈다. 이 오차가 고랑마다 쌓인다.
        _field_heading 은 헤드랜드 이동의 기준 방위이므로, 여기가 틀어지면
        로봇이 헤드랜드를 벗어나 밭 밖으로 나간다.

        해결
        ----
        방금 고랑을 따라 밖으로 나왔으므로 로봇은 고랑 축과 나란히 바깥을
        향하고 있다. 즉 **지금** 각도의 반대가 밭 안쪽이다.

        [핵심] 정렬 시점에 저장해 둔 값을 쓰면 안 된다. 그 사이 고랑을
        왕복하며 유턴을 했고, 미끄러짐이 있으면 그 회전마다 오차가 쌓인다.
        반드시 '지금' 값을 써야 한다.
        """
        new_heading = normalize_angle(self.odom.theta + math.pi)
        drift = normalize_angle(new_heading - self._field_heading)

        # [신규/중요] 검증 없이 채택하면 안 된다. 이 재기준은 "고랑을 따라
        #   똑바로 빠져나왔다"를 전제하는데, 비전이 로봇을 비스듬히 내보내면
        #   그 **틀린 각도까지 밭 기준이 되어** 이후 헤드랜드 이동이 통째로
        #   어긋난다(실측 중앙 드리프트 40도). 상식적인 범위를 넘으면 버린다.
        limit = math.radians(FIELD_HEADING_REANCHOR_MAX_DRIFT_DEG)
        if limit > 0 and abs(drift) > limit:
            log.warning(
                "밭 방위 재기준 %+.1f도는 상한(%.0f도)을 넘어 무시합니다. "
                "고랑을 비스듬히 빠져나왔을 가능성이 큽니다.",
                math.degrees(drift), FIELD_HEADING_REANCHOR_MAX_DRIFT_DEG,
            )
            return

        self._field_heading = new_heading
        if abs(drift) > math.radians(3.0):
            log.info(
                "고랑 축 기준으로 밭 방위를 %+.1f도 보정했습니다.",
                math.degrees(drift),
            )

    def _state_evaluate_mission(self):
        # HOME 으로 돌아갈 사유는 두 가지뿐이다.
        #   (1) END 마커를 본 고랑까지 급수를 마쳤다 = 임무 완료
        #   (2) 물이 떨어졌다
        #
        # [수정] 진입 도중 물 부족으로 중단한 경우도 HOME 사유로 인정한다.
        #   센서가 디바운스 경계에서 흔들려 이 순간 '정상'으로 읽히더라도
        #   방금 물이 떨어져 중단한 사실은 유지되어야 한다.
        if self._last_furrow_flagged:
            log.info(
                "마지막 고랑(%d번) 급수 완료 -> 임무 종료. HOME 으로 복귀합니다.",
                self.furrow_mgr.current_index,
            )
            self.target = NavigationTarget.HOME
            self._mission_finished_by_end_marker = True
        elif self.water_sensor.is_water_low() or self._water_low_aborted:
            log.info("물통 부족 확인 -> HOME 으로 복귀합니다.")
            self.target = NavigationTarget.HOME
        else:
            self.target = NavigationTarget.NEXT_FURROW

        # [주의] 플래그 초기화는 아래 조기 반환보다 **먼저** 해야 한다.
        self._water_low_aborted = False
        self._last_furrow_flagged = False
        self._search_retry_count = 0
        self._approach_start_time = None
        self._transit_count = 0

        if self.target == NavigationTarget.HOME:
            # HOME = 1번 고랑 입구. 지금 N번 고랑에 있으므로 (N-1)칸 되돌아간다.
            self._home_transits_remaining = max(
                0, self.furrow_mgr.current_index - 1
            )
            if self._home_transits_remaining == 0:
                # [수정/버그] 이미 HOME(1번 고랑) 앞에 서 있다.
                #   예전에는 이 경우에도 헤드랜드 이동을 한 번 더 해서
                #   **HOME 을 지나쳐 버렸다**. 게다가 팻말이 END 쪽으로
                #   기울어져 있으므로, HOME 을 지나친 위치(-x)에서는 마커
                #   뒷면만 보여 영영 찾지 못하고 SAFE_HALT 로 끝났다.
                log.info("이미 HOME(1번 고랑) 앞입니다. 바로 정렬합니다.")
                self._goto(MissionState.SEARCH_AND_ALIGN)
                return
            log.info(
                "HOME(1번 고랑 입구)까지 헤드랜드 %d칸을 되돌아갑니다.",
                self._home_transits_remaining,
            )
        if HEADLAND_ENABLED:
            self._start_headland_transit()
        else:
            self._goto(MissionState.SEARCH_AND_ALIGN)

    # ------------------------------------------------------------------
    # 헤드랜드 이동
    # ------------------------------------------------------------------
    def _max_transits_allowed(self) -> int:
        """
        헤드랜드 이동 허용 횟수.

        다음 고랑을 찾을 때는 옆으로 한두 칸이면 충분하므로 MAX_HEADLAND_TRANSITS
        로 충분하다. 그러나 **HOME 으로 되돌아갈 때**는 지나온 고랑 수만큼
        되돌아가야 하므로 그만큼 허용해야 한다. 고정값 3으로 두면 고랑이 4개
        이상인 밭에서 물을 다 주고도 집에 못 돌아와 SAFE_HALT 로 끝난다.

        상한은 "실제로 지나온 고랑 수 + 여유 2" 이므로, 왔던 거리보다 더 멀리
        가는 일은 없다(무한 전진 방지 원칙 유지).
        """
        if self.target == NavigationTarget.HOME:
            # 지나온 고랑 수만큼은 반드시 되돌아갈 수 있어야 한다.
            # +3 은 오도메트리 드리프트로 한두 칸 어긋났을 때의 여유.
            return max(MAX_HEADLAND_TRANSITS, self.furrow_mgr.current_index + 3)
        return MAX_HEADLAND_TRANSITS

    def _headland_heading(self) -> float:
        """
        헤드랜드를 따라 이동할 때 향해야 할 **절대 방위**.

        _field_heading(밭 안쪽 방위)을 기준으로 계산하므로, 지금 로봇이
        밭을 향하고 있든 등지고 있든 결과가 같다.
        예전처럼 '현재 각도에서 상대적으로 90도'를 두 번 도는 방식은
        시작 자세에 따라 엉뚱한 방향으로 가버릴 수 있었다.
        """
        # 밭을 마주 봤을 때 다음 고랑이 오른쪽이면 방위는 field_heading - 90도
        d = -1.0 if HEADLAND_DIRECTION.lower() == "right" else +1.0
        if self.target == NavigationTarget.HOME:
            d = -d
        return self._field_heading + d * HEADLAND_TURN_RAD

    def _start_headland_transit(self):
        self._transit_phase = TransitPhase.TURN_OUT
        self._transit_count += 1
        # 위치가 바뀌므로 다음 탐색은 1단계(회전)부터 다시 시작한다.
        self._search_rotate_done = False
        self._search_creep_exhausted = False
        self._goto(MissionState.HEADLAND_TRANSIT)

    def _state_headland_transit(self):
        """
        헤드랜드 방위로 선회 -> 고랑 간격만큼 이동 -> 다시 밭 안쪽으로 선회.
        회전은 모두 **절대 방위 목표**로 계산하므로 시작 자세와 무관하다.
        모든 단계에 거리/시간 상한이 있다.
        """
        # [신규] 헤드랜드를 옆으로 지나가는 동안은 여러 고랑의 팻말이 한 화면에
        #   들어올 가능성이 가장 높다. IMU 가 없는 구성에서 밭 기준 방위를
        #   절대값으로 다시 잡을 수 있는 몇 안 되는 기회이므로 여기서도 시도한다.
        #   (선회 오차가 누적되는 바로 그 구간이라 효과가 크다)
        if HEADLAND_ANCHOR_HEADING:
            self._reanchor_heading_from_posts(self._safe_detect())
        # [신규/방어] 어떤 분기에도 걸리지 않는 상태를 그냥 두면 아무 일도
        #   하지 않는 무한 루프가 된다. 실제로 이 자리에서 NameError 가
        #   매 틱 발생했는데, 다음 틱에는 예외가 안 나서 연속 예외 카운터가
        #   리셋되는 바람에 **탐지되지도 않고 영원히 멈춰 있었다.**
        #   이제는 명시적으로 복구를 시도하고, 그래도 안 되면 정지한다.
        if self._transit_phase is None:
            log.error(
                "헤드랜드 이동 단계가 비어 있습니다(내부 오류). 이동을 다시 시작합니다."
            )
            if self._transit_count < self._max_transits_allowed():
                self._start_headland_transit()
            else:
                self._safe_halt("헤드랜드 이동 단계 복구 실패")
            return

        if self._transit_phase == TransitPhase.TURN_OUT:
            self.pump.set_zone(False)
            delta = normalize_angle(self._headland_heading() - self.odom.theta)
            self.motors.turn_by_angle_blocking(delta)
            self._transit_phase = TransitPhase.DRIVE
            self._transit_start_len = self.odom.path_length
            self._transit_start_time = time.monotonic()
            self._transit_start_ticks = self.odom.total_ticks
            return

        if self._transit_phase == TransitPhase.DRIVE:
            travelled = self.odom.path_length - self._transit_start_len
            elapsed = time.monotonic() - self._transit_start_time

            # [수정] 이동 중에는 END 마커를 판정하지 않는다.
            # 헤드랜드를 옆으로 지나가는 동안에는 멀리 있는 END 마커가
            # 비스듬히 보일 수 있어, 아직 남은 고랑이 있는데도 "모든 고랑 완료"로
            # 오판하게 된다(시뮬레이션에서 실제로 재현된 버그).
            # END 판정은 이동을 마치고 밭을 정면으로 바라보는
            # SEARCH_AND_ALIGN 에서만 한다.

            if elapsed > MAX_HEADLAND_DURATION_SEC:
                self.motors.stop()
                self._safe_halt(
                    f"헤드랜드 이동이 {MAX_HEADLAND_DURATION_SEC:.0f}초를 초과했습니다. "
                    f"엔코더 또는 구동계 이상이 의심되어 정지합니다."
                )
                return

            # [수정] 엔코더 생존 판정을 '이번 이동 중에 틱이 들어왔는가'로 바꾼다.
            # 예전에는 누적 total_ticks > 0 만 봤기 때문에, 임무 도중 엔코더가
            # 죽어도 '살아 있음'으로 판정되어 거리 기준을 계속 기다리다가
            # MAX_HEADLAND_DURATION_SEC 타임아웃 -> SAFE_HALT 로 끝났다.
            # 이제는 시간 기반 폴백으로 자연스럽게 넘어간다.
            ticks_now = self.odom.total_ticks - self._transit_start_ticks
            encoder_alive = ticks_now > 0 or elapsed < 1.0
            done = (
                travelled >= HEADLAND_STEP_DISTANCE_M
                if encoder_alive
                else elapsed >= HEADLAND_FALLBACK_DURATION_SEC
            )
            if done:
                self.motors.stop()
                self._transit_phase = TransitPhase.TURN_IN
                return

            self.motors.forward(BASE_SPEED * HEADLAND_SPEED_RATIO)
            return

        if self._transit_phase == TransitPhase.TURN_IN:
            delta = normalize_angle(self._field_heading - self.odom.theta)
            self.motors.turn_by_angle_blocking(delta)
            self.motors.stop()
            self._transit_phase = None
            self._search_retry_count = 0
            self._approach_start_time = None

            # 방위 재보정.
            # [수정] 팻말이 고랑당 1개뿐이라, 정거장에서 팻말 2개가 동시에
            #   보이는 일이 거의 없다(카메라 수평화각 62도 기준으로 옆 고랑
            #   팻말은 시야 밖). 그래서 두 팻말로 헤드랜드 방향을 재는 방식은
            #   실질적으로 성립하지 않는다.
            #   대신 고랑을 빠져나온 시점의 실제 헤딩으로 재보정한다
            #   (_reanchor_field_heading, EXIT_FURROW 에서 수행).
            self._reanchor_heading_from_posts(self._safe_detect())

            # ---------------- HOME 복귀 중의 중간 정거장 ----------------
            # [수정/성능·안정성] 예전에는 매 정거장에서 SEARCH_AND_ALIGN 으로
            #   들어가 HOME 마커를 360도 전체 탐색했다. 그런데 5번 고랑
            #   자리에서 1번 고랑의 HOME 마커가 보일 리가 없다. 즉 **있지도
            #   않은 마커를 찾느라** 정거장마다 수십 초를 버렸다.
            #   (고랑 12개 밭에서 측정: HOME 복귀가 전체 임무 시간의 49%)
            #
            #   게다가 헛도는 동안 다른 고랑의 마커를 HOME 으로 오검출하면
            #   엉뚱한 곳에서 멈춘다.
            #
            #   로봇은 자기가 몇 번째 고랑에 있는지 알고 있다. 돌아갈 칸 수를
            #   세어서, 도착 예정 지점 전까지는 전체 탐색을 건너뛰고 이동만
            #   반복한다. 대신 매 정거장에서 **한 번만** 마커를 훑어보고,
            #   예상보다 일찍 HOME 이 보이면 즉시 정렬한다(오도메트리가 실제보다
            #   많이 갔다고 착각한 경우의 보정).
            if (
                self.target == NavigationTarget.HOME
                and self._home_transits_remaining > 0
            ):
                # --- 마커로 현재 위치를 절대 보정 ---
                # 팻말이 30도로 기울어져 있어 헤드랜드에서도 게이트 마커가
                # 보인다. 보이면 칸 수를 다시 계산해 엔코더 드리프트를 지운다.
                observed = self._safe_detect()
                idx = self._localize_from_markers(observed)
                # [중요] 위치뿐 아니라 **방위**도 마커로 다시 잡는다.
                # 복귀 중에는 게이트 정렬을 하지 않으므로, 이걸 빼면
                # 회전 오차를 바로잡을 기회가 한 번도 없다.
                self._reanchor_heading_from_posts(observed)
                if idx is not None:
                    corrected = max(0, idx - 1)
                    if corrected != self._home_transits_remaining:
                        log.info(
                            "마커로 위치 확인: %d번 고랑 앞. "
                            "남은 칸을 %d -> %d 로 보정합니다.",
                            idx, self._home_transits_remaining, corrected,
                        )
                    self._home_transits_remaining = corrected

                if self._home_transits_remaining <= 0 or self._home_markers_in_sight():
                    log.info("HOME 도착 지점으로 판단. 게이트에 정렬합니다.")
                    self._home_transits_remaining = 0
                else:
                    self._home_transits_remaining -= 1
                    if self._transit_count < self._max_transits_allowed():
                        log.debug(
                            "HOME 까지 %d칸 남음 - 탐색을 건너뛰고 계속 이동합니다.",
                            self._home_transits_remaining,
                        )
                        self._start_headland_transit()
                        return
                    log.warning(
                        "HOME 복귀 이동 상한(%d회)에 도달했습니다. "
                        "이 자리에서 HOME 마커를 탐색합니다.",
                        self._max_transits_allowed(),
                    )
                    self._home_transits_remaining = 0

            self._goto(MissionState.SEARCH_AND_ALIGN)
            return

    def _home_markers_in_sight(self) -> bool:
        """
        지금 자리에서 HOME 게이트(1번 고랑 입구)가 실제로 보이는가.

        ArUco 검출 1회만 수행하는 값싼 확인이다. 전체 탐색(회전 포함)과
        달리 로봇을 움직이지 않는다.
        """
        from sensors.aruco_detector import compute_post_bearing

        observed = self._safe_detect()
        post = observed.get(HOME_MARKER_ID)
        if post is None:
            return False
        return compute_post_bearing(post).valid

    def _safe_detect(self):
        try:
            return self.aruco.detect()
        except Exception as exc:
            log.warning("마커 검출 실패: %s", exc)
            return {}

    def _localize_from_markers(self, observed):
        """
        [신규] 보이는 마커로 **지금 몇 번 고랑 앞에 있는지** 알아낸다.

        팻말을 END 쪽으로 30도 기울여 박았기 때문에, 헤드랜드를 따라
        복귀하는 중에도 각 고랑의 입구 팻말 마커가 보인다.
        입구 팻말 ID = 고랑 번호이므로, 마커 3 이 보이면 지금 3번 고랑
        앞이라는 뜻이다.

        이것이 이 설계의 핵심이다. 엔코더로 칸 수를 세는 추측항법은 바퀴가
        미끄러질 때마다 오차가 **누적**되고, 로봇은 자기가 틀렸다는 것조차
        모른다. 마커는 밭에 물리적으로 박힌 절대 기준점이므로, 볼 때마다
        위치가 **리셋**된다. 누적 오차가 원천적으로 생기지 않는다.

        반환: 고랑 번호(1부터). 판단 불가면 None.
              HOME 마커(0)를 봤으면 1을 반환한다(HOME = 1번 고랑 입구).
        """
        if not HOME_RETURN_MARKER_LOCALIZATION:
            return None

        # HOME 마커가 보이면 그것이 가장 확실한 근거
        if HOME_MARKER_ID in observed:
            return 1

        # 그 외 입구 팻말 마커: id N -> 고랑 번호 N
        best_idx, best_dist = None, float("inf")
        for mid, obs in observed.items():
            # 입구 팻말 ID = 고랑 번호. END(249)는 고랑 번호가 아니므로 제외.
            if mid < 1 or mid == FIELD_END_MARKER_ID:
                continue
            idx = furrow_index_from_marker(mid)
            # 여러 개가 보이면 가장 가까운 것을 신뢰한다
            if obs.distance_m < best_dist:
                best_idx, best_dist = idx, obs.distance_m
        return best_idx

    # ------------------------------------------------------------------
    def _state_home_arrived(self):
        self.motors.stop()
        self.pump.set_zone(False)
        self.pump.turn_off()

        if self._mission_finished_by_end_marker:
            log.info(
                "마지막 고랑까지 급수를 마치고 HOME 복귀 완료. %s",
                self.furrow_mgr.summary(),
            )
        elif self.water_sensor.is_water_low():
            log.info("물통 부족으로 HOME 복귀 완료. %s", self.furrow_mgr.summary())
            log.info("물을 보충한 뒤 다시 실행하면 이어서 급수할 수 있습니다.")
        else:
            log.info("HOME 복귀 완료. %s", self.furrow_mgr.summary())
        self._goto(MissionState.MISSION_COMPLETE)

    # ------------------------------------------------------------------
    def _monitor_vision_fallback(self, result):
        if result.using_vision:
            self._vision_fallback_since = None
            return

        now = time.monotonic()
        if self._vision_fallback_since is None:
            self._vision_fallback_since = now
            return

        elapsed = now - self._vision_fallback_since

        if VISION_FALLBACK_HALT_SEC > 0 and elapsed > VISION_FALLBACK_HALT_SEC:
            self.motors.stop()
            self._safe_halt(
                f"비전 신뢰도 부족으로 {VISION_FALLBACK_HALT_SEC:.0f}초 이상 "
                f"ToF/엔코더 대체 조향이 지속되었습니다 "
                f"(vision_confidence={result.vision_confidence:.2f}). "
                f"카메라 고장 또는 SOIL_HSV 임계값 문제가 의심됩니다."
            )
            return

        if elapsed > VISION_FALLBACK_WARN_SEC:
            log.warning(
                "비전 신뢰도 부족으로 %.0f초 이상 대체 조향 중입니다 "
                "(confidence=%.2f). 조명/흙색 임계값(SOIL_HSV_*) 점검을 권장합니다.",
                elapsed, result.vision_confidence,
            )
            self._vision_fallback_since = now - VISION_FALLBACK_WARN_SEC / 2.0

    def _telemetry(self, result):
        if TELEMETRY_EVERY_N_TICKS <= 0 or self._tick % TELEMETRY_EVERY_N_TICKS != 0:
            return
        log.debug(
            "L=%.0fmm R=%.0fmm err=%+.2f steer=%+.2f vision=%s(%.2f) "
            "centered=%s theta=%+.2frad",
            result.left_mm, result.right_mm, result.error, result.steer,
            "Y" if result.using_vision else "N", result.vision_confidence,
            "Y" if result.tof_centered else "N", self.odom.theta,
        )


if __name__ == "__main__":
    MissionStateMachine().run_forever()

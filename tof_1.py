import math
import time
import config
from actuators.motor_driver import MotorDriver
from sensors.tof_sensor import ToFPair
from sensors.odometry import Odometry


class TOFPinpointer:
    """ToFPair, Odometry, MotorDriver 기반 고랑 중앙 핀포인팅 클래스"""

    def __init__(
        self,
        motor: MotorDriver,
        tof_pair: ToFPair,
        odom: Odometry,
        threshold_a: float,
        side: str = "right",
    ):
        self.motor = motor
        self.tof_pair = tof_pair
        self.odom = odom
        self.threshold_a = threshold_a  # 단차 임계값 (mm)
        self.side = side  # 감지할 측면 ('right' 또는 'left')

        # 속도 설정
        self.search_speed = getattr(config, "BASE_SPEED", 0.2)
        self.reverse_speed = getattr(config, "REVERSE_SPEED", 0.15)

        # 1차원 이동 거리 산출용 기준 좌표
        self.x0 = 0.0
        self.y0 = 0.0
        self.theta0 = 0.0

        # 상태 변수 및 위치 기록값 (m 단위)
        self.state = "STATE_FIND_ENTRY"
        self.S1 = 0.0
        self.S2 = 0.0
        self.S_target = 0.0

    def reset(self):
        """핀포인터 시작 시 기준점 및 상태 초기화"""
        self.state = "STATE_FIND_ENTRY"
        self.x0 = self.odom.x
        self.y0 = self.odom.y
        self.theta0 = self.odom.theta
        self.S1 = 0.0
        self.S2 = 0.0
        self.S_target = 0.0

    def _get_displacement(self) -> float:
        """시작 기준점 대비 주행 방향으로의 부호 있는 이동 거리 S(m) 계산"""
        dx = self.odom.x - self.x0
        dy = self.odom.y - self.y0
        # 주행 방향 사영 (전진 시 증가, 후진 시 감소)
        return dx * math.cos(self.theta0) + dy * math.sin(self.theta0)

    def _get_side_raw_distance(self) -> float:
        """지연 오차 방지를 위한 Raw 거리 값 반환"""
        sensor = (
            self.tof_pair.right if self.side == "right" else self.tof_pair.left
        )
        return sensor.last_raw_mm

    def update(self) -> bool:
        """
        메인 제어 루프에서 매 틱(Tick)마다 호출되는 함수
        :return: 핀포인팅 완료 및 회전 위치 정지 시 True 반환
        """
        # 1. 센서 및 오도메트리 매 틱 갱신 (필수)
        self.odom.update()
        self.tof_pair.read()

        d_raw = self._get_side_raw_distance()
        S = self._get_displacement()

        # ----------------------------------------------------
        # [1단계] 고랑 진입점 감지 (d > a)
        # ----------------------------------------------------
        if self.state == "STATE_FIND_ENTRY":
            self.motor.drive(base=self.search_speed, steer=0.0)
            if d_raw > self.threshold_a:
                self.S1 = S
                print(f"[Pinpointer] 진입 감지! S1 = {self.S1:.3f}m (Raw: {d_raw:.1f}mm)")
                self.state = "STATE_FIND_EXIT"

        # ----------------------------------------------------
        # [2단계] 고랑 탈출점 감지 (d <= a)
        # ----------------------------------------------------
        elif self.state == "STATE_FIND_EXIT":
            self.motor.drive(base=self.search_speed, steer=0.0)
            if d_raw <= self.threshold_a:
                self.motor.drive(base=0.0, steer=0.0)  # 즉시 정지
                self.S2 = S
                print(f"[Pinpointer] 탈출 감지! S2 = {self.S2:.3f}m (Raw: {d_raw:.1f}mm)")
                self.state = "STATE_CALCULATE"

        # ----------------------------------------------------
        # [3단계] 고랑 중앙 위치 연산
        # ----------------------------------------------------
        elif self.state == "STATE_CALCULATE":
            width = self.S2 - self.S1
            self.S_target = (self.S1 + self.S2) / 2.0
            print(f"[Pinpointer] 연산 완료 - 고랑 폭: {width:.3f}m, 목표 S_target = {self.S_target:.3f}m")
            self.state = "STATE_REVERSE"

        # ----------------------------------------------------
        # [4단계] 고랑 중앙으로 후진 (S <= S_target)
        # ----------------------------------------------------
        elif self.state == "STATE_REVERSE":
            # base에 음수를 인가하여 후진
            self.motor.drive(base=-self.reverse_speed, steer=0.0)
            if S <= self.S_target:
                self.motor.drive(base=0.0, steer=0.0)  # 즉시 정지
                print(f"[Pinpointer] 고랑 중앙 정지 완료! (현재 S = {S:.3f}m)")
                self.state = "STATE_ROTATE"

        # ----------------------------------------------------
        # [5단계] 정지 및 회전 준비 완료
        # ----------------------------------------------------
        elif self.state == "STATE_ROTATE":
            return True

        return False

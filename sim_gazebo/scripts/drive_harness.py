# -*- coding: utf-8 -*-
"""
sim_gazebo/scripts/drive_harness.py
------------------------------------
Gazebo 안에서 **주행 시퀀스**를 여러 시나리오로 돌리는 하네스.
컨테이너 안에서 실행한다(gz-transport 파이썬 바인딩 사용).

무엇을 검증하는가 (2D 시뮬이 답하지 못한 것)
  1. **마커가 실제로 검출되는가** — cv2.aruco 를 렌더 프레임에 직접 돌린다
  2. **비전이 실패해도 이랑에 부딪히지 않고 고랑을 주행하는가** — 실제 ToF
  3. 고랑 끝 유턴 -> 복귀
  4. HOME 복귀

비전 모델
  요청대로 **실측 CRDLD 오차 분포에 가까운 랜덤**으로 가정한다.
  (reports/crdld_furrow_v1/metrics_frames.csv 의 통계를 근사한 분포)
  vision_mode: "measured" | "blind"

제어
  farm_robot 의 LineFollower 정책을 그대로 옮긴 최소 구현:
    비전 신뢰도 비례 가중치 + ToF 거부권 + ToF 폴백
  (전체 FSM 을 옮기지 않는 이유: 여기서 보려는 것은 '고랑 주행'과 '마커 검출'
   이지 임무 상태 전이가 아니다. 상태 전이는 2D 시뮬이 이미 검증한다)

    python3 /sim/scripts/drive_harness.py --scenario furrow_drive --vision measured
"""

import argparse
import math
import random
import sys
import time

import cv2
import numpy as np
from gz.msgs10.laserscan_pb2 import LaserScan
from gz.msgs10.image_pb2 import Image
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node

# --- 밭/로봇 상수 (make_world.py 와 같은 값) ---
SPACING = 1.0
FURROW_HALF = 0.20
BODY_HALF = 0.10
MARGIN = FURROW_HALF - BODY_HALF        # 0.10m 넘으면 이랑 밟음
TOF_MAX = 0.80

# --- 비전 오차 모델 (CRDLD 실측 통계 근사) ---
#   실측: 각도오차 중앙 1.26deg / p95 28.4deg, conf 중앙 0.67 / 최대 0.73
#   오답(큰 오차)의 conf 가 오히려 높다는 성질을 재현한다.
VIS_GOOD_RATE = 0.78          # 정상 프레임 비율
VIS_ERR_GOOD = 0.03           # 정상 프레임의 횡오차(무차원, 고랑 반폭 기준)
VIS_ERR_BAD = 0.9             # 오답 프레임의 횡오차
VIS_CONF_RANGE = (0.55, 0.73)


class Harness:
    def __init__(self, vision_mode="measured", seed=0, gate=0.25,
                 veto=0.15, tof_weight=0.25):
        self.node = Node()
        self.rng = random.Random(seed)
        self.vision_mode = vision_mode
        self.gate, self.veto, self.tof_w = gate, veto, tof_weight

        self.frame = None
        self.tof_l = float("inf")
        self.tof_r = float("inf")
        self.pose = None                     # (x, y, yaw)

        self.node.subscribe(Image, "/camera", self._on_image)
        self.node.subscribe(LaserScan, "/tof_left", self._on_tof_l)
        self.node.subscribe(LaserScan, "/tof_right", self._on_tof_r)
        self.pub = self.node.advertise("/cmd_vel", Twist)

        d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250) \
            if hasattr(cv2.aruco, "Dictionary_get") \
            else cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self._dict = d
        p = cv2.aruco.DetectorParameters_create() \
            if hasattr(cv2.aruco, "DetectorParameters_create") \
            else cv2.aruco.DetectorParameters()
        # 저해상도 대응 튜닝 (실기 aruco_detector.py 와 동일)
        p.minMarkerPerimeterRate = 0.01
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 15
        p.adaptiveThreshWinSizeStep = 2
        p.perspectiveRemovePixelPerCell = 8
        p.polygonalApproxAccuracyRate = 0.05
        self._params = p

        self.stats = {"ticks": 0, "marker_frames": 0, "markers": set(),
                      "vision_used": 0, "vision_vetoed": 0, "tof_only": 0,
                      "ridge_hits": 0, "max_dev": 0.0}

    # ---------------- 콜백 ----------------
    def _on_image(self, msg):
        try:
            a = np.frombuffer(msg.data, np.uint8)
            self.frame = a[: msg.width * msg.height * 3].reshape(
                msg.height, msg.width, 3)
        except Exception:
            pass

    def _on_tof_l(self, msg):
        if len(msg.ranges):
            self.tof_l = msg.ranges[0]

    def _on_tof_r(self, msg):
        if len(msg.ranges):
            self.tof_r = msg.ranges[0]

    # ---------------- 센서 해석 ----------------
    def detect_markers(self):
        if self.frame is None:
            return []
        gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        if hasattr(cv2.aruco, "ArucoDetector"):
            det = cv2.aruco.ArucoDetector(self._dict, self._params)
            _, ids, _ = det.detectMarkers(gray)
        else:
            _, ids, _ = cv2.aruco.detectMarkers(gray, self._dict,
                                                parameters=self._params)
        return sorted(int(i) for i in ids.flatten()) if ids is not None else []

    def tof_error(self):
        """좌우 벽 대칭 오차(무차원). 한쪽이라도 사거리 밖이면 None."""
        l, r = self.tof_l, self.tof_r
        if not (0 < l < TOF_MAX) or not (0 < r < TOF_MAX):
            return None
        return (r - l) / FURROW_HALF        # 양수 = 오른쪽 여유 많음 = 오른쪽으로

    def vision_error(self):
        """실측 분포를 근사한 (오차, 신뢰도). blind 면 (None, 0)."""
        if self.vision_mode == "blind":
            return None, 0.0
        conf = self.rng.uniform(*VIS_CONF_RANGE)
        if self.rng.random() < VIS_GOOD_RATE:
            err = self.rng.gauss(0.0, VIS_ERR_GOOD)
        else:
            # 오답: 큰 오차인데 conf 는 여전히 높다 (실측에서 확인된 성질)
            err = self.rng.choice((-1, 1)) * abs(self.rng.gauss(VIS_ERR_BAD, 0.3))
        return err, conf

    # ---------------- 제어 ----------------
    def fuse(self):
        """LineFollower 정책 축약: 신뢰도 비례 + ToF 거부권 + ToF 폴백."""
        v_err, conf = self.vision_error()
        t_err = self.tof_error()
        use_vision = v_err is not None and conf >= self.gate

        if use_vision and t_err is not None and self.veto > 0:
            if abs(v_err - t_err) > self.veto:
                use_vision = False
                self.stats["vision_vetoed"] += 1

        if use_vision:
            if t_err is not None:
                w = min(1.0, max(0.0, (conf - self.gate) / (0.60 - self.gate)))
                w *= (1.0 - self.tof_w)
                err = w * v_err + (1.0 - w) * t_err
            else:
                err = v_err
            self.stats["vision_used"] += 1
        elif t_err is not None:
            err = t_err
            self.stats["tof_only"] += 1
        else:
            err = 0.0
        return max(-1.5, min(1.5, err))

    def send(self, lin, ang):
        m = Twist()
        m.linear.x = lin
        m.angular.z = ang
        self.pub.publish(m)

    # ---------------- 시나리오 ----------------
    def run_furrow_drive(self, seconds=25.0, speed=0.22, kp=1.6):
        """고랑을 따라 주행하며 이랑 충돌 여부를 본다."""
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.stats["ticks"] += 1
            ids = self.detect_markers()
            if ids:
                self.stats["marker_frames"] += 1
                self.stats["markers"].update(ids)

            err = self.fuse()
            # 각속도 상한: 비전 오답(무차원 ±0.9)이 그대로 들어오면 회전이
            # 폭주해 고랑을 벗어난다. 실기 MAX_STEER_CORRECTION 에 해당.
            ang = max(-0.8, min(0.8, -kp * err))
            self.send(speed, ang)            # 양수 오차 = 오른쪽 = 시계(-z)

            # 이랑 밟음 판정: ToF 로 추정한 중심 이탈
            if 0 < self.tof_l < TOF_MAX and 0 < self.tof_r < TOF_MAX:
                dev = abs(self.tof_r - self.tof_l) / 2.0
                self.stats["max_dev"] = max(self.stats["max_dev"], dev)
                if dev > MARGIN:
                    self.stats["ridge_hits"] += 1
            time.sleep(0.05)
        self.send(0.0, 0.0)

    def run_marker_scan(self, seconds=14.0, omega=0.5):
        """제자리 회전하며 몇 개의 마커를 실제로 검출하는지 본다."""
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.stats["ticks"] += 1
            ids = self.detect_markers()
            if ids:
                self.stats["marker_frames"] += 1
                self.stats["markers"].update(ids)
            self.send(0.0, omega)
            time.sleep(0.05)
        self.send(0.0, 0.0)

    def report(self, label):
        s = self.stats
        t = max(1, s["ticks"])
        print(f"[{label}]")
        print(f"  틱 {s['ticks']}  |  마커 검출 프레임 {s['marker_frames']} "
              f"({s['marker_frames']*100//t}%)  검출 ID {sorted(s['markers'])}")
        print(f"  비전 채택 {s['vision_used']}  거부권 {s['vision_vetoed']}  "
              f"ToF단독 {s['tof_only']}")
        print(f"  최대 중심이탈 {s['max_dev']*100:.1f}cm (여유 {MARGIN*100:.0f}cm)"
              f"  이랑밟음 {s['ridge_hits']}틱")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="furrow_drive",
                    choices=["furrow_drive", "marker_scan"])
    ap.add_argument("--vision", default="measured", choices=["measured", "blind"])
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--veto", type=float, default=0.15)
    args = ap.parse_args()

    h = Harness(vision_mode=args.vision, seed=args.seed, veto=args.veto)
    time.sleep(3.0)      # 첫 센서 프레임 대기
    if h.frame is None:
        print("경고: 카메라 프레임을 못 받았습니다", file=sys.stderr)
    if args.scenario == "furrow_drive":
        h.run_furrow_drive(seconds=args.seconds)
    else:
        h.run_marker_scan(seconds=args.seconds)
    h.report(f"{args.scenario} / vision={args.vision} / veto={args.veto}")


if __name__ == "__main__":
    main()

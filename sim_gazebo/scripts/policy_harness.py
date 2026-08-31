# -*- coding: utf-8 -*-
"""
sim_gazebo/scripts/policy_harness.py
-------------------------------------
**정책 스윕용** 하네스. drive_harness.py 가 "고랑 하나를 잘 달리는가"를 봤다면,
여기서는 시작 위치 / 팻말 각도 / 주행 시퀀스(정책)를 바꿔가며
**어떤 정책이 가장 안정적인가**를 찾는다.

정책 = (탐색 방식) x (진입 방식)
    탐색 rotate : 제자리 회전
         sweep  : 시작 헤딩 +-55도 왕복 (회전 누적 오차를 줄인다)
         creep  : 밭 안쪽으로 천천히 전진하며 탐색 (거리를 줄여 검출률을 올린다)
         back   : 뒤로 물러난 뒤 회전 탐색.
                  [관측] 고랑 입구 1m 앞에서는 좌우 팻말이 화각(62도) **가장자리**
                  에 걸려 검출되지 않는다(x=0,y=-1 에서 검출 0건, 카메라 이미지
                  out/view_x0.png 로 확인). 물러나면 각도가 줄어 둘 다 들어온다.
    진입 survey : 측량값(이랑 간격)을 안다는 전제.
                  "팻말에서 간격/2 만큼 옆" 을 진입 목표점으로 계산한다.
         tof    : 측량값을 안 쓴다. 팻말 근처까지 간 뒤 밭 방향으로 서서
                  전진/후진+횡이동을 반복하며 ToF 로 좌우 벽을 찾는다.
         mid    : **팻말 2개의 중점**을 진입 목표로 삼는다 + 방위도 그 2개로.
                  고랑은 이랑 k 와 이랑 k+1 **사이**이므로, 인접한 두 팻말의
                  중점이 곧 고랑 중심이다. 측량값(간격)을 전혀 쓰지 않으면서
                  "팻말 + 간격/2" 보다 정확하다 -- 거리 추정 오차가 두 팻말에
                  같은 방향으로 실려 중점에서 상당 부분 상쇄되기 때문이다.
                  [관측] survey 는 목표를 10cm 빗나가 이랑 끝 모서리에 걸려
                  로봇이 180도 돌아가 버렸다(참값 헤딩 91->180도).
         pair   : survey + **팻말 2개로 밭 방위를 다시 잡는다**.
                  [관측] 시작 헤딩이 20도 틀어지면 다른 모든 정책이 0/16 으로
                  전멸했다. 출발 자세를 밭 방향으로 가정하기 때문이다.
                  팻말 2개를 이으면 그 선이 곧 헤드랜드 방향이므로, 수직이
                  밭 방향이다. **단일 마커 yaw 추정(기각됨)을 쓰지 않는다** --
                  베어링과 거리만 쓰므로 실제 검출기로도 성립한다.

★ 설계 원칙: 로봇은 **월드 좌표를 모른다**.
  제어에 쓰는 자세는 명령속도를 적분한 데드레커닝(=엔코더 오도메트리)뿐이다.
  월드 참값(/world/field/dynamic_pose/info)은 **채점에만** 쓴다.
  이걸 섞으면 2D 시뮬에서 겪은 '순환 검증'을 그대로 반복하게 된다.

밭 규약 (make_world.py 와 동일)
    이랑 k 중심 x = (k-0.5)*spacing, 팻말 k 가 그 위에 있다
    고랑 k 중심 x = (k-1)*spacing    (이랑 k-1 과 이랑 k 사이)
    -> 팻말 m 을 보면 **그 이랑을 지나친** 고랑 m+1 로 들어간다
    고랑 길이 y = 0 ~ 6m, 헤드랜드는 y < 0

    python3 /sim/scripts/policy_harness.py --search rotate --entry survey \
        --start-x -0.9 --start-y -1.0 --stage entry
"""

import argparse
import json
import math
import random
import sys
import time

import cv2
import numpy as np
from gz.msgs10.laserscan_pb2 import LaserScan
from gz.msgs10.image_pb2 import Image
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.odometry_pb2 import Odometry
from gz.transport13 import Node

# ---- 밭/로봇 상수 ----
SPACING = 1.0
FURROW_HALF = 0.20          # 고랑 반폭 (이랑 밑면 0.6 -> 틈 0.4m)
BODY_HALF = 0.09            # 차체 폭 18cm (실기 확정). 측정 당시는 20cm 였다.
MARGIN = FURROW_HALF - BODY_HALF     # 0.11m 넘으면 이랑 밟음 (20cm 였을 땐 0.10m)
FIELD_LEN = 6.0
TOF_MAX = 0.80
CAM_W, CAM_H, HFOV = 640, 480, math.radians(62.0)
FX = (CAM_W / 2.0) / math.tan(HFOV / 2.0)

# ---- 비전 오차 모델 (CRDLD 실측 근사, drive_harness.py 와 동일) ----
VIS_GOOD_RATE = 0.78
VIS_ERR_GOOD = 0.03
VIS_ERR_BAD = 0.9
VIS_CONF_RANGE = (0.55, 0.73)

DT = 0.05                   # 20Hz 제어 주기

# 밭에 실제로 존재하는 고랑 팻말 ID 상한. 이보다 큰 ID 는 오검출로 본다.
#   [관측] 실제 검출기가 이랑 텍스처에서 ID 190 같은 **가짜 마커**를 만들어
#   냈고, 그것으로 밭 방위를 잡으면 오차가 +20.6도까지 벌어졌다.
#   (END 팻말 249 는 진입 후보/방위 기준에서 제외된다)
MAX_FURROW_MARKER_ID = 20


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class PolicyHarness:
    def __init__(self, args):
        self.a = args
        self.rng = random.Random(args.seed)
        self.node = Node()

        self.frame = None
        self.tof_l = float("inf")
        self.tof_r = float("inf")
        self.truth = None                      # (x, y, yaw) 월드 참값 = 채점 전용

        self.node.subscribe(Image, "/camera", self._on_image)
        self.node.subscribe(LaserScan, "/tof_left", self._on_tof_l)
        self.node.subscribe(LaserScan, "/tof_right", self._on_tof_r)
        self.node.subscribe(Pose_V, "/world/field/dynamic_pose/info", self._on_pose)
        # /odom = DiffDrive 가 **바퀴 조인트 회전**으로 만든 오도메트리
        #   = 실기의 엔코더 오도메트리와 같은 정보(미끄러지면 그대로 틀어진다).
        self.node.subscribe(Odometry, "/odom", self._on_odom)
        self.pub = self.node.advertise("/cmd_vel", Twist)

        # OpenCV 4.6(컨테이너)은 ArucoDetector 클래스가 없다 -> 구 API 병행
        self._dict = (cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
                      if hasattr(cv2.aruco, "Dictionary_get")
                      else cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250))
        p = (cv2.aruco.DetectorParameters_create()
             if hasattr(cv2.aruco, "DetectorParameters_create")
             else cv2.aruco.DetectorParameters())
        p.minMarkerPerimeterRate = 0.01
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 15
        p.adaptiveThreshWinSizeStep = 2
        p.perspectiveRemovePixelPerCell = 8
        p.polygonalApproxAccuracyRate = 0.05
        self._params = p
        self._det = (cv2.aruco.ArucoDetector(self._dict, p)
                     if hasattr(cv2.aruco, "ArucoDetector") else None)

        # 제어가 쓰는 자세는 **엔코더 오도메트리뿐**이다.
        # 시작 자세를 원점, 밭 방향을 +y 로 가정한다(사용자 전제: 밭을 보고 출발).
        #   [수정] 예전에는 명령속도를 적분했는데, 트랙이 지면에 끌리면서
        #   명령 -98도 회전이 실제 -25도로 나오는 등 **4배 과대평가**되었다.
        #   /odom(바퀴 회전 기반)을 쓰면 그 오차가 실기와 같은 방식으로 들어온다.
        self._odom_raw = None
        self._odom0 = None
        self.od = [0.0, 0.0, math.radians(90.0)]
        self.cmd = (0.0, 0.0)
        # 밭 방향(od 프레임). 기본값은 '출발 시 밭을 보고 있다'는 가정.
        self.field_th = math.radians(90.0)
        self._mpos = {}                 # id -> od 좌표 (가장 최근 관측)

        self.r = {
            "search": args.search, "entry": args.entry,
            "tilt": args.tilt, "marker_cm": args.marker_cm,
            "start": [args.start_x, args.start_y, args.start_yaw],
            "seed": args.seed, "stage": args.stage,
            "ticks": 0, "detect_frames": 0, "ids": [],
            "t_first_marker": None, "first_marker_id": None,
            "t_entered": None, "entered_furrow": None,
            "entry_ok": False, "max_dev_cm": 0.0, "ridge_hit_ticks": 0,
            "uturn_ok": False, "returned_ok": False,
            "home_err_cm": None, "drive_len_m": 0.0,
            "drive_len_out": 0.0, "drive_len_back": 0.0,
            "vision_used": 0, "vision_vetoed": 0, "tof_only": 0,
            "fail": "",
        }
        self._ids_seen = set()

    # ------------------------------------------------ 콜백
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

    def _on_odom(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        self._odom_raw = (p.x, p.y, yaw)

    def _update_od(self):
        """/odom 을 '시작 자세 = 원점, 밭 방향 = +y' 인 od 프레임으로 옮긴다."""
        if self._odom_raw is None:
            return
        if self._odom0 is None:
            self._odom0 = self._odom_raw
        x, y, th = self._odom_raw
        x0, y0, th0 = self._odom0
        dx, dy = x - x0, y - y0
        c, s = math.cos(-th0), math.sin(-th0)
        rx = (c * dx - s * dy) * (1.0 + self.a.slip)   # 엔코더 스케일 오차
        ry = (s * dx + c * dy) * (1.0 + self.a.slip)
        # 로봇 전방을 od 의 +y 로 (전방=+y, 좌측=-x)
        self.od = [-ry, rx, wrap(math.pi / 2 + wrap(th - th0))]

    def _on_pose(self, msg):
        for p in msg.pose:
            if p.name == "agv":
                q = p.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y * q.y + q.z * q.z))
                self.truth = (p.position.x, p.position.y, yaw)
                return

    # ------------------------------------------------ 센서 해석
    def see_markers(self):
        """[(id, 베어링rad(좌+), 거리m)]. 실제 cv2.aruco 를 렌더 프레임에 돌린다."""
        if self.frame is None:
            return []
        gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        if self._det is not None:
            corners, ids, _ = self._det.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._dict, parameters=self._params)
        if ids is None:
            return []
        out = []
        size = self.a.marker_cm / 100.0
        for c, i in zip(corners, ids.flatten()):
            q = c.reshape(4, 2)
            u = float(q[:, 0].mean())
            # 세로변은 팻말 yaw 에 영향받지 않으므로 거리 추정에 안정적이다.
            # (가로변으로 재면 기울어진 팻말에서 거리가 과대평가된다)
            vpx = (np.linalg.norm(q[0] - q[3]) + np.linalg.norm(q[1] - q[2])) / 2.0
            dist = FX * size / max(1.0, vpx)
            bearing = math.atan2((CAM_W / 2.0 - u), FX)   # 화면 왼쪽 = CCW 양수
            out.append((int(i), bearing, dist))
        return out

    def tof_error(self):
        l, r = self.tof_l, self.tof_r
        if not (0 < l < TOF_MAX) or not (0 < r < TOF_MAX):
            return None
        return (r - l) / FURROW_HALF

    def walls_seen(self):
        return (0 < self.tof_l < TOF_MAX) and (0 < self.tof_r < TOF_MAX)

    def vision_error(self):
        if self.a.vision == "blind":
            return None, 0.0
        conf = self.rng.uniform(*VIS_CONF_RANGE)
        if self.rng.random() < VIS_GOOD_RATE:
            return self.rng.gauss(0.0, VIS_ERR_GOOD), conf
        return self.rng.choice((-1, 1)) * abs(
            self.rng.gauss(VIS_ERR_BAD, 0.3)), conf

    def fuse(self):
        """LineFollower 정책 축약: 신뢰도 비례 + ToF 거부권 + ToF 폴백."""
        v, conf = self.vision_error()
        t = self.tof_error()
        use = v is not None and conf >= 0.25
        if use and t is not None and self.a.veto > 0 and abs(v - t) > self.a.veto:
            use = False
            self.r["vision_vetoed"] += 1
        if use:
            if t is not None:
                w = min(1.0, max(0.0, (conf - 0.25) / 0.35)) * 0.75
                e = w * v + (1 - w) * t
            else:
                e = v
            self.r["vision_used"] += 1
        elif t is not None:
            e = t
            self.r["tof_only"] += 1
        else:
            e = 0.0
        return max(-1.5, min(1.5, e))

    # ------------------------------------------------ 구동 + 오도메트리
    def step(self, lin, ang):
        """명령 발행 + 데드레커닝 적분 + 채점. 한 틱(50ms)."""
        m = Twist()
        m.linear.x = lin
        m.angular.z = ang
        self.pub.publish(m)
        self.cmd = (lin, ang)

        self._update_od()
        self.r["ticks"] += 1
        self._score_tick()
        time.sleep(DT)

    def _score_tick(self):
        """월드 참값으로 채점만 한다(제어에는 절대 쓰지 않는다)."""
        if self.truth is None:
            return
        x, y, _ = self.truth
        if 0.2 < y < FIELD_LEN - 0.2:
            k = round(x / SPACING)                 # 가장 가까운 고랑 중심
            dev = abs(x - k * SPACING)
            self.r["max_dev_cm"] = max(self.r["max_dev_cm"], dev * 100)
            if dev > MARGIN:
                self.r["ridge_hit_ticks"] += 1

    def _dbg(self, msg):
        if self.a.debug:
            print("DBG " + msg, file=sys.stderr, flush=True)

    def scan_tick(self):
        """마커 관측 기록(모든 상태에서 매 틱 호출)."""
        seen = self.see_markers()
        for i, b, d in seen:
            th = self.od[2] + b
            self._mpos[i] = (self.od[0] + d * math.cos(th),
                             self.od[1] + d * math.sin(th))
        if seen:
            self.r["detect_frames"] += 1
            for i, _, _ in seen:
                self._ids_seen.add(i)
            if self.r["t_first_marker"] is None:
                self.r["t_first_marker"] = round(self.r["ticks"] * DT, 2)
                self.r["first_marker_id"] = seen[0][0]
        return seen

    # ------------------------------------------------ 상태 1: 탐색
    def do_search(self, timeout=25.0):
        """정책별 탐색. 3프레임 이상 안정적으로 보이는 팻말을 고른다."""
        t0 = time.time()
        hits = {}
        sweep_dir = 1.0
        sweep_amp = math.radians(55)
        yaw0 = self.od[2]
        while time.time() - t0 < timeout:
            seen = self.scan_tick()
            for i, b, d in seen:
                hits.setdefault(i, []).append((b, d))
            # 너무 먼 팻말은 무시한다. 회전 중에 밭 반대편 팻말이 먼저 잡히면
            # 엉뚱한 고랑(실측: 고랑 1 대신 고랑 4)으로 들어간다.
            ready = [i for i, v in hits.items()
                     if len(v) >= 3 and i <= MAX_FURROW_MARKER_ID
                     and v[-1][1] <= self.a.max_dist]
            # pair 정책은 밭 방위를 잡으려면 팻말이 2개 필요하다.
            # 15초 안에 2개를 못 보면 1개로 진행한다(가정 헤딩 사용).
            if (self.a.entry in ("pair", "mid") and len(ready) < 2
                    and time.time() - t0 < 15.0):
                ready = []
            if ready:
                best = min(ready, key=lambda i: hits[i][-1][1])   # 가장 가까운 것
                b, d = hits[best][-1]
                self.step(0.0, 0.0)
                if self.a.entry in ("pair", "mid"):
                    self.r["anchored"] = self.anchor_field_heading()
                return best, b, d

            if self.a.search == "rotate":
                self.step(0.0, 0.5)
            elif self.a.search == "sweep":
                if abs(wrap(self.od[2] - yaw0)) > sweep_amp:
                    sweep_dir = -math.copysign(1.0, wrap(self.od[2] - yaw0))
                self.step(0.0, 0.5 * sweep_dir)
            elif self.a.search == "back":
                # 1단계: 밭에서 멀어져 화각을 확보한다. 2단계: 회전 탐색.
                if self.od[1] > -1.2:
                    self.step(-0.20, 0.0)
                else:
                    self.step(0.0, 0.5)
            elif self.a.search == "creep":
                # 밭 안쪽으로 천천히 전진해 팻말과의 거리를 줄인다.
                # 이랑에 올라타지 않도록 전진량을 0.8m 로 제한한다.
                if self.od[1] < 0.8:
                    self.step(0.12, 0.0)
                else:
                    self.step(0.0, 0.5)          # 다 왔으면 회전 탐색으로 전환
            else:
                raise ValueError(self.a.search)
        self.r["fail"] = "search_timeout"
        self.step(0.0, 0.0)
        return None, None, None

    # ------------------------------------------------ 이동 유틸
    def goto(self, gx, gy, timeout=25.0, speed=0.22, stop_r=0.06,
             refresh_id=None):
        """오도메트리 좌표 (gx,gy) 로 이동. 도중에 팻말을 다시 보면 목표를 갱신."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            seen = self.scan_tick()
            if refresh_id is not None:
                for i, b, d in seen:
                    if i == refresh_id:
                        gx, gy = self._marker_goal(b, d)
                        break
            dx, dy = gx - self.od[0], gy - self.od[1]
            dist = math.hypot(dx, dy)
            if dist < stop_r:
                self.step(0.0, 0.0)
                return True
            err = wrap(math.atan2(dy, dx) - self.od[2])
            if abs(err) > math.radians(35):        # 크게 틀어졌으면 제자리 회전
                self.step(0.0, max(-0.7, min(0.7, 1.5 * err)))
            else:
                self.step(min(speed, 0.5 * dist + 0.08),
                          max(-0.7, min(0.7, 1.5 * err)))
        self.r["fail"] = self.r["fail"] or "goto_timeout"
        self.step(0.0, 0.0)
        return False

    def turn_to(self, target_th, timeout=12.0, tol=math.radians(4)):
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.scan_tick()
            e = wrap(target_th - self.od[2])
            if abs(e) < tol:
                self.step(0.0, 0.0)
                return True
            self.step(0.0, max(-0.7, min(0.7, 1.8 * e)))
        self.step(0.0, 0.0)
        return False

    def anchor_field_heading(self, min_baseline=0.4):
        """팻말 2개의 위치로 밭 방위를 다시 잡는다.

        팻말은 헤드랜드를 따라 ID 순서대로 늘어서 있다(이랑 k 중심).
        따라서 낮은 ID -> 높은 ID 벡터가 헤드랜드 방향이고, 그것을 +90도
        돌리면 밭 안쪽 방향이다. 단일 마커 yaw 추정과 달리 **베어링/거리만**
        쓰므로 실제 검출기에서도 성립한다.
        """
        ids = sorted(i for i in self._mpos if i <= MAX_FURROW_MARKER_ID)
        if len(ids) < 2:
            return False
        # 인접한 쌍을 우선한다(가장 짧은 베이스라인이 아니라 **가장 확실한 쌍**).
        adj = [(x, y) for x, y in zip(ids, ids[1:]) if y - x == 1]
        a, b = adj[0] if adj else (ids[0], ids[-1])
        vx = self._mpos[b][0] - self._mpos[a][0]
        vy = self._mpos[b][1] - self._mpos[a][1]
        if math.hypot(vx, vy) < min_baseline:
            return False
        self.field_th = wrap(math.atan2(vy, vx) + math.pi / 2)
        self.r["field_th_deg"] = round(math.degrees(self.field_th), 1)
        self.r["anchor_ids"] = [a, b]
        return True

    def _adjacent_pair(self):
        """관측된 팻말 중 **ID 가 연속인** 가장 가까운 쌍의 좌표를 준다.

        ID 가 1 차이여야 그 사이가 진짜 고랑이다(0 과 2 의 중점은 이랑 1 위다).
        """
        ids = sorted(i for i in self._mpos if i <= MAX_FURROW_MARKER_ID)
        best = None
        for a, b in zip(ids, ids[1:]):
            if b - a != 1:
                continue
            d = min(math.hypot(*self._mpos[k]) for k in (a, b))
            if best is None or d < best[0]:
                best = (d, self._mpos[a], self._mpos[b])
        return None if best is None else (best[1], best[2])

    def _marker_goal(self, bearing, dist):
        """관측된 팻말로 **진입 목표점**(오도메트리 좌표)을 만든다.

        팻말은 이랑 위에 있고, 들어갈 고랑은 그 이랑을 **지나친** 쪽이다.
        survey 정책은 측량값(간격/2)으로 목표점을 바로 계산한다.
        tof 정책은 옆으로 밀지 않고 팻말 정면으로 가서 ToF 로 찾는다.
        """
        th = self.od[2] + bearing
        mx = self.od[0] + dist * math.cos(th)
        my = self.od[1] + dist * math.sin(th)
        fx, fy = math.cos(self.field_th), math.sin(self.field_th)
        lx, ly = fy, -fx        # 밭 기준 오른쪽(= 팻말을 지나친 쪽)

        if self.a.entry == "mid":
            pair = self._adjacent_pair()
            if pair is not None:
                (ax, ay), (bx, by) = pair
                cx, cy = (ax + bx) / 2.0, (ay + by) / 2.0
                return cx - 0.5 * fx, cy - 0.5 * fy

        off = 0.0 if self.a.entry == "tof" else SPACING / 2.0
        # 진입점 = 팻말 + (지나친 쪽으로 off) - (밭 안쪽으로 0.7m)
        return mx + off * lx - 0.7 * fx, my + off * ly - 0.7 * fy

    # ------------------------------------------------ 상태 2: 진입
    def do_enter(self, timeout=150.0, adv_limit=1.6):
        """밭 방향으로 서서 전진. 좌우 벽이 연속으로 잡히면 진입 성공."""
        self.turn_to(self.field_th)
        t0 = time.time()
        good = 0
        probes = 0
        p0 = (self.od[0], self.od[1])
        while time.time() - t0 < timeout:
            self.scan_tick()
            if self.walls_seen():
                good += 1
                if good >= 8:      # 입구에서 벽이 깜빡이므로 넉넉히 확인
                    self.step(0.0, 0.0)
                    return True
            else:
                good = 0
            adv = ((self.od[0] - p0[0]) * math.cos(self.field_th)
                   + (self.od[1] - p0[1]) * math.sin(self.field_th))
            # [수정] 예전 0.9m 는 너무 짧았다. 진입점이 팻말 0.7m 앞이고
            #   오도메트리가 회전 미끄러짐으로 20~25% 과대보고하기 때문에
            #   실제로는 밭 경계 4cm 앞에서 포기했다(실측).
            if adv > adv_limit:                    # 이만큼 갔는데 벽이 없다
                if self.a.entry != "tof" or probes >= 5:
                    self.r["fail"] = "no_walls"
                    self.step(0.0, 0.0)
                    return False
                # tof 정책: 후진 -> 옆으로 한 칸 -> 재시도 (측량값 미사용)
                probes += 1
                for _ in range(int(1.1 / (0.22 * DT))):
                    self.step(-0.22, 0.0)
                self.turn_to(wrap(self.field_th - math.pi / 2))   # 밭 기준 오른쪽
                for _ in range(int(0.30 / (0.20 * DT))):
                    self.step(0.20, 0.0)
                self.turn_to(self.field_th)
                p0 = (self.od[0], self.od[1])
                good = 0
                continue
            # [수정] 예전에는 한쪽 벽이 보이면 그냥 계속 회전시켰다. 그러면
            #   회전이 멈추지 않아 로봇이 밭 방향에서 90도까지 돌아가
            #   이랑 위를 옆으로 긁으며 달렸다(실측: 헤딩 0도, 오도메트리 4.3m).
            #   이제는 **밭 방향 유지**가 기본이고, 벽은 거기에 얹는 보정일 뿐이다.
            l_ok = 0 < self.tof_l < TOF_MAX
            r_ok = 0 < self.tof_r < TOF_MAX
            bias = 0.0
            if l_ok and not r_ok:
                bias = -(0.35 if self.tof_l < 0.18 else 0.20)   # 왼쪽 벽 -> 우로
            elif r_ok and not l_ok:
                bias = +(0.35 if self.tof_r < 0.18 else 0.20)
            hold = 2.0 * wrap(self.field_th - self.od[2])       # 밭 방향 유지
            self.step(0.14, max(-0.6, min(0.6, hold + bias)))
            if self.a.debug and self.r["ticks"] % 20 == 0:
                self._dbg(f"  enter adv={adv:.2f} od=({self.od[0]:.2f},"
                          f"{self.od[1]:.2f},{math.degrees(self.od[2]):.0f}) "
                          f"truth=({self.truth[0]:.2f},{self.truth[1]:.2f},"
                          f"{math.degrees(self.truth[2]):.0f}) "
                          f"tof=({self.tof_l:.2f},{self.tof_r:.2f})")
        self.r["fail"] = self.r["fail"] or "enter_timeout"
        self.step(0.0, 0.0)
        return False

    # ------------------------------------------------ 상태 3: 고랑 주행
    def do_furrow(self, timeout=45.0, speed=0.22, kp=1.2, kd=0.45):
        """고랑 끝까지 주행. 좌우 벽이 연속으로 사라지면 끝으로 판정."""
        t0 = time.time()
        lost = 0
        prev = None
        p0 = (self.od[0], self.od[1])
        while time.time() - t0 < timeout:
            self.scan_tick()
            travelled = math.hypot(self.od[0] - p0[0], self.od[1] - p0[1])
            if not self.walls_seen():
                lost += 1
                # [수정] 예전에는 0.6초만 벽이 없으면 '고랑 끝'으로 판정했다.
                #   입구 바로 안쪽에서는 ToF 가 깜빡이기 때문에 진입하자마자
                #   끝났다고 착각해 0.12m 만에 유턴했다(실측 6/6).
                #   1.5초 연속 + 최소 1.5m 주행을 모두 만족해야 끝으로 본다.
                # 3초(60틱) 연속 + 최소 1.5m. 1.5초로는 고랑 중간의 일시적인
                # ToF 드롭아웃을 끝으로 오판했다(실측 2/6이 1.50m 에서 멈춤).
                if lost >= 60 and travelled > 1.5:
                    self.step(0.0, 0.0)
                    self.r["drive_len_m"] = round(travelled, 2)
                    return True
                if self.a.debug and self.r["ticks"] % 20 == 0:
                    self._dbg(f"  furrow(벽없음) d={travelled:.2f} lost={lost} "
                              f"truth=({self.truth[0]:.2f},{self.truth[1]:.2f},"
                              f"{math.degrees(self.truth[2]):.0f}) "
                              f"tof=({self.tof_l:.2f},{self.tof_r:.2f})")
                self.step(speed, 0.0)            # 벽이 잠깐 없으면 직진 유지
                continue
            lost = 0
            # [수정] 비례항만 쓰면 '횡오차 -> 조향각속도' 가 2중 적분이라
            #   반드시 진동한다. 실제로 좌우로 흔들리다 ±9~10cm 에서 ToF 가
            #   벽에 닿아(최소거리 2cm 미만) inf 를 뱉고, 그걸 '고랑 끝'으로
            #   오판했다(비전 유무와 무관하게 6m 중 1.5~4.6m 만 주행).
            #   미분항으로 감쇠를 준다.
            err = self.fuse()
            d = 0.0 if prev is None else (err - prev) / DT
            prev = err
            ang = max(-0.8, min(0.8, -(kp * err + kd * d)))
            self.step(speed, ang)
            if self.a.debug and self.r["ticks"] % 20 == 0:
                self._dbg(f"  furrow d={travelled:.2f} lost={lost} "
                          f"truth=({self.truth[0]:.2f},{self.truth[1]:.2f},"
                          f"{math.degrees(self.truth[2]):.0f}) "
                          f"tof=({self.tof_l:.2f},{self.tof_r:.2f})")
        self.r["drive_len_m"] = round(
            math.hypot(self.od[0] - p0[0], self.od[1] - p0[1]), 2)
        self.r["fail"] = self.r["fail"] or "furrow_timeout"
        self.step(0.0, 0.0)
        return False

    # ------------------------------------------------ 실행
    def run(self):
        time.sleep(3.0)                              # 첫 센서 프레임 대기
        self._odom0 = self._odom_raw                 # 이번 시행의 오도메트리 원점
        self._update_od()
        if self.frame is None:
            self.r["fail"] = "no_camera"
            return self.r
        t_start = self.truth

        mid, bearing, dist = self.do_search()
        self.r["ids"] = sorted(self._ids_seen)
        if mid is None:
            return self.r

        gx, gy = self._marker_goal(bearing, dist)
        self._dbg(f"marker id={mid} bearing={math.degrees(bearing):.1f}deg "
                  f"dist={dist:.2f}m -> goal_od=({gx:.2f},{gy:.2f}) "
                  f"od={self.od[0]:.2f},{self.od[1]:.2f},"
                  f"{math.degrees(self.od[2]):.0f}deg truth={self.truth}")
        self.goto(gx, gy, refresh_id=mid)
        self._dbg(f"goto 종료 od={self.od[0]:.2f},{self.od[1]:.2f},"
                  f"{math.degrees(self.od[2]):.0f}deg truth={self.truth}")
        entered = self.do_enter()
        self._dbg(f"enter={entered} od={self.od[0]:.2f},{self.od[1]:.2f} "
                  f"truth={self.truth} tof=({self.tof_l:.2f},{self.tof_r:.2f})")
        self.r["t_entered"] = round(self.r["ticks"] * DT, 2) if entered else None
        if entered and self.truth is not None:
            x, y, _ = self.truth
            k = round(x / SPACING)
            self.r["entered_furrow"] = int(k) + 1     # 고랑 중심 x=(k-1)*spacing
            self.r["entry_ok"] = abs(x - k * SPACING) < 0.15 and y > -0.2
        if not entered or self.a.stage == "entry":
            return self.r

        # --- stage=full: 고랑 끝 -> 유턴 -> 복귀 ---
        self.do_furrow()
        self.r["drive_len_out"] = self.r["drive_len_m"]

        # 유턴: 밭 방위도 함께 뒤집는다(복귀 구간의 헤딩 유지 기준이 된다)
        self.r["uturn_ok"] = self.turn_to(wrap(self.od[2] + math.pi), timeout=16)
        self.field_th = wrap(self.field_th + math.pi)

        # [수정] 유턴 직후에는 고랑 **밖**(끝을 지나친 지점)에 서 있다.
        #   바로 do_furrow 를 부르면 벽이 없어 곧장 '고랑 끝'으로 오판하고
        #   최소 주행거리 1.5m 만 달리고 끝났다(실측 6/6 모두 1.50m).
        #   먼저 고랑에 다시 들어간 뒤 추종을 시작한다.
        # 고랑 끝 판정 후 3초를 더 달렸으므로 1m 남짓 지나쳐 있다.
        # 재진입 예산을 넉넉히 준다.
        reacq = self.do_enter(timeout=60.0, adv_limit=2.8)
        self.r["reacquired"] = bool(reacq)
        self.r["returned_ok"] = bool(reacq and self.do_furrow(timeout=60.0))
        self.r["drive_len_back"] = self.r["drive_len_m"]
        if self.truth is not None and t_start is not None:
            self.r["home_err_cm"] = round(math.hypot(
                self.truth[0] - t_start[0], self.truth[1] - t_start[1]) * 100, 1)
        return self.r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", default="rotate",
                    choices=["rotate", "sweep", "creep", "back"])
    ap.add_argument("--entry", default="survey",
                    choices=["survey", "tof", "pair", "mid"])
    ap.add_argument("--stage", default="entry", choices=["entry", "full"])
    ap.add_argument("--vision", default="measured", choices=["measured", "blind"])
    ap.add_argument("--veto", type=float, default=0.15)
    ap.add_argument("--slip", type=float, default=0.0,
                    help="엔코더 스케일 오차(0.03=3%%). 실제 미끄러짐은 물리가 낸다")
    ap.add_argument("--marker-cm", type=float, default=20.0)
    ap.add_argument("--tilt", type=float, default=30.0)   # 기록용(실제 각은 월드)
    ap.add_argument("--start-x", type=float, default=-0.9)
    ap.add_argument("--start-y", type=float, default=-1.0)
    ap.add_argument("--start-yaw", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-dist", type=float, default=3.0,
                    help="이 거리보다 먼 팻말은 무시(엉뚱한 고랑 진입 방지)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    h = PolicyHarness(args)
    try:
        r = h.run()
    finally:
        h.step(0.0, 0.0)
    print("RESULT " + json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()

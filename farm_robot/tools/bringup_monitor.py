# -*- coding: utf-8 -*-
"""
farm_robot/tools/bringup_monitor.py
------------------------------------
브링업/모니터링 화면. **주행 제어 시퀀스와 무관하게 단독으로** 돕니다.

    라즈베리파이 : USB 웹캠(마커 인식) + 좌/우 1D ToF (I2C)
    아두이노 메가 : 초음파 수위 + 펌프 PWM  (USB 시리얼)

화면에 한 번에 띄우는 것
    · 웹캠 영상 + ArUco 마커 검출 결과(ID / 거리 / 좌우 오프셋)
    · 좌우 ToF 값(막대 + 숫자)
    · 물통 수위 게이지 + 상태(FULL/OK/LOW/EMPTY/FAULT)
    · 펌프 상태(OFF/ON/LOCK) 와 듀티

없는 장치는 자동으로 건너뜁니다. 아무것도 안 붙은 PC 에서도 --mock 으로
레이아웃을 확인할 수 있습니다.

    # 라즈베리파이 (창 띄우기)
    python tools/bringup_monitor.py

    # 헤드리스 -> 브라우저에서 http://<pi>:8080 으로 보기
    python tools/bringup_monitor.py --stream 8080

    # 하드웨어 없이 화면만 확인
    python tools/bringup_monitor.py --mock

키 (창 모드)
    q / ESC   종료
    space     펌프 on/off 토글
    + / -     펌프 듀티 ±16
    0         펌프 즉시 정지
    s         현재 화면 저장
"""

import argparse
import math
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

# ---------------------------------------------------------------- 한글 텍스트
#   OpenCV 는 한글을 못 그린다. PIL 과 한글 폰트가 있으면 한글로,
#   없으면 영문으로 자동 전환한다(라즈베리파이에 폰트가 없을 수 있다).
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/malgun.ttf",
]

try:
    from PIL import Image, ImageDraw, ImageFont
    _FONT_PATH = next((p for p in _FONT_CANDIDATES if Path(p).exists()), None)
except Exception:
    _FONT_PATH = None

KOREAN = _FONT_PATH is not None
_font_cache = {}


def _pil_font(size):
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(_FONT_PATH, size)
    return _font_cache[size]


def L(ko, en):
    """한글 폰트가 있으면 한글, 없으면 영문."""
    return ko if KOREAN else en


class TextLayer:
    """한 프레임에 그릴 텍스트를 모았다가 한 번에 합성한다(PIL 왕복 최소화)."""

    def __init__(self):
        self.items = []

    def add(self, xy, text, size=18, color=(240, 240, 240), bold=False):
        self.items.append((xy, text, size, color))

    def render(self, bgr):
        if not self.items:
            return bgr
        if not KOREAN:
            for (x, y), text, size, color in self.items:
                cv2.putText(bgr, text, (x, y + size), cv2.FONT_HERSHEY_SIMPLEX,
                            size / 32.0, color[::-1], 1, cv2.LINE_AA)
            self.items = []
            return bgr
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for xy, text, size, color in self.items:
            d.text(xy, text, font=_pil_font(size), fill=color)
        self.items = []
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- 메가 링크
class TankPumpLink:
    """agrix_tank_pump.ino 와 USB 로 대화한다.

    수신 : TANK <dist> <level> <pct> <state> <duty> <pump> <flags>
    송신 : PUMP <duty> / STOP / STATUS / PING
    """

    def __init__(self, port, baud=115200):
        self.port = port
        self.available = False
        self.last_error = ""
        self.dist_mm = None
        self.level_mm = 0
        self.level_pct = 0
        self.tank_state = "?"
        self.pump_duty = 0
        self.pump_state = "?"
        self.flags = 0
        self.last_rx = 0.0

        self._serial = None
        self._closed = False
        try:
            import serial

            self._serial = serial.Serial(port, baud, timeout=0.3)
            time.sleep(2.0)                     # 메가 자동 리셋 대기
            self._serial.reset_input_buffer()
            threading.Thread(target=self._reader, daemon=True).start()
            self.available = True
        except Exception as exc:
            self.last_error = str(exc)

    def _reader(self):
        while not self._closed:
            try:
                line = self._serial.readline().decode(errors="ignore").strip()
            except Exception as exc:
                self.last_error = str(exc)
                time.sleep(0.5)
                continue
            if not line.startswith("TANK "):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            try:
                self.dist_mm = int(p[1]) if int(p[1]) >= 0 else None
                self.level_mm = int(p[2])
                self.level_pct = int(p[3])
                self.tank_state = p[4]
                self.pump_duty = int(p[5])
                self.pump_state = p[6]
                self.flags = int(p[7])
                self.last_rx = time.monotonic()
            except ValueError:
                continue

    def send(self, text):
        if not self.available:
            return
        try:
            self._serial.write((text + "\n").encode())
        except Exception as exc:
            self.last_error = str(exc)

    def set_pump(self, duty):
        self.send(f"PUMP {max(0, min(255, int(duty)))}")

    def link_ok(self):
        return self.available and (time.monotonic() - self.last_rx) < 1.5

    def close(self):
        self.send("STOP")
        self._closed = True
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass


class MockLink:
    """하드웨어 없이 레이아웃을 확인하기 위한 가짜 링크."""

    def __init__(self):
        self.available = True
        self.last_error = ""
        self.pump_duty = 0
        self._t0 = time.monotonic()
        self._target = 0

    def _level(self):
        t = time.monotonic() - self._t0
        return max(20, 300 - int((t * 8) % 300))

    @property
    def level_mm(self):
        return self._level()

    @property
    def level_pct(self):
        return int(self._level() / 3)

    @property
    def dist_mm(self):
        return 340 - self._level()

    @property
    def tank_state(self):
        lv = self._level()
        return "EMPTY" if lv <= 50 else "LOW" if lv <= 90 else \
               "FULL" if lv >= 270 else "OK"

    @property
    def pump_state(self):
        if self.tank_state == "EMPTY":
            return "LOCK"
        return "ON" if self.pump_duty > 0 else "OFF"

    @property
    def flags(self):
        return 1 if self.tank_state == "EMPTY" else 0

    def set_pump(self, duty):
        self._target = max(0, min(255, int(duty)))
        self.pump_duty = 0 if self.tank_state == "EMPTY" else self._target

    def send(self, text):
        pass

    def link_ok(self):
        return True

    def close(self):
        pass


# ---------------------------------------------------------------- 화면 구성
#  Gazebo 녹화 화면과 같은 방식으로, 옆 패널이 아니라 **영상 위에 겹쳐** 그린다.
#  화면 크기가 달라도 같은 비율로 보이도록 640px 폭을 기준으로 스케일한다.
COL_OK = (120, 220, 120)
COL_WARN = (70, 190, 250)
COL_BAD = (80, 80, 250)
COL_DIM = (165, 165, 165)
COL_FG = (245, 245, 245)


def state_color(state):
    return {"OK": COL_OK, "FULL": COL_OK, "LOW": COL_WARN,
            "EMPTY": COL_BAD, "FAULT": COL_BAD}.get(state, COL_DIM)


def shade(img, x0, y0, x1, y1, alpha=0.45):
    """영상 위 글씨가 묻히지 않도록 반투명 어둡게."""
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(img.shape[1], x1)
    y1 = min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = cv2.addWeighted(
        roi, 1 - alpha, np.zeros_like(roi), alpha, 0)


def draw_tof_bar(img, txt, x, y, w, h, mm, max_mm, label, s,
                 label_left=0):
    """ToF 한 채널의 세로 막대. 가까울수록 채워지고, 아주 가까우면 빨강."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (38, 38, 38), -1)
    if mm is None or mm <= 0 or mm >= max_mm:
        value, color = "--", COL_DIM
    else:
        fill = int(h * (1.0 - min(1.0, mm / max_mm)))
        color = COL_BAD if mm < 60 else COL_WARN if mm < 120 else COL_OK
        cv2.rectangle(img, (x, y + h - fill), (x + w, y + h), color, -1)
        value = f"{mm:.0f}"
    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)
    # 오른쪽 막대는 라벨을 왼쪽으로 밀어야 화면 밖으로 안 나간다
    lx = x - int(6 * s) - int(label_left * s)
    txt.add((lx, y - int(22 * s)), label, int(15 * s), COL_FG)
    txt.add((lx, y + h + int(2 * s)), value, int(15 * s), color[::-1])


def draw_tank(img, txt, x, y, w, h, link, tank_mm, empty_mm, s):
    """물통 게이지(세로). 눈금은 실제 물통 높이 기준."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (38, 38, 38), -1)
    level = getattr(link, "level_mm", 0) or 0
    state = getattr(link, "tank_state", "?")
    col = state_color(state)
    fill = int(h * min(1.0, max(0.0, level / max(1, tank_mm))))
    if fill > 0:
        cv2.rectangle(img, (x, y + h - fill), (x + w, y + h), col, -1)

    ey = y + h - int(h * empty_mm / max(1, tank_mm))     # 빈통 문턱선
    cv2.line(img, (x - int(5 * s), ey), (x + w + int(5 * s), ey),
             (60, 60, 235), max(1, int(2 * s)))

    for cm in range(10, tank_mm // 10, 10):              # 10cm 눈금
        gy = y + h - int(h * (cm * 10) / max(1, tank_mm))
        cv2.line(img, (x, gy), (x + int(7 * s), gy), (140, 140, 140), 1)

    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)
    return level, state, col


def compose(frame, link, tof, markers, tank_mm, empty_mm, tof_max, pump_target,
            fps):
    img = frame.copy()
    h, w = img.shape[:2]
    s = w / 640.0
    txt = TextLayer()

    level = getattr(link, "level_mm", 0) or 0
    tstate = getattr(link, "tank_state", "?")
    tcol = state_color(tstate)
    pstate = getattr(link, "pump_state", "?")
    duty = getattr(link, "pump_duty", 0)
    pcol = (COL_BAD if pstate == "LOCK" else
            COL_OK if pstate == "ON" else COL_DIM)
    dist = getattr(link, "dist_mm", None)

    # ---- 좌상단 상태 블록 ----
    rows = [
        (L("마커", "MARKER") + "   " +
         (", ".join(str(m) for m in sorted(markers)) if markers
          else L("없음", "none")),
         (235, 235, 90) if markers else COL_DIM),
        (f"{L('물통','TANK')}   {level/10:.1f} cm  "
         f"({getattr(link, 'level_pct', 0)}%)   {tstate}", tcol[::-1]),
        (f"{L('펌프','PUMP')}   {pstate}   duty {duty} "
         f"({duty*100//255}%)", pcol[::-1]),
    ]
    line_h = int(23 * s)
    shade(img, 0, 0, int(330 * s), int(14 * s) + line_h * len(rows))
    for i, (text, color) in enumerate(rows):
        txt.add((int(10 * s), int(6 * s) + i * line_h), text, int(17 * s),
                color)

    # ---- 우상단: 링크 / fps ----
    ok = link.link_ok()
    shade(img, w - int(150 * s), 0, w, int(46 * s))
    txt.add((w - int(142 * s), int(4 * s)),
            L("메가 연결됨", "Mega OK") if ok else L("메가 끊김", "Mega DOWN"),
            int(15 * s), COL_OK[::-1] if ok else COL_BAD[::-1])
    txt.add((w - int(142 * s), int(24 * s)),
            f"{fps:.0f} fps   {L('목표','tgt')} {pump_target}",
            int(14 * s), COL_DIM)

    # ---- 좌우 하단: ToF 막대 (가제보 녹화와 같은 자리) ----
    bar_w, bar_h = int(22 * s), int(80 * s)
    by = h - bar_h - int(20 * s)
    draw_tof_bar(img, txt, int(14 * s), by, bar_w, bar_h, tof[0], tof_max,
                 L("좌 ToF", "ToF L"), s)
    draw_tof_bar(img, txt, w - int(36 * s), by, bar_w, bar_h, tof[1], tof_max,
                 L("우 ToF", "ToF R"), s, label_left=46)

    if tof[0] and tof[1] and 0 < tof[0] < tof_max and 0 < tof[1] < tof_max:
        diff = tof[1] - tof[0]
        side = L("오른쪽 여유", "room right") if diff > 0 else                L("왼쪽 여유", "room left")
        label = f"{L('차이','diff')} {diff:+.0f} mm  ({side})"
        shade(img, int(52 * s), h - int(30 * s), int(280 * s), h - int(6 * s))
        txt.add((int(58 * s), h - int(28 * s)), label, int(15 * s), COL_DIM)

    # ---- 우측: 물통 게이지 ----
    gx = w - int(112 * s)
    gy = int(58 * s)
    gh = h - gy - int(150 * s)
    gw = int(20 * s)
    if gh > int(60 * s):
        shade(img, gx - int(12 * s), gy - int(22 * s), gx + int(46 * s),
              gy + gh + int(26 * s), 0.35)
        draw_tank(img, txt, gx, gy, gw, gh, link, tank_mm, empty_mm, s)
        txt.add((gx - int(6 * s), gy - int(20 * s)),
                f"{tank_mm//10}cm", int(13 * s), COL_DIM)
        txt.add((gx - int(6 * s), gy + gh + int(4 * s)),
                (f"{dist} mm" if dist is not None else L("이상", "fault")),
                int(13 * s), COL_DIM)

    return txt.render(img)


# ---------------------------------------------------------------- MJPEG 스트림
class MjpegServer:
    """헤드리스 파이에서 브라우저로 보기 위한 최소 MJPEG 서버."""

    def __init__(self, port):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self._frame = None
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with outer._lock:
                            buf = outer._frame
                        if buf is not None:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(buf)
                            self.wfile.write(b"\r\n")
                        time.sleep(0.05)
                except Exception:
                    pass

        self._srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def publish(self, bgr):
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._frame = buf.tobytes()


# ---------------------------------------------------------------- 메인
def build_detector(params_only=False):
    """실기 aruco_detector.py 와 **같은 튜닝**의 검출기를 만든다."""
    d = (cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
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
    det = (cv2.aruco.ArucoDetector(d, p)
           if hasattr(cv2.aruco, "ArucoDetector") else None)
    return d, p, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--serial", default="/dev/ttyACM0",
                    help="메가 포트. 가능하면 /dev/serial/by-id/... 를 쓰세요")
    ap.add_argument("--marker-cm", type=float, default=18.0)
    ap.add_argument("--tank-mm", type=int, default=300, help="물통 높이(mm)")
    ap.add_argument("--empty-mm", type=int, default=50, help="빈통 판정 수심(mm)")
    ap.add_argument("--tof-max", type=float, default=800.0)
    ap.add_argument("--no-tof", action="store_true")
    ap.add_argument("--no-mega", action="store_true")
    ap.add_argument("--mock", action="store_true",
                    help="하드웨어 없이 레이아웃만 확인")
    ap.add_argument("--stream", type=int, default=0,
                    help="MJPEG 포트(0=사용 안 함)")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    # --- 카메라 ---
    cap = None
    if not args.mock:
        cap = cv2.VideoCapture(args.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            print(f"경고: 웹캠 {args.camera_index} 를 열지 못했습니다. "
                  f"--mock 으로 화면만 확인할 수 있습니다.")
            cap.release()
            cap = None

    # --- ToF ---
    tof_pair = None
    if not args.no_tof and not args.mock:
        try:
            from config import TOF_LEFT, TOF_RIGHT
            from sensors.tof_sensor import ToFPair

            tof_pair = ToFPair(TOF_LEFT, TOF_RIGHT)
            tof_pair.init_hardware()
        except Exception as exc:
            print(f"ToF 초기화 실패: {exc} — ToF 없이 계속합니다.")
            tof_pair = None

    # --- 메가 ---
    if args.mock:
        link = MockLink()
    elif args.no_mega:
        link = MockLink()
        link.available = False
    else:
        link = TankPumpLink(args.serial)
        if not link.available:
            print(f"메가 연결 실패({link.last_error}) — 가짜 값으로 표시합니다.")
            link = MockLink()
            link.available = False

    dic, params, det = build_detector()
    stream = MjpegServer(args.stream) if args.stream else None
    if stream:
        print(f"MJPEG: http://<이 장치 IP>:{args.stream}/")

    pump_target = 0
    t_prev = time.monotonic()
    fps = 0.0
    shot = 0

    try:
        while True:
            # 프레임
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    frame = np.full((args.height, args.width, 3), 40, np.uint8)
            else:
                frame = np.full((args.height, args.width, 3), 40, np.uint8)
                cv2.putText(frame, "NO CAMERA", (args.width // 2 - 90,
                                                 args.height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2)

            # 마커
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if det is not None:
                corners, ids, _ = det.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, dic, parameters=params)
            found = []
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                fx = (frame.shape[1] / 2.0) / math.tan(math.radians(62.0) / 2)
                for c, i in zip(corners, ids.flatten()):
                    q = c.reshape(4, 2)
                    found.append(int(i))
                    # 세로변으로 거리 근사 (팻말이 기울어도 덜 흔들린다).
                    #   ★ 카메라 캘리브레이션 전까지는 어림값입니다.
                    vpx = (np.linalg.norm(q[0] - q[3]) +
                           np.linalg.norm(q[1] - q[2])) / 2.0
                    dist = fx * (args.marker_cm / 100.0) / max(1.0, vpx)
                    u = float(q[:, 0].mean())
                    bear = math.degrees(math.atan2(frame.shape[1] / 2 - u, fx))
                    cv2.putText(frame, f"ID{int(i)} {dist:.2f}m {bear:+.0f}deg",
                                (int(q[:, 0].min()), int(q[:, 1].min()) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255),
                                2, cv2.LINE_AA)

            # ToF
            if tof_pair is not None:
                try:
                    tof = tof_pair.read()
                except Exception:
                    tof = (None, None)
            elif args.mock:
                t = time.monotonic()
                tof = (200 + 60 * math.sin(t), 200 - 60 * math.sin(t))
            else:
                tof = (None, None)

            # 합성
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(1e-3, now - t_prev)
            t_prev = now
            out = compose(frame, link, tof, found, args.tank_mm, args.empty_mm,
                          args.tof_max, pump_target, fps)

            if stream:
                stream.publish(out)
            if not args.headless:
                cv2.imshow("Agri-X bring-up", out)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord(" "):
                    pump_target = 0 if pump_target else 128
                    link.set_pump(pump_target)
                elif k in (ord("+"), ord("=")):
                    pump_target = min(255, pump_target + 16)
                    link.set_pump(pump_target)
                elif k in (ord("-"), ord("_")):
                    pump_target = max(0, pump_target - 16)
                    link.set_pump(pump_target)
                elif k == ord("0"):
                    pump_target = 0
                    link.send("STOP")
                elif k == ord("s"):
                    name = f"bringup_{shot:03d}.png"
                    cv2.imwrite(name, out)
                    print("저장:", name)
                    shot += 1
            else:
                time.sleep(0.03)

            # 펌프를 켜 둔 동안에는 메가 워치독(1.5초)이 걸리지 않게 재전송
            if pump_target:
                link.set_pump(pump_target)
    except KeyboardInterrupt:
        pass
    finally:
        link.close()
        if cap is not None:
            cap.release()
        if tof_pair is not None:
            try:
                tof_pair.close()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

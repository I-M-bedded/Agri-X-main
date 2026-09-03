# -*- coding: utf-8 -*-
"""
farm_robot/tools/bringup_overlay_fixed.py
------------------------------------------
bringup_monitor.py 의 **오버레이가 잘 나오는 case** 를, **카메라만 실제
장치로** 쓰고 나머지 수치는 전부 고정해서 띄우는 도구.

    실제 : USB 웹캠 영상 + ArUco 마커 검출(ID / 거리 / 방위)
    고정 : 좌/우 ToF, 물통 수위/상태, 펌프 상태/듀티, 메가 링크, fps

ToF(adafruit_vl53l1x/blinka)나 메가가 아직 안 붙은 상태에서도 카메라 화면
위에 오버레이가 어떻게 얹히는지 그대로 확인할 수 있다.

bringup_monitor.py 와의 차이
    · compose() / MjpegServer / 저장 루틴은 **bringup_monitor 에서 그대로
      가져다 쓴다**(import). 따라서 여기서 보이는 레이아웃은 실기 모니터와
      100% 같은 코드가 그린 것이다.
    · 카메라 외의 값이 고정이다. MockLink 처럼 수위가 계속 내려가거나
      ToF 가 사인파로 흔들리지 않는다 -> 스크린샷/보고서용으로 쓰기 좋다.
    · 카메라가 없으면 합성 고랑 장면 + 실제 렌더링된 ArUco 마커로 대체한다
      (--no-camera 로 강제할 수도 있다).

    # 실제 웹캠 + 고정 오버레이 (기본. config.CAMERA_INDEX 사용)
    python tools/bringup_overlay_fixed.py

    # 웹캠 번호 지정 / case 골라서
    python tools/bringup_overlay_fixed.py --camera 2 --case low
    python tools/bringup_overlay_fixed.py --list-cases

    # 사진 1장만 저장하고 종료 (--warmup 초만큼 웹캠 예열)
    python tools/bringup_overlay_fixed.py --snapshot ../media/bringup/fixed_good.jpg

    # 모든 case 를 한 장씩 저장 (같은 카메라 화면 위에)
    python tools/bringup_overlay_fixed.py --snapshot-all ../media/bringup

    # 브라우저로 확인 -> http://<이 장치 IP>:8080
    python tools/bringup_overlay_fixed.py --stream 8080

    # 카메라도 안 쓰기: 합성 장면 / 사진 배경
    python tools/bringup_overlay_fixed.py --no-camera
    python tools/bringup_overlay_fixed.py --image ../media/field.jpg

키 (창 모드)
    q / ESC   종료
    c         다음 case 로 순환 (오버레이 상태별로 눈으로 비교)
    s         현재 화면 저장 (media/bringup)
"""

# 이 도구에서 '진짜'인 것은 카메라(+마커 검출)뿐이다. ToF/메가는 건드리지
# 않으므로 adafruit_vl53l1x, RPi.GPIO, pyserial 이 없어도 그냥 돈다.

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

# 오버레이 그리는 코드는 실기 모니터와 공유한다(복제하지 않는다).
from bringup_monitor import (            # noqa: E402
    _DEFAULT_SNAPSHOT_DIR,
    _save_frame,
    _timestamped_snapshot,
    MjpegServer,
    build_detector,
    compose,
)


# ---------------------------------------------------------------- 고정 case
#  compose() 가 읽는 필드만 채우면 된다.
#    level_mm / level_pct / tank_state / pump_state / pump_duty / dist_mm
#    flags / link_ok()
TANK_MM = 300          # 물통 높이(mm) - compose 의 게이지 눈금 기준
EMPTY_MM = 50          # 빈통 문턱(mm)
TOF_MAX = 800.0        # config.TOF_OUT_OF_RANGE_MM 와 같은 값

CASES = {
    # 보고서/시연용 정상 화면. 마커 보임, 물통 OK, 펌프 ON, 좌우 벽 잡힘.
    "good": {
        "desc": "정상 - 마커 검출 + 물통 OK + 펌프 ON + 좌우 벽 감지",
        "level_mm": 246, "tank_state": "OK",
        "pump_state": "ON", "pump_duty": 153,      # 60%
        "tof": (212.0, 198.0),
        "markers": [3],
        "link": True,
    },
    "full": {
        "desc": "물 가득 - 펌프 대기(OFF)",
        "level_mm": 288, "tank_state": "FULL",
        "pump_state": "OFF", "pump_duty": 0,
        "tof": (205.0, 205.0),
        "markers": [3],
        "link": True,
    },
    "low": {
        "desc": "저수위 경고 - 노란 게이지, 펌프는 계속 ON",
        "level_mm": 78, "tank_state": "LOW",
        "pump_state": "ON", "pump_duty": 153,
        "tof": (168.0, 243.0),                      # 왼쪽으로 붙은 상태
        "markers": [3],
        "link": True,
    },
    "empty": {
        "desc": "빈통 - 펌프 LOCK(펌웨어 인터록)",
        "level_mm": 32, "tank_state": "EMPTY",
        "pump_state": "LOCK", "pump_duty": 0,
        "tof": (198.0, 206.0),
        "markers": [3],
        "link": True,
    },
    "furrow_end": {
        "desc": "고랑 끝 - 좌우 ToF 모두 out-of-range(--), 마커 없음",
        "level_mm": 210, "tank_state": "OK",
        "pump_state": "OFF", "pump_duty": 0,
        "tof": (None, None),
        "markers": [],
        "link": True,
    },
    "megadown": {
        "desc": "메가 끊김 - 수위/펌프 불명(FAULT)",
        "level_mm": 0, "tank_state": "FAULT",
        "pump_state": "?", "pump_duty": 0,
        "tof": (216.0, 190.0),
        "markers": [3],
        "link": False,
        "dist_mm": None,
    },
}
DEFAULT_CASE = "good"


class FixedLink:
    """메가 링크 자리에 끼우는 **고정값** 링크. 값이 절대 변하지 않는다."""

    def __init__(self, case: dict, tank_mm: int = TANK_MM):
        self.available = case.get("link", True)
        self.last_error = ""
        self.level_mm = int(case["level_mm"])
        self.level_pct = int(round(self.level_mm * 100.0 / max(1, tank_mm)))
        self.tank_state = case["tank_state"]
        self.pump_state = case["pump_state"]
        self.pump_duty = int(case["pump_duty"])
        self.flags = 1 if self.tank_state == "EMPTY" else 0
        # 초음파 거리 = (센서 높이 - 수위). 물통 위 40mm 에 달린 것으로 가정.
        self.dist_mm = case.get("dist_mm", tank_mm + 40 - self.level_mm)
        self._ok = bool(case.get("link", True))

    def link_ok(self):
        return self._ok

    def set_pump(self, duty):       # 고정 화면이므로 무시
        pass

    def send(self, text):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------- 합성 배경
def _aruco_tile(marker_id, side_px, dictionary):
    """검출 잘 되게 여백(quiet zone) 붙인 마커 타일을 만든다."""
    img = cv2.aruco.generateImageMarker(dictionary, int(marker_id),
                                        int(side_px))
    q = max(6, int(side_px) // 8)
    tile = np.full((int(side_px) + 2 * q, int(side_px) + 2 * q), 245, np.uint8)
    tile[q:q + int(side_px), q:q + int(side_px)] = img
    return cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)


def furrow_scene(w, h, dictionary, marker_id=None, marker_side_px=90,
                 marker_center=None):
    """고랑 안에서 팻말을 바라보는 장면을 합성한다(난수 시드 고정 = 매번 동일).

    marker_side_px 로 표기 거리가 결정된다.
        dist ≈ fx * 마커실크기 / 세로변픽셀,  fx = (w/2)/tan(62°/2)
        640px / 0.20m 마커 / 90px  ->  약 1.18 m
    """
    img = np.zeros((h, w, 3), np.uint8)
    horizon = int(h * 0.34)

    # 하늘(위로 갈수록 밝은 파랑) - BGR
    for y in range(horizon):
        t = y / max(1, horizon)
        img[y, :] = (int(196 - 26 * t), int(170 - 18 * t), int(140 - 14 * t))
    # 흙바닥(아래로 갈수록 밝고 붉게)
    for y in range(horizon, h):
        t = (y - horizon) / max(1, h - horizon)
        img[y, :] = (int(58 + 32 * t), int(76 + 52 * t), int(94 + 74 * t))

    # 좌/우 이랑(벽). 소실점 쪽으로 모이는 사다리꼴.
    top = horizon + int(h * 0.02)
    left_wall = np.array([[0, h], [int(w * 0.27), h],
                          [int(w * 0.45), top], [int(w * 0.39), top]], np.int32)
    right_wall = np.array([[w, h], [int(w * 0.73), h],
                           [int(w * 0.55), top], [int(w * 0.61), top]], np.int32)
    for poly, shade_bgr in ((left_wall, (44, 62, 80)), (right_wall, (40, 57, 74))):
        cv2.fillConvexPoly(img, poly, shade_bgr)
        cv2.polylines(img, [poly], True, (86, 116, 146), 1, cv2.LINE_AA)

    # 흙덩이/자잘한 텍스처(시드 고정)
    rng = np.random.default_rng(7)
    for _ in range(140):
        y = int(rng.integers(horizon + 4, h))
        x = int(rng.integers(0, w))
        r = max(1, int(2 + 5 * (y - horizon) / max(1, h - horizon)))
        d = int(rng.integers(-22, 23))
        base = img[y, x].astype(int) + d
        cv2.circle(img, (x, y), r, tuple(int(np.clip(v, 0, 255)) for v in base),
                   -1, cv2.LINE_AA)

    # 팻말 + 마커
    if marker_id is not None:
        tile = _aruco_tile(marker_id, marker_side_px, dictionary)
        th, tw = tile.shape[:2]
        cx, cy = marker_center or (int(w * 0.45), int(h * 0.42))
        x0, y0 = int(cx - tw / 2), int(cy - th / 2)
        x0 = max(4, min(w - tw - 4, x0))
        y0 = max(4, min(h - th - 4, y0))
        # 지지대
        cv2.rectangle(img, (cx - 4, y0 + th), (cx + 4, h - int(h * 0.06)),
                      (52, 62, 74), -1)
        # 팻말 테두리(흰 여백 밖으로 한 겹)
        cv2.rectangle(img, (x0 - 5, y0 - 5), (x0 + tw + 5, y0 + th + 5),
                      (206, 210, 214), -1)
        img[y0:y0 + th, x0:x0 + tw] = tile
    return img


def load_background(path, w, h):
    """유니코드 경로에서도 되는 배경 사진 로더."""
    buf = np.fromfile(str(Path(path).expanduser()), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"배경 사진을 읽을 수 없습니다: {path}")
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# ------------------------------------------------- 마커 주석(실기와 같은 표기)
def annotate_markers(frame, dic, params, det, marker_cm):
    """bringup_monitor.main() 의 마커 표기(EST 경로)와 같은 그림을 그린다.

    보정 파일 없이 화각 근사값(fx)을 쓰므로 실기와 동일하게 'EST' 로 뜬다.
    검출된 ID 목록을 돌려준다.
    """
    import math

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if det is not None:
        corners, ids, _ = det.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dic, parameters=params)
    found = []
    if ids is None:
        return found

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    fx = (frame.shape[1] / 2.0) / math.tan(math.radians(62.0) / 2)
    for c, i in zip(corners, ids.flatten()):
        q = c.reshape(4, 2)
        found.append(int(i))
        vpx = (np.linalg.norm(q[0] - q[3]) + np.linalg.norm(q[1] - q[2])) / 2.0
        dist = fx * (marker_cm / 100.0) / max(1.0, vpx)
        u = float(q[:, 0].mean())
        bear = math.degrees(math.atan2(frame.shape[1] / 2 - u, fx))
        cv2.putText(frame, f"ID{int(i)} {dist:.2f}m {bear:+.0f}deg EST",
                    (int(q[:, 0].min()), int(q[:, 1].min()) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 2,
                    cv2.LINE_AA)
    return found


# ---------------------------------------------------------------- 한 장 만들기
def render_case(name, args, dic, params, det, cap=None, background=None):
    """case 이름 -> 오버레이가 얹힌 최종 프레임 1장.

    background 를 주면 그 프레임을 배경으로 쓴다(모든 case 를 **같은** 카메라
    화면 위에 그려 비교할 때 사용). 원본은 건드리지 않는다.
    """
    case = CASES[name]
    marker_id = (case["markers"][0] if case["markers"] else None)

    if background is not None:
        frame = (background.copy()
                 if (background.shape[1] == args.width and
                     background.shape[0] == args.height)
                 else cv2.resize(background, (args.width, args.height)))
    elif cap is not None:
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.full((args.height, args.width, 3), 40, np.uint8)
            cv2.putText(frame, "NO FRAME", (args.width // 2 - 80,
                                            args.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2)
        elif frame.shape[1] != args.width or frame.shape[0] != args.height:
            frame = cv2.resize(frame, (args.width, args.height))
    elif args.image:
        frame = load_background(args.image, args.width, args.height)
    else:
        frame = furrow_scene(args.width, args.height, dic, marker_id,
                             args.marker_px)

    # 마커는 항상 **실제로 검출**한다(실기와 같은 표기).
    # 카메라를 쓸 때는 검출 결과를 그대로 믿는다 — 안 보이면 "없음" 이 맞고,
    # 그게 카메라/마커 상태를 확인하는 목적이다.
    # 합성/사진 배경에서 검출이 안 되면 case 에 적힌 ID 로 채워 준다.
    found = annotate_markers(frame, dic, params, det, args.marker_cm)
    if not found and cap is None:
        found = list(case["markers"])

    link = FixedLink(case, args.tank_mm)
    return compose(frame, link, case["tof"], found, args.tank_mm,
                   args.empty_mm, args.tof_max, link.pump_duty, args.fps)


def _config_defaults():
    """config.py 값을 쓰되, 없으면(설정 import 실패) 안전한 기본값."""
    try:
        from config import CAMERA_INDEX, CAMERA_RESOLUTION, MARKER_SIZE_M

        return (int(CAMERA_INDEX), int(CAMERA_RESOLUTION[0]),
                int(CAMERA_RESOLUTION[1]), float(MARKER_SIZE_M) * 100.0)
    except Exception as exc:
        print(f"config 로드 실패({exc}) - 기본값 640x480 / 카메라 0 을 씁니다.")
        return 0, 640, 480, 20.0


def main():
    cam_index, cam_w, cam_h, marker_cm = _config_defaults()

    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=DEFAULT_CASE, choices=sorted(CASES),
                    help=f"띄울 고정 case (기본 {DEFAULT_CASE})")
    ap.add_argument("--list-cases", action="store_true", help="case 목록만 출력")
    ap.add_argument("--width", type=int, default=cam_w)
    ap.add_argument("--height", type=int, default=cam_h)
    ap.add_argument("--fps", type=float, default=30.0,
                    help="화면에 표시할 fps 숫자(고정)")
    ap.add_argument("--tank-mm", type=int, default=TANK_MM)
    ap.add_argument("--empty-mm", type=int, default=EMPTY_MM)
    ap.add_argument("--tof-max", type=float, default=TOF_MAX)
    ap.add_argument("--marker-cm", type=float, default=marker_cm,
                    help="마커 한 변(cm). 기본값은 config.MARKER_SIZE_M")
    ap.add_argument("--marker-px", type=int, default=90,
                    help="합성 배경의 마커 크기(px). 작을수록 먼 거리로 표기됨")
    ap.add_argument("--image", help="배경으로 쓸 사진(카메라 대신)")
    ap.add_argument("--camera", type=int, default=cam_index,
                    help=f"실제 웹캠 인덱스 (기본 config.CAMERA_INDEX={cam_index})")
    ap.add_argument("--no-camera", action="store_true",
                    help="웹캠도 쓰지 않고 합성 고랑 장면으로 대체")
    ap.add_argument("--warmup", type=float, default=1.0,
                    help="웹캠 예열 시간(초). 첫 프레임이 검게 나오는 것 방지")
    ap.add_argument("--stream", type=int, default=0, help="MJPEG 포트(0=안 씀)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--snapshot", metavar="PATH",
                    help="한 장 저장하고 종료")
    ap.add_argument("--snapshot-all", metavar="DIR",
                    help="모든 case 를 DIR 에 한 장씩 저장하고 종료")
    ap.add_argument("--snapshot-dir", default=str(_DEFAULT_SNAPSHOT_DIR),
                    help="창에서 s 키로 저장할 폴더")
    args = ap.parse_args()

    if args.list_cases:
        for name in sorted(CASES):
            print(f"{name:11s} {CASES[name]['desc']}")
        return 0

    dic, params, det = build_detector()

    # --- 카메라만 진짜 장치 ---
    cap = None
    if not args.no_camera and not args.image:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            print(f"경고: 웹캠 {args.camera} 를 열지 못했습니다 - 합성 배경을 씁니다.")
            cap.release()
            cap = None
        else:
            # 웹캠은 자동노출/화이트밸런스가 잡히기 전 프레임이 검게 나온다.
            deadline = time.monotonic() + max(0.0, args.warmup)
            while time.monotonic() < deadline:
                cap.read()
            got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print(f"웹캠 {args.camera} 열림 {got[0]}x{got[1]} - "
                  f"ToF/수위/펌프는 case '{args.case}' 고정값입니다.")

    # --- case 전체 저장 ---
    #     모든 case 가 같은 카메라 화면 위에 그려지도록 프레임 1장만 잡는다.
    if args.snapshot_all:
        shot = None
        if cap is not None:
            ok, shot = cap.read()
            if not ok:
                shot = None
        for name in sorted(CASES):
            out = render_case(name, args, dic, params, det, cap, shot)
            saved = _save_frame(Path(args.snapshot_all) / f"fixed_{name}.jpg",
                                out)
            print("저장:", saved)
        if cap is not None:
            cap.release()
        return 0

    # --- 한 장 저장 ---
    if args.snapshot:
        out = render_case(args.case, args, dic, params, det, cap)
        print("저장:", _save_frame(args.snapshot, out))
        if cap is not None:
            cap.release()
        return 0

    order = sorted(CASES)
    idx = order.index(args.case)
    stream = MjpegServer(args.stream) if args.stream else None
    if stream:
        print(f"MJPEG: http://<이 장치 IP>:{args.stream}/  (case={order[idx]})")

    print(f"case '{order[idx]}' : {CASES[order[idx]]['desc']}")
    if not args.headless:
        print("키: q/ESC 종료, c 다음 case, s 저장")

    # 배경이 정지 화면이면 매 프레임 다시 그릴 필요가 없다.
    static_bg = cap is None
    frame_cache = {}

    try:
        while True:
            name = order[idx]
            if static_bg:
                if name not in frame_cache:
                    frame_cache[name] = render_case(name, args, dic, params,
                                                    det, None)
                out = frame_cache[name]
            else:
                out = render_case(name, args, dic, params, det, cap)

            if stream:
                stream.publish(out)

            if args.headless:
                time.sleep(0.05)
                continue

            cv2.imshow("Agri-X bring-up (fixed)", out)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("c"):
                idx = (idx + 1) % len(order)
                print(f"case '{order[idx]}' : {CASES[order[idx]]['desc']}")
            elif k == ord("s"):
                print("저장:", _save_frame(
                    _timestamped_snapshot(args.snapshot_dir), out))
    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

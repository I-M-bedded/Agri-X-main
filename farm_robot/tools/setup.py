# -*- coding: utf-8 -*-
"""
tools/setup.py
---------------
현장 브링업(bring-up) 도구 모음. 실주행 전에 이 파일 하나만 실행하면 됩니다.

    $ python3 tools/setup.py

메뉴
  1) 센서 배선 점검   - 모터를 돌리지 않고 ToF/카메라/수위/엔코더만 확인
  2) 모터·부호 점검   - 바퀴를 띄운 상태에서 조향 부호와 저속 구동을 확인
  3) 흙 색상 보정     - 비전이 흙을 제대로 인식하도록 HSV 임계값 조절

반드시 1 -> 2 -> 3 순서로 진행하세요.
2번은 바퀴가 실제로 돕니다. 로봇을 받침대에 올려 바퀴를 띄우고 실행하세요.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math  # noqa: E402

import config  # noqa: E402


# ======================================================================
# 공용 유틸
# ======================================================================
def title(text):
    print("\n" + "=" * 64)
    print(f" {text}")
    print("=" * 64)


def ask(question: str) -> bool:
    while True:
        answer = input(f"  -> {question} (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def hint(sign_name: str):
    print(f"     [조치] config.py 의 {sign_name} 부호를 뒤집으세요 (+1 <-> -1).")


def mega_firmware_hint(name: str):
    print(
        f"     [조치] firmware/agrix_motor_mega/agrix_motor_mega.ino 의 {name} "
        "값을 점검한 뒤 Mega에 다시 업로드하세요."
    )


def hold_drive(motors, command, duration: float, refresh_sec: float = 0.10):
    """Keep a DRIVE command alive without weakening the Mega's 400 ms watchdog."""
    deadline = time.monotonic() + max(0.0, float(duration))
    try:
        while True:
            if not command():
                raise RuntimeError(motors.last_error or "Mega DRIVE 송신 실패")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(refresh_sec, remaining))
    finally:
        motors.stop()


# ======================================================================
# 1) 센서 배선 점검 - 모터를 돌리지 않는다
# ======================================================================
def check_sensors() -> int:
    print("\n모터는 움직이지 않습니다. 안심하고 실행하세요.")
    ok_all = True

    # ---------------- ToF ----------------
    title("1-1. ToF 거리 센서 (VL53L1X x2)")
    from sensors.tof_sensor import ToFPair

    pair = ToFPair(config.TOF_LEFT, config.TOF_RIGHT)
    if not pair.hardware_ok():
        ok_all = False
        print("  [실패] ToF 초기화 실패.")
        print("    - sudo raspi-config 에서 I2C 활성화 확인")
        print("    - i2cdetect -y 1 실행 -> 실행 중에 0x30, 0x31 이 보여야 정상")
        print(f"    - XSHUT 배선 확인 (BCM): 좌={config.TOF_LEFT['xshut_pin']} "
              f"우={config.TOF_RIGHT['xshut_pin']}")
    else:
        print("  [OK] ToF 2개 초기화 성공")
        print("  10초간 값을 출력합니다. 각 센서 앞에 손을 대었다 떼 보세요.")
        left_vals, right_vals = [], []
        t_end = time.monotonic() + 10.0
        while time.monotonic() < t_end:
            l, r = pair.read()
            left_vals.append(pair.left.last_raw_mm)
            right_vals.append(pair.right.last_raw_mm)
            print(f"    좌={l:7.1f}mm  우={r:7.1f}mm  "
                  f"고랑끝판정={pair.both_out_of_range()}")
            time.sleep(0.3)

        for name, vals in (("좌", left_vals), ("우", right_vals)):
            spread = max(vals) - min(vals)
            if spread < 20.0:
                ok_all = False
                print(f"  [의심] {name} ToF 값이 거의 변하지 않았습니다 "
                      f"(변동 {spread:.0f}mm). 센서가 죽었거나 주소가 겹쳤을 수 있습니다.")
            else:
                print(f"  [OK] {name} ToF 반응 정상 (변동 {spread:.0f}mm)")
    pair.close()

    # ---------------- 카메라 ----------------
    title("1-2. 카메라 + 비전")
    from sensors.camera import Camera
    from sensors.vision_line_detector import VisionLineDetector

    cam = Camera()
    if not cam.available:
        ok_all = False
        print("  [실패] 카메라를 열지 못했습니다.")
        print("    - CSI 카메라: rpicam-hello 로 확인, 리본 케이블 방향 확인")
        print("    - USB 웹캠: ls /dev/video* 확인 후 config.CAMERA_INDEX 조정")
    else:
        det = VisionLineDetector(cam)
        print("  5초간 비전 결과를 출력합니다. 카메라를 고랑 쪽으로 향하게 하세요.")
        confs = []
        t_end = time.monotonic() + 5.0
        while time.monotonic() < t_end:
            res = det.compute()
            if res is None:
                print("    프레임 없음")
            else:
                confs.append(res.confidence)
                print(f"    오차={res.normalized_error:+.2f} "
                      f"기울기={res.heading_error:+.2f} "
                      f"커버리지={res.coverage:.2f} 신뢰도={res.confidence:.2f}")
            time.sleep(0.4)

        if confs and max(confs) >= config.VISION_MIN_CONFIDENCE:
            print(f"  [OK] 비전 신뢰도 최대 {max(confs):.2f}")
        else:
            print(f"  [의심] 신뢰도가 계속 {config.VISION_MIN_CONFIDENCE} 미만입니다.")
            print("    -> 이 도구의 3번 메뉴(흙 색상 보정)를 실행하세요.")
    cam.close()

    # ---------------- 수위 ----------------
    title("1-3. 수위 센서")
    from sensors.water_tank_sensor import WaterTankSensor

    ws = WaterTankSensor()
    print("  5초간 상태를 확인합니다. 센서를 물에 넣었다 빼 보세요.")
    states = set()
    t_end = time.monotonic() + 5.0
    while time.monotonic() < t_end:
        ws.poll()
        raw = ws._read_raw()
        states.add(raw)
        print(f"    원시신호={'물부족' if raw else '정상'}  "
              f"확정상태={'물부족' if ws.is_water_low() else '정상'}")
        time.sleep(0.5)
    if len(states) == 1:
        print("  [참고] 상태가 한 번도 바뀌지 않았습니다. 실제로 담갔다 빼서 재확인하세요.")
        print("    항상 '물부족'으로 나온다면 config.WATER_LOW_SIGNAL_ACTIVE_HIGH 를")
        print("    반대로 바꾸거나 WATER_SENSOR_PULL 을 조정하세요.")
    else:
        print("  [OK] 수위 센서 상태 전환 확인")
    ws.cleanup()

    # ---------------- 엔코더 ----------------
    title("1-4. 엔코더")
    from sensors.odometry import Odometry

    odom = Odometry()
    mega = None
    try:
        if getattr(config, "ODOMETRY_SOURCE", "") == "mega_usb":
            from control.mega_motion import MegaMotion

            mega = MegaMotion(odometry=odom)
            encoder_ready = mega.available and not mega.faulted
            if not encoder_ready:
                ok_all = False
                print(f"  [실패] Mega Motion USB 연결 실패: {mega.last_error}")
                print("    - Mega에 agrix_motor_mega.ino 업로드 여부 확인")
                print("    - /dev/ttyACM0 또는 /dev/serial/by-id 장치 확인")
                print("    - Mega 엔코더: M1 D2/D3, M2 D18/D19 배선 확인")
        else:
            encoder_ready = odom.is_available()
            if not encoder_ready:
                ok_all = False
                print("  [실패] 엔코더 GPIO 초기화 실패.")

        if encoder_ready:
            print("  10초간 바퀴를 **손으로** 굴려 보세요. 값이 올라가야 합니다.")
            start = odom.total_ticks
            t_end = time.monotonic() + 10.0
            while time.monotonic() < t_end:
                odom.update()
                print(f"    누적 갱신={odom.total_ticks}  "
                      f"주행거리={odom.path_length:.3f}m  헤딩={odom.theta:+.2f}rad")
                time.sleep(0.5)
            if odom.total_ticks == start:
                ok_all = False
                if mega is not None:
                    print("  [실패] Mega STATE에서 엔코더 변화가 전혀 들어오지 않았습니다.")
                    print("    - M1 엔코더 D2/D3, M2 엔코더 D18/D19 배선 확인")
                    print("    - 펌웨어 ENCODER1_SIGN / ENCODER2_SIGN 및 CPR 확인")
                else:
                    print(f"  [실패] 틱이 전혀 들어오지 않았습니다. 핀 확인 (BCM): "
                          f"좌={config.ENCODER_PINS['left']} 우={config.ENCODER_PINS['right']}")
            else:
                print(f"  [OK] 엔코더 갱신 {odom.total_ticks - start}회 수신")
                if mega is not None:
                    print("  [참고] Mega가 출력축 각도를 계산하므로 Pi의 TICKS_PER_REVOLUTION은")
                    print("         사용하지 않습니다. CPR은 Mega 펌웨어에서 관리합니다.")
                else:
                    print("  [참고] 바퀴를 정확히 10바퀴 돌린 뒤 '누적 틱 / 10' 을")
                    print("         config.TICKS_PER_REVOLUTION 에 넣으세요.")
    finally:
        if mega is not None:
            mega.cleanup()
        odom.cleanup()

    title("센서 점검 결과")
    if ok_all:
        print("  통과. 다음은 2번 메뉴(모터·부호 점검)입니다.")
        print("  [주의] 2번부터는 바퀴가 실제로 돕니다. 받침대에 올려 띄우세요.")
        return 0
    print("  실패 항목이 있습니다. 위 안내대로 배선을 고친 뒤 다시 실행하세요.")
    return 1


# ======================================================================
# 2) 모터·부호 벤치 점검 - 바퀴가 실제로 돈다
# ======================================================================
def check_motors() -> int:
    from control.mega_motion import MegaMotion
    from actuators.pump_controller import PumpController
    from sensors.odometry import Odometry

    print("\n" + "!" * 64)
    print(" 경고: 바퀴가 실제로 회전합니다.")
    print(" 로봇을 받침대에 올려 바퀴를 공중에 띄운 뒤 진행하세요.")
    print("!" * 64)
    if not ask("바퀴가 바닥에서 떠 있습니까?"):
        print("  중단합니다. 바퀴를 띄운 뒤 다시 실행하세요.")
        return 1

    odom = Odometry()
    motors = MegaMotion(odometry=odom)
    problems = []

    if not motors.available or motors.faulted:
        print(f"  [실패] Mega Motion 연결 실패: {motors.last_error}")
        motors.cleanup()
        odom.cleanup()
        return 1

    try:
        # ---------- 좌/우 식별 ----------
        title("2-1. 좌/우 바퀴 식별")
        print("  좌측 바퀴만 전진합니다 (2초)")
        hold_drive(motors, lambda: motors.set_speeds(0.4, 0.0), 2.0)
        if not ask("실제로 '왼쪽' 바퀴가 '전진' 방향으로 돌았습니까?"):
            problems.append("좌측 모터")
            print("     [조치] 다른 바퀴가 돌면 Mega M1/M2 드라이버 배선을 교환하세요.")
            mega_firmware_hint("MOTOR1_FORWARD_SIGN (방향이 반대인 경우)")

        print("\n  우측 바퀴만 전진합니다 (2초)")
        hold_drive(motors, lambda: motors.set_speeds(0.0, 0.4), 2.0)
        if not ask("실제로 '오른쪽' 바퀴가 '전진' 방향으로 돌았습니까?"):
            problems.append("우측 모터")
            print("     [조치] 다른 바퀴가 돌면 Mega M1/M2 드라이버 배선을 교환하세요.")
            mega_firmware_hint("MOTOR2_FORWARD_SIGN (방향이 반대인 경우)")

        # ---------- 조향 부호 ----------
        title("2-2. 조향 부호 (가장 중요)")
        print("  drive(base=0.3, steer=+0.2) - 규약상 '오른쪽으로 조향' (2초)")
        hold_drive(motors, lambda: motors.drive(0.3, 0.2), 2.0)
        if not ask("좌측 바퀴가 우측보다 '빠르게' 돌았습니까? (= 우회전 자세)"):
            problems.append("조향 믹싱")
            print("     [조치] Pi의 arcade mix는 정상 규약입니다. Mega M1/M2가 실제")
            print("            좌/우 바퀴와 바뀌어 연결되지 않았는지 확인하세요.")

        # ---------- 엔코더 방향 ----------
        title("2-3. 엔코더 방향")
        print("  양 바퀴 전진 (3초)")
        odom.reset()
        before = odom.total_ticks
        hold_drive(motors, lambda: motors.forward(0.4), 3.0)
        odom.update()
        gained = odom.total_ticks - before
        print(f"     갱신 수신: {gained}회, 추정 이동거리: {odom.x:+.3f} m")
        if gained == 0:
            problems.append("엔코더 신호 없음")
            print("     [조치] Mega 엔코더 배선 M1 D2/D3, M2 D18/D19를 확인하세요.")
        elif odom.x <= 0:
            problems.append("엔코더 방향")
            mega_firmware_hint("ENCODER1_SIGN / ENCODER2_SIGN")

        print("\n  제자리 좌회전(반시계) - 규약상 theta 가 '증가'해야 함 (2초)")
        odom.reset()
        hold_drive(
            motors,
            lambda: motors.rotate_in_place(clockwise=False, speed=0.4),
            2.0,
        )
        odom.update()
        print(f"     theta 변화: {odom.theta:+.3f} rad")
        if odom.theta <= 0:
            problems.append("헤딩 부호")
            print("     [조치] Mega M1/M2 엔코더가 서로 바뀌었거나 ENCODER*_SIGN이 반대입니다.")

        # ---------- 180도 정밀도 ----------
        title("2-4. 180도 회전 정밀도 (유턴에 직결)")
        input("  로봇 정면 방향을 바닥에 표시하고 Enter...")
        odom.reset()
        ok = motors.turn_180_blocking()
        odom.update()
        print(f"     엔코더 추정: {odom.theta:+.3f} rad "
              f"({odom.theta * 57.3:+.1f}도), 도달={ok}")
        print("     실제로 몇 도 돌았는지 눈으로 재보세요.")
        print("     오차가 크면 TRACK_WIDTH_M / TRACK_SLIP_FACTOR / WHEEL_RADIUS_M 을")
        print("     다시 실측하세요. Mega 엔코더 CPR은 펌웨어에서 관리합니다.")

        # ---------- 저속 구동 ----------
        title("2-5. 최소 안정 속도 명령 확인")
        print("  Mega는 PWM 듀티가 아니라 목표 RPM을 PID로 제어합니다.")
        print("  바퀴가 안정적으로 돌기 시작하는 정규화 속도 명령을 확인합니다.")
        found = None
        for command in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30]:
            hold_drive(
                motors,
                lambda command=command: motors.set_speeds(command, command),
                1.2,
            )
            moving = ask(f"속도 명령 {command:.2f} 에서 바퀴가 안정적으로 돌았습니까?")
            if moving:
                found = command
                break
        if found is not None:
            rpm = found * config.MEGA_DRIVE_MAX_RPM
            print(f"     [결과] 최소 안정 속도 명령 ~= {found:.2f} ({rpm:.1f} RPM 목표)")
            print("     [참고] config.MOTOR_MIN_DUTY는 Pi 직접 GPIO 경로용이며 Mega에는")
            print("            적용되지 않습니다. 저속 성능은 Mega PID/전원/마찰을 튜닝하세요.")
        else:
            problems.append("저속 구동 불안정")
            print("     [경고] 0.30 명령에서도 안 돌면 전원/드라이버/PID/기계 마찰을 확인하세요.")

        # ---------- 펌프 인터록 ----------
        title("2-6. 펌프 인터록")
        print("  물은 빼두는 것을 권장합니다.")
        pump = PumpController()
        got = pump.turn_on()
        print(f"     고랑 밖에서 turn_on() 결과 = {got}  (False 여야 정상)")
        if got:
            problems.append("펌프 인터록")
        pump.set_zone(True)
        pump.turn_on()
        print("     고랑 안 상태로 전환 후 ON. 릴레이가 '딸깍' 했는지 확인하세요.")
        time.sleep(1.5)
        pump.set_zone(False)
        print("     고랑 밖으로 전환 -> 즉시 OFF 되어야 합니다.")
        if pump.is_on():
            problems.append("펌프 OFF 실패")
        pump.cleanup()

    finally:
        motors.stop()
        motors.cleanup()
        odom.cleanup()

    title("모터·부호 점검 결과")
    if problems:
        print("  실패 항목:")
        for p in problems:
            print(f"   - {p}")
        print("  위 항목을 모두 해결한 뒤 실주행을 시도하세요.")
        return 1
    print("  통과. 다음은 3번 메뉴(흙 색상 보정)입니다.")
    return 0


# ======================================================================
# 3) 흙 색상(HSV) 캘리브레이션
# ======================================================================
def calibrate_soil_hsv(image_path=None) -> int:
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("OpenCV 가 필요합니다: sudo apt install python3-opencv")
        return 1

    from sensors.vision_line_detector import VisionLineDetector

    still = None
    camera = None
    if image_path:
        still = cv2.imread(image_path)
        if still is None:
            print(f"이미지를 열 수 없습니다: {image_path}")
            return 1
    else:
        from sensors.camera import Camera

        camera = Camera()
        if not camera.available:
            print("카메라를 열 수 없습니다. 사진 파일 경로를 인자로 주세요:")
            print("  python3 tools/setup.py 3 photo.jpg")
            return 1

    det = VisionLineDetector(camera=None)

    win = "HSV calibration"
    cv2.namedWindow(win)
    lo, hi = config.SOIL_HSV_LOWER, config.SOIL_HSV_UPPER
    for name, val, maxv in [
        ("H lo", lo[0], 179), ("S lo", lo[1], 255), ("V lo", lo[2], 255),
        ("H hi", hi[0], 179), ("S hi", hi[1], 255), ("V hi", hi[2], 255),
    ]:
        cv2.createTrackbar(name, win, val, maxv, lambda _: None)

    print("\n트랙바로 조절하세요.  s = 값 출력 / q = 종료")
    print("목표: 고랑 바닥이 하얗게 잡히고, 커버리지 0.25~0.75, "
          f"신뢰도 >= {config.VISION_MIN_CONFIDENCE}")
    print("맑을 때와 흐릴 때 각각 맞춰 보는 것을 권장합니다.")

    import sensors.vision_line_detector as vld

    while True:
        frame = still.copy() if still is not None else camera.capture_frame()
        if frame is None:
            continue

        lower = tuple(cv2.getTrackbarPos(n, win) for n in ("H lo", "S lo", "V lo"))
        upper = tuple(cv2.getTrackbarPos(n, win) for n in ("H hi", "S hi", "V hi"))

        # 검출기가 참조하는 임계값을 실시간으로 바꿔가며 결과를 본다
        vld.SOIL_HSV_LOWER = lower
        vld.SOIL_HSV_UPPER = upper
        result = det.compute_from_frame(frame)

        h, w = frame.shape[:2]
        y0 = int(h * config.VISION_ROI_Y_START_RATIO)
        y1 = int(h * config.VISION_ROI_Y_END_RATIO)
        hsv = cv2.cvtColor(frame[y0:y1, :], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        view = frame.copy()
        cv2.rectangle(view, (0, y0), (w - 1, y1), (0, 255, 255), 2)
        cx = int(w / 2 + result.normalized_error * (w / 2))
        cv2.line(view, (cx, y0), (cx, y1), (0, 0, 255), 3)      # 추정 중앙선
        cv2.line(view, (w // 2, y0), (w // 2, y1), (255, 255, 255), 1)  # 화면 중앙

        ok = result.confidence >= config.VISION_MIN_CONFIDENCE
        color = (0, 255, 0) if ok else (0, 0, 255)
        cv2.putText(
            view,
            f"cov={result.coverage:.2f} conf={result.confidence:.2f} "
            f"err={result.normalized_error:+.2f} {'OK' if ok else 'LOW'}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )

        cv2.imshow(win, view)
        cv2.imshow("mask (ROI)", mask)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            print("\n--- config.py 에 붙여넣으세요 ---")
            print(f"SOIL_HSV_LOWER = {lower}")
            print(f"SOIL_HSV_UPPER = {upper}")
            print(f"# 커버리지={result.coverage:.2f}, "
                  f"신뢰도={result.confidence:.2f}\n")

    cv2.destroyAllWindows()
    if camera is not None:
        camera.close()
    return 0



# ======================================================================
# 4) 무한궤도 보정 - 회전 미끄러짐과 직진 거리
# ======================================================================
def calibrate_tracks() -> int:
    """
    궤도는 회전할 때 지면을 비비며 미끄러진다. 그래서 이론값대로 안 돈다.
    엔코더도 '궤도가 움직인 양'을 잴 뿐 로봇이 간 거리를 재지 않는다.

    이 두 오차는 흙 상태에 따라 달라져 계산으로 못 구한다. 재는 수밖에 없다.
    이 도구가 로봇을 움직여 주고, 사용자는 자로 재서 숫자만 입력하면 된다.
    """
    from control.mega_motion import MegaMotion
    from sensors.odometry import Odometry

    print("\n" + "!" * 64)
    print(" 무한궤도 보정")
    print(" 로봇이 실제로 움직입니다. **바닥에 내려놓고** 진행하세요.")
    print(" (앞뒤로 3m, 좌우로 1m 정도 빈 공간이 필요합니다)")
    print(" 실제 밭과 **같은 흙 위**에서 재야 의미가 있습니다.")
    print("!" * 64)
    if not ask("바닥에 내려놓았고 주변이 비어 있습니까?"):
        print("  중단합니다.")
        return 1

    odom = Odometry()
    motors = MegaMotion(odometry=odom)
    results = {}

    if not motors.available or motors.faulted:
        print(f"  [실패] Mega Motion 연결 실패: {motors.last_error}")
        motors.cleanup()
        odom.cleanup()
        return 1

    try:
        # ---------------- 직진 거리 보정 ----------------
        title("4-1. 직진 거리 보정")
        print("  로봇을 앞으로 2m 정도 보냅니다.")
        print("  바닥에 출발 지점을 표시하세요 (테이프나 돌).")
        input("  준비되면 Enter...")

        odom.reset()
        target = 2.0
        t0 = time.monotonic()
        try:
            while odom.path_length < target and time.monotonic() - t0 < 30.0:
                if not motors.forward(0.4):
                    raise RuntimeError(motors.last_error or "Mega DRIVE 송신 실패")
                odom.update()
                time.sleep(0.05)
        finally:
            motors.stop()
        odom.update()
        est = odom.path_length
        print(f"\n  엔코더 추정 이동거리: {est:.3f} m")

        while True:
            try:
                actual = float(input("  실제로 몇 m 이동했습니까? (자로 재서 입력): "))
                if actual > 0:
                    break
            except ValueError:
                pass
            print("    숫자를 입력하세요 (예: 1.85)")

        factor = actual / est if est > 0 else 1.0
        results["DISTANCE_CALIBRATION_FACTOR"] = round(factor, 3)
        print(f"  -> DISTANCE_CALIBRATION_FACTOR = {factor:.3f}")
        if factor < 0.85:
            print("     [경고] 궤도가 15% 이상 헛돕니다. 장력이나 지면을 확인하세요.")

        # ---------------- 회전 미끄러짐 보정 ----------------
        title("4-2. 회전 미끄러짐 보정 (가장 중요)")
        print("  제자리에서 180도 회전을 명령합니다.")
        print("  로봇 정면 방향을 바닥에 표시하세요.")
        input("  준비되면 Enter...")

        odom.reset()
        motors.turn_180_blocking()
        odom.update()
        print(f"\n  엔코더는 {math.degrees(abs(odom.theta)):.0f}도 돌았다고 봅니다.")
        print("  실제로 몇 도 돌았는지 재세요.")
        print("    정확히 반대를 보고 있으면 180, 덜 돌았으면 그보다 작은 값.")

        while True:
            try:
                real_deg = float(input("  실제 회전각(도): "))
                if 30.0 < real_deg < 330.0:
                    break
            except ValueError:
                pass
            print("    30~330 사이의 숫자를 입력하세요 (예: 140)")

        # 명령 180도 대비 실제로 돈 각도의 비율만큼 유효 궤도폭을 넓힌다
        slip = 180.0 / real_deg
        current = config.TRACK_SLIP_FACTOR
        new_slip = current * slip
        results["TRACK_SLIP_FACTOR"] = round(new_slip, 3)
        print(f"  -> TRACK_SLIP_FACTOR = {new_slip:.3f}  (현재 {current})")
        if new_slip > 1.8:
            print("     [경고] 미끄러짐이 매우 큽니다. 궤도 장력 또는 지면을 확인하세요.")
            print("            이 상태로는 유턴 정밀도가 낮아 IMU 추가를 권합니다.")

        # ---------------- 최소 안정 속도 ----------------
        title("4-3. 최소 안정 속도 명령")
        print("  Mega PID에서 궤도가 안정적으로 움직이기 시작하는 속도를 확인합니다.")
        found = None
        for command in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
            hold_drive(
                motors,
                lambda command=command: motors.set_speeds(command, command),
                1.2,
            )
            moving = ask(f"속도 명령 {command:.2f} 에서 궤도가 움직였습니까?")
            if moving:
                found = command
                break
        if found is not None:
            rpm = found * config.MEGA_DRIVE_MAX_RPM
            print(f"  -> 최소 안정 속도 명령 ~= {found:.2f} ({rpm:.1f} RPM 목표)")
            print("     config.MOTOR_MIN_DUTY는 Mega 경로에 적용되지 않습니다.")
        else:
            print("     [경고] 0.45 명령에서도 안 움직입니다. 전원/드라이버/PID를 확인하세요.")

    finally:
        motors.stop()
        motors.cleanup()
        odom.cleanup()

    # ---------------- 결과 ----------------
    title("보정 결과 - config.py 에 붙여넣으세요")
    if not results:
        print("  측정된 값이 없습니다.")
        return 1
    print()
    for k, v in results.items():
        print(f"  {k} = {v}")
    print()
    print("  위 값을 config.py 에서 찾아 바꾸면 됩니다.")
    print("  바꾼 뒤 tools/setup.py 2번으로 180도 회전이 맞는지 다시 확인하세요.")
    return 0

# ======================================================================
def main():
    menu = {
        "1": ("센서 배선 점검 (모터 안 돎)", check_sensors),
        "2": ("모터·부호 점검 (바퀴 띄우고!)", check_motors),
        "3": ("흙 색상 보정 (HSV)", None),
        "4": ("무한궤도 보정 (회전·거리·저속)", calibrate_tracks),
    }

    # 인자로 바로 실행:  python3 tools/setup.py 2
    choice = sys.argv[1] if len(sys.argv) > 1 else None
    image_arg = sys.argv[2] if len(sys.argv) > 2 else None

    if choice is None:
        print("=" * 64)
        print(" 농장 로봇 현장 브링업 도구")
        print("=" * 64)
        for key, (label, _) in menu.items():
            print(f"  {key}) {label}")
        print("  q) 종료")
        print("\n  1 -> 2 -> 4 -> 3 순서를 권합니다.")
        choice = input("\n선택: ").strip()

    if choice in ("q", "Q"):
        return 0
    if choice == "3":
        return calibrate_soil_hsv(image_arg)
    if choice in menu:
        return menu[choice][1]()

    print(f"알 수 없는 선택: {choice}")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단됨.")
        sys.exit(130)
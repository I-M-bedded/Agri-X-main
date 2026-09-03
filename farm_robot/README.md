# 농장형 자율주행 급수 로봇

라즈베리파이4 + Arduino Mega USB 모션 제어기 + RGB 카메라 + 1D ToF ×2 +
수위센서 기반 고랑 급수 로봇의 제어 소프트웨어.

**이 파일은 설치·운용 절차서입니다.** 아래 순서대로 따라가세요.

> 코드를 처음 보시나요? **`OVERVIEW.md`** 를 먼저 읽으세요.
> 로봇이 뭘 하는지, 파일이 각각 무슨 역할인지, 배선을 어떻게 하는지
> 그림과 함께 정리되어 있습니다.

---

## 0. 명령어 4개만 기억하면 됩니다

```bash
python3 selftest.py       # 코드가 멀쩡한지 확인 (하드웨어 불필요)
python3 tools/setup.py    # 현장 브링업 (센서→모터→비전 순서로 안내해 줌)
python3 main.py           # 실제 임무
```

설치는 최초 1회만 하며, 명령어는 6-2 절에 그대로 복사해 쓸 수 있게 정리해 뒀습니다.

---

## 1. 파일 구조

```
farm_robot/
├── main.py                    실행 진입점
├── config.py            ★    당신이 만지게 될 유일한 파일
├── logutil.py                 공용 로거 (11개 모듈이 임포트 - 지우면 안 됨)
├── selftest.py                자체 점검 (150개 항목)
├── OVERVIEW.md                처음 보는 사람용 전체 설명서
├── VISION_NAVIGATION_REVIEW.md  SegFormer 중심선·FSM·HOME 연동 검토
├── WIRING.md                  배선도 (핀 번호 + 외부 결선 전체)
│
├── sensors/                   센서 읽기
│   ├── camera.py                  카메라 (항상 BGR 반환 보장)
│   ├── tof_sensor.py              ToF ×2 + I2C 주소 재할당
│   ├── odometry.py                엔코더 → 위치/각도 추정
│   ├── aruco_detector.py          마커 → 입구 위치/진입각
│   ├── vision_line_detector.py    영상 → 고랑 중앙선 오차
│   ├── onnx_furrow_line_detector.py  비동기 SegFormer ONNX 어댑터
│   ├── ccrdnet_line_detector.py   비동기 CCRDNet 3-클래스 어댑터 (CONTROL_PIPELINE.md)
│   ├── furrow_geometry.py         line mask → Hough 중심선
│   └── water_tank_sensor.py       수위 (디바운스)
│
├── actuators/                 물리 출력
│   ├── motor_driver.py            Gazebo/내부 시뮬레이션용 모터
│   └── pump_controller.py         릴레이 + 3중 인터록
│
├── control/                   주행 제어
│   ├── pid_controller.py          와인드업 방지 PID
│   ├── mega_motion.py             Mega USB DRIVE/MOVE 통합 인터페이스
│   └── line_follower.py           비전(주) + ToF/엔코더(보조) 융합
│
├── navigation/                의사 결정
│   ├── furrow_manager.py          고랑 번호 관리
│   └── mission_state_machine.py   전체 임무 FSM
│
└── tools/
    ├── setup.py                   현장 브링업 도구 (메뉴 4개)
    └── simulation.py              selftest 용 시뮬레이터
```

`config.py` 를 제외한 나머지는 **원칙적으로 손대지 않습니다.**
현장에서 뭔가 반대로 동작하면 코드가 아니라 config 의 `SIGN_*` 를 뒤집으세요.

---

## 2. 부호 규약 — 가장 중요한 4줄

새 센서를 붙이거나 배선을 바꿀 때 반드시 확인하세요.
실기가 안 움직이는 원인 1순위가 여기입니다.

1. `theta` (헤딩) : **반시계(좌회전)가 양수**
2. `error` (오차) : **양수 = 오른쪽으로 가야 함**, 무차원(−1~+1)
3. `steer` (PID 출력) : **양수 = 오른쪽(시계방향) 조향**
4. 바퀴 믹싱 : `left = base + steer`, `right = base − steer`

부호가 반대로 나오면 **코드를 고치지 말고** `config.py` 맨 위의
`SIGN_TOF_ERROR`, `SIGN_VISION_ERROR`, `SIGN_MARKER_LATERAL`,
`SIGN_MARKER_YAW`, `SIGN_HEADING_ERROR`, `SIGN_LEFT_MOTOR`,
`SIGN_RIGHT_MOTOR`, `SIGN_LEFT_ENCODER`, `SIGN_RIGHT_ENCODER` 를
`+1 ↔ −1` 로 뒤집으세요.

---

## 3. 마커는 입구에만 — 주행의 핵심

마커는 **가장 어려운 문제(입구 식별)만** 담당합니다. 고랑 안에서는 로봇이
스스로 밭을 보고 달립니다.

| 문제 | 해결 |
|---|---|
| 몇 번 고랑인가 / 입구가 어느 방향인가 | 마커 |
| **고랑 중심이 어디인가** | **비전** |
| **고랑이 끝났는가** | **ToF + 비전** |
| 집이 어디인가 / 임무가 끝났는가 | HOME·END 마커 |

### 마커는 중심선을 알려주지 못합니다 ★

마커는 "여기 고랑 입구가 있다"만 말해줍니다. 중심선이 마커에서 왼쪽
30cm 인지 오른쪽 50cm 인지는 **마커 안에 담긴 정보가 아닙니다.**

그 값을 상수로 넣어두면 현장에서 팻말을 조금만 옮겨 박아도 로봇이
그만큼 빗나가 이랑으로 돌진합니다. **그런 사전 지식은 쓰지 않습니다.**
중심선은 비전이 고랑 자체를 보고 찾습니다.

실제로 팻말 위치를 바꿔가며 검증했습니다. **단, 비전이 정상일 때의 결과입니다.**

| 팻말 위치 | 결과 | 정렬 좌우 이탈 |
|---|---|---|
| 오른쪽 30cm | 완주 | 평균 0.5cm |
| 오른쪽 45cm | 완주 | 평균 0.4cm |
| **왼쪽** 35cm | 완주 | 평균 0.3cm |
| 오른쪽 15cm | 완주 | 평균 0.3cm |

**팻말은 고랑이든 이랑이든, 로봇이 잘 볼 수 있는 곳에 세우면 됩니다.**
다만 중심선에서 너무 멀면 안 됩니다 — 아래 한계선을 보세요.

### 팻말을 얼마나 옆으로 박아도 되는가 ★

코드가 팻말 위치를 모른다는 것과, 얼마나 치우쳐 박아도 되는지는 다른
문제입니다. 너무 멀면 로봇이 팻말 앞에 도착했을 때 정작 **정렬할 고랑이
센서 범위 밖**에 있습니다.

| 횡오프셋 | 비전 정상 | 비전 실명(흐린 날) |
|---|---|---|
| 0.20m | 완주 | 완주 |
| 0.28m | 완주 | 완주 |
| 0.31m | 완주 | 완주 |
| **0.33m** | 완주 | **SAFE_HALT (0/3)** |
| 0.60m | 완주 | SAFE_HALT |
| **0.65m** | **SAFE_HALT (2/3)** | SAFE_HALT |

한계가 둘이고 근거가 다릅니다.

- **약 0.32m — ToF 탐침의 한계 ≈ 고랑 폭.** 비전이 안 보이면 로봇은 팻말
  방위로 다가간 뒤 코를 들이미는데, 팻말이 고랑 폭보다 더 옆에 있으면
  로봇 코가 이미 이랑 위에 올라가 있어 좌우 벽을 못 잡습니다.
  (근거 상수: 고랑 반폭 = `TOF_NOMINAL_WALL_DISTANCE_MM` 150mm)
- **약 0.62m — 비전의 한계.** 이보다 멀면 고랑이 아예 화면에 안 잡힙니다.
  (근거 상수: `tools/simulation.py` 의 `VISION_ENTRANCE_CAPTURE_M` 0.6)

> **권장: 0.15 ~ 0.20m.**
> 기본값 `0.30m` 은 첫 번째 한계선 바로 밑이라 여유가 거의 없습니다.
> 현장에서 팻말이 몇 cm 밀리면 **맑은 날엔 멀쩡히 돌다가 흐린 날에만
> SAFE_HALT** 나는, 원인 찾기 제일 나쁜 형태의 고장이 됩니다.
>
> 위 숫자는 **고랑 폭 30cm 가정**에서 나왔습니다. 실제 고랑 폭을 잰 뒤
> 다시 뽑으세요. 일반화하면 **"고랑 폭 이내, 여유 있게 그 절반쯤"** 입니다.

### 입구 정렬 3단계

1. **마커 방위로 거친 접근** — 비전이 아직 고랑을 못 볼 때
2. **비전으로 중심 정렬** — 고랑이 화면에 들어오면 중앙선 오차로 조향
3. **진입각만 제자리 회전으로 조정** — 위치가 안 변하니 중심이 유지됨

진입각은 마커 자세(yaw)로 재지 않습니다. 포즈 모호성 때문에 프레임마다
수십 도씩 튀거든요. `_field_heading`(밭 안쪽 방위)을 씁니다.

### 비전이 고랑을 못 볼 때 — ToF 탐침

비전만 믿으면 흐린 날이나 비 온 뒤 로봇이 **아예 못 들어갑니다.**
그래서 ToF 대체 경로를 둡니다.

```
고랑에 코를 조금씩 들이민다
  양쪽 다 벽 보임      → 고랑 안. 좌우 거리차로 중심을 맞춘다
  한쪽만, 아주 가까이  → 이랑에 붙었다. 반대쪽으로 조향
  양쪽 다 안 보임      → 아직 앞. 조금 더 전진 (최대 0.8m)
```

중심을 **벽 대칭**으로 직접 찾습니다. 비전과 하는 일은 같지만 근거가
달라서(색상 vs 거리) 조명이 나빠도 동작합니다.

탐침 전에 **밭 안쪽으로 자세를 먼저 잡습니다.** 안 그러면 마커를 향하던
헤딩 그대로 직진해 팻말 쪽으로 밀려납니다(중심에서 28cm 벗어나는 것을
재현했습니다).

**검증**: 비전을 완전히 실명시켜도 고랑 3개 전부 급수하고 HOME 복귀합니다.

탐침마저 실패하면(0.8m 전진해도 벽 없음) `SAFE_HALT` 합니다.
`ENTRANCE_ALLOW_BLIND_CREEP=True` 로 두면 마커만 믿고 들어가지만,
이랑을 밟을 수 있어 권하지 않습니다.

### 제어원 우선순위 (고랑 안)

| 순위 | 소스 | 역할 |
|---|---|---|
| 1 | **비전 (흙 HSV)** | 중앙선 추종 |
| 2 | ToF ×2 | 좌우 벽 대칭. 비전에 25% 섞음 |
| 3 | 엔코더 | 헤딩 유지 (최후 폴백) |

### 고랑 끝 판정 (출구 마커 없음)

**주 근거는 ToF 양쪽 벽 소실.** 오판 방지 장치가 셋입니다.

1. 양쪽 동시 — 한쪽만 사라지면 이랑 유실 구간
2. 연속 확인 `TOF_END_CONFIRM_TICKS`(3틱)
3. 최소 주행거리 `FURROW_END_MIN_TRAVEL_M`(1.0m)

`FURROW_END_REQUIRE_VISION_AGREE=True` 로 두면 비전 신뢰도 급락까지
함께 요구합니다. 기본값은 `False` 입니다.

**복귀할 때는 입구 마커가 정면에 나타나므로 더 쉽습니다.**

---

## 4. 임무 흐름 (FSM)

```
INIT → SEARCH_AND_ALIGN ──────────────┐
         ↓ 입구 정렬 완료              │
      TRAVEL_INTO_FURROW   (펌프 ON)   │
         ↓ 좌우 ToF 모두 벽 없음       │
      TURN_AROUND          (180° 유턴) │
         ↓                             │
      TRAVEL_BACK          (펌프 ON)   │
         ↓ 입구 도달                   │
      EXIT_FURROW          (펌프 OFF)  │
         ↓                             │
      EVALUATE_MISSION ────────────────┤ 물 부족? → 목표를 HOME 으로
         ↓                             │
      HEADLAND_TRANSIT ────────────────┘
      (90° 선회 → 옆으로 이동 → 90° 선회 → 다음 고랑 탐색)

      END 마커 확인 → HOME 복귀 → MISSION_COMPLETE
      판단 불가 상황 → SAFE_HALT (사람 대기)
```

**헤드랜드 이동**이 핵심입니다. 다음 고랑 입구는 옆으로 수 미터 떨어져
있어서 제자리 회전만으로는 물리적으로 도달할 수 없습니다.
회전은 현재 각도 기준 상대 90°가 아니라 `_field_heading`(밭 안쪽 기준
방위)에 대한 **절대 각도**로 계산하므로, 로봇이 어느 쪽을 보고 있든
결과가 같습니다.

### HOME 복귀 방식

HOME 은 **1번 고랑 입구**입니다. 5번 고랑에서 물이 떨어졌다면 옆으로 4칸을
되돌아가야 합니다.

로봇은 자기가 몇 번째 고랑에 있는지 알고 있으므로 **돌아갈 칸 수를 직접
계산**합니다. 도착 예정 지점 전까지는 중간 정거장에서 전체 탐색을 건너뛰고
이동만 반복하며, 각 정거장에서 마커를 **한 번만** 훑어봅니다.

> [수정/성능] 예전에는 정거장마다 HOME 마커를 360도 전체 탐색했습니다.
> 그런데 5번 고랑 자리에서 1번 고랑의 HOME 마커가 보일 리가 없습니다.
> 있지도 않은 마커를 찾느라 정거장마다 수십 초를 버렸습니다.
> **고랑 12개 밭 측정: HOME 복귀 404초 → 80초 (5배 단축).**
> 헛도는 동안 다른 고랑의 마커를 HOME 으로 오검출할 위험도 있었습니다.

오도메트리가 어긋나 예상보다 일찍 HOME 이 보이면 즉시 정렬하고, 늦게
도착하면 남은 이동 허용 횟수 안에서 계속 찾습니다. 이동 허용 횟수는
`지나온 고랑 수 + 3` 이므로 왔던 거리보다 더 멀리 가는 일은 없습니다.

---

## 5. 안전 설계

- **확실한 근거 없이는 진행하지 않는다.** 애매하면 `SAFE_HALT` 후 사람 대기.
  "다음 마커가 안 보임"은 완료가 아니라 **모름**입니다. 완료 확정은
  END 마커(ID 998)를 정면 가까이에서 실제로 봤을 때만.
- **모든 전진에 시간/거리 상한**이 있습니다.
- **펌프 3중 방어**: 고랑 밖이면 즉시 OFF / 물 부족 시 잠금 / 최대 연속
  가동 시간 워치독.
- **마지막 고랑도 반드시 급수**: END 마커는 "여기까지 물을 주고 끝내라"는
  뜻이지 "즉시 돌아가라"가 아니다.
- **HOME 복귀 보장**: HOME(1번 고랑 입구)까지 되돌아갈 칸 수를 로봇이 직접
  계산한다. 고랑이 몇 개든 밭 한가운데서 멈추지 않는다. (고랑 16개 밭까지
  자체 점검으로 검증)
- **물 부족 시 즉시 유턴**: 살수가 불가능한 상태로 고랑 끝까지 헛돌지 않는다
  (`WATER_LOW_ABORT_INBOUND_LEG`). 복귀 주행은 고랑을 빠져나오는 유일한
  경로이므로 생략할 수 없다.
- **회전은 반드시 끝난다**: 부호 교차 판정 + 절대 타임아웃 + 엔코더 stall 감지.
- **SIGTERM 처리**: `systemctl stop` 시 모터·펌프를 정리하고 종료합니다.
- **출발 전 하드웨어 점검**: 센서가 죽어 있으면 바퀴를 굴리지 않습니다.

---

## 6. 라즈베리파이에 올리기

### 6-1. GitHub 경유
```bash
# PC에서
git init && git add . && git commit -m "농장 급수 로봇"
git branch -M main
git remote add origin https://github.com/<계정>/farm_robot.git
git push -u origin main

# 라즈베리파이에서
sudo apt update && sudo apt install -y git
cd ~ && git clone https://github.com/<계정>/farm_robot.git
cd farm_robot
```
핀 배치와 마커 정보가 들어 있으니 **비공개 저장소**를 권장합니다.

이후 코드 갱신은 PC에서 `git push`, Pi에서 `git pull` 입니다.
단 `config.py` 는 현장에서 계속 만지게 되므로, Pi에서 고친 값은
PC로 다시 옮겨 커밋하세요(안 그러면 다음 `git pull` 에서 충돌).

### 6-2. 설치 (최초 1회, 그대로 복사해서 붙여넣으세요)

```bash
# (1) 시스템 패키지
sudo apt update
sudo apt install -y python3-venv python3-pip python3-dev \
    python3-opencv python3-numpy python3-picamera2 python3-rpi.gpio \
    i2c-tools git

# (2) I2C / 카메라 활성화  (0 = enable)
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_camera 0

# (3) I2C 속도 400kHz 로 상향 (ToF 2개 폴링 여유 확보)
echo "dtparam=i2c_arm=on,i2c_arm_baudrate=400000" | sudo tee -a /boot/firmware/config.txt

# (4) 하드웨어 접근 권한
sudo usermod -aG i2c,gpio,video $USER

# (5) 가상환경  ★ --system-site-packages 필수
cd ~/farm_robot
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# (6) ToF 라이브러리
pip install --upgrade pip
pip install adafruit-circuitpython-vl53l1x adafruit-blinka

# (7) 재부팅 (I2C 설정과 그룹 권한 적용)
sudo reboot
```

재부팅 후에는 작업 전에 매번:
```bash
cd ~/farm_robot && source .venv/bin/activate
```

설치 확인:
```bash
python3 -c "import cv2, numpy, RPi.GPIO, adafruit_vl53l1x, picamera2; print('전부 OK')"
```

> **(5)번의 `--system-site-packages` 가 핵심입니다.**
> picamera2 는 시스템 libcamera 와 묶여 있어 pip 설치본이 동작하지 않고,
> opencv-python 은 Pi4 에서 소스 빌드로 넘어가 수십 분이 걸립니다.
> 둘 다 apt 로 깔고(1번), venv 가 그걸 볼 수 있게 해야 합니다.
> 이 옵션 없이 만들면 `ModuleNotFoundError: picamera2` 가 납니다.

> **git 관련**: `__pycache__`, `.venv`, `*.log` 는 커밋하지 않는 게 좋습니다.
> 프로젝트 루트에 `.gitignore` 파일을 만들어 아래 네 줄을 넣으세요.
> ```
> __pycache__/
> .venv/
> *.log
> *.pyc
> ```

---

## 7. 배선

| 기능 | config 키 | 기본값 |
|------|-----------|--------|
| Mega USB | `SERIAL_MEGA_PORT` | `/dev/ttyACM0`, 115200 |
| 좌 BTS7960 RPWM / LPWM | Mega 펌웨어 | D6 / D7 |
| 우 BTS7960 RPWM / LPWM | Mega 펌웨어 | D44 / D45 |
| 펌프 릴레이 | `PUMP_RELAY_PIN` | 26 |
| 좌 ToF XSHUT | `TOF_LEFT` | 5 (물리핀 29) |
| 우 ToF XSHUT | `TOF_RIGHT` | 6 (물리핀 31) |
| 수위 센서 | `WATER_LEVEL_SENSOR_PIN` | 22 |
| 좌 엔코더 A / B | Mega 펌웨어 | D2 / D3 |
| 우 엔코더 A / B | Mega 펌웨어 | D18 / D19 |
| I2C (ToF 공통) | 고정 | SDA=2, SCL=3 |

> **전체 배선도는 `WIRING.md` 를 보세요.** 물리 핀 번호, 전원 계통,
> 모듈별 단자 대응, 배선 후 점검 순서가 모두 들어 있습니다.

**전원 경고**
- 모터·펌프 전원을 라즈베리파이 5V 핀에서 뽑지 마세요. 기동 전류로 전압이
  주저앉아 Pi가 리부팅되거나 SD카드가 손상됩니다. 별도 배터리를 쓰고
  **GND만 Pi와 공통**으로 연결하세요.
- 펌프 릴레이는 플라이백 다이오드가 있는 모듈을 쓰세요.
- ToF 2개는 VIN/GND/SDA/SCL 을 공통으로 묶고 **XSHUT 만 따로** 뺍니다.

---

## 8. 브링업 — 이 순서를 지키세요

### 8-1. 로직 점검
```bash
python3 selftest.py
```
`150/150 통과` 가 나와야 합니다. 여기서 실패하면 실기에서는 확실히 실패합니다.
(이 파일과 `tools/simulation.py` 는 열어볼 필요 없습니다. 명령만 돌리세요)

### 8-2. 현장 브링업 도구
```bash
python3 tools/setup.py
```
메뉴 4개가 나옵니다. **1 → 2 → 4 → 3 순서로** 진행하세요.

**1) 센서 배선 점검** — 모터가 돌지 않습니다.
ToF 2개 초기화·값 변화, 카메라 프레임, 수위센서, 엔코더 틱을 확인합니다.
ToF가 안 잡히면 `i2cdetect -y 1` 을 보세요.
프로그램 실행 **중**에 `0x30`, `0x31` 두 개가 보여야 정상입니다.
계속 `0x29` 하나만 보이면 XSHUT 배선 문제입니다.

**2) 모터·부호 점검** — **바퀴가 실제로 돕니다. 받침대에 올려 띄우세요.**
좌우 식별, 조향 부호, 엔코더 방향, 180° 회전 정밀도, 데드밴드 실측,
펌프 인터록을 하나씩 확인합니다.
여기서 측정한 최소 구동 듀티를 `config.MOTOR_MIN_DUTY` 에 반영하세요.

**4) 무한궤도 보정** — **바닥에 내려놓고** 실행합니다.
직진 2m·180도 회전을 시킨 뒤 실제 값을 자로 재서 입력하면, config 에
붙여넣을 세 줄을 출력해 줍니다. **실제 밭 흙 위에서 재야 의미가 있습니다.**

**3) 흙 색상 보정** — 트랙바로 HSV 임계값을 조절합니다.
목표: 커버리지 0.25~0.75, 신뢰도 ≥ `VISION_MIN_CONFIDENCE`.
**맑을 때와 흐릴 때 각각** 맞춰 보세요. `s` 키를 누르면 config 에
붙여넣을 형태로 출력됩니다.
사진 파일로 하려면: `python3 tools/setup.py 3 photo.jpg`

**웹캠 + ToF + Mega 물통/펌프 오버레이 촬영**

```bash
# 창에서 실시간 확인: s 키로 media/bringup에 저장
python3 tools/bringup_monitor.py

# 3초 예열 후 사진 1장 저장하고 자동 종료
python3 tools/bringup_monitor.py --headless \
  --snapshot ../media/bringup/agrix_demo.jpg

# 헤드리스: http://<Pi IP>:8080 에서 실시간 확인/사진 다운로드
python3 tools/bringup_monitor.py --headless --stream 8080
```

상단 오버레이에 Pi 물리핀 `XSHUT=29/31`, `I2C=3/5`와
Mega `TRIG/ECHO=D30/D31`, `PWM=D9`가 표시됩니다. Mega에는
`agrix_tank_pump.ino`가 올라가 있어야 수위·펌프 값이 실값으로 보입니다.

**카메라만 진짜 + 나머지는 고정값 오버레이** — ToF(`adafruit_vl53l1x`)나
Mega 가 아직 안 붙어도, **실제 웹캠 화면 위에** 오버레이가 어떻게 얹히는지
확인할 수 있습니다. 영상과 ArUco 마커 검출만 실제이고 좌우 ToF·수위·펌프·
링크 상태는 고정값이라 화면이 흔들리지 않습니다. `bringup_monitor.py` 의
`compose()` 를 그대로 import 하므로 레이아웃은 실기 모니터와 같은 코드가
그립니다.

```bash
# 실제 웹캠(config.CAMERA_INDEX) + good case 고정 오버레이
#   c 키로 case 순환, s 키로 media/bringup 에 저장
python3 tools/bringup_overlay_fixed.py

# case 목록: good / full / low / empty / furrow_end / megadown
python3 tools/bringup_overlay_fixed.py --list-cases
python3 tools/bringup_overlay_fixed.py --camera 2 --case low

# 전체 case 를 같은 카메라 화면 위에 한 장씩 저장
python3 tools/bringup_overlay_fixed.py --snapshot-all ../media/bringup_fixed

# 헤드리스 Pi: http://<Pi IP>:8080
python3 tools/bringup_overlay_fixed.py --stream 8080

# 웹캠도 없을 때: 합성 고랑 장면 + 렌더링된 마커로 대체
python3 tools/bringup_overlay_fixed.py --no-camera
```

### 8-3. 현장에서 반드시 맞춰야 하는 값 ★

기본값 그대로 두면 **로봇은 반드시 실패합니다.** 아래 순서대로 채우세요.
`tools/setup.py` 가 대부분 자동으로 재줍니다.

#### A. 자로 재는 값 (5분)

| 값 | 재는 법 |
|---|---|
| `TRACK_WIDTH_M` | 좌우 궤도 접지면 **중심 간** 거리 |
| `WHEEL_RADIUS_M` | 스프로킷 중심 ~ 궤도 접지면 (궤도 두께 포함) |
| `MARKER_SIZE_M` | 인쇄한 마커 한 변 |
| `HEADLAND_STEP_DISTANCE_M` | 고랑 중심 간 간격 |

#### B. `setup.py 2번`이 재주는 값

| 값 | 의미 |
|---|---|
| `SIGN_*` 9개 | 부호가 반대면 뒤집기. **안 움직이는 원인 1순위** |
| `TICKS_PER_REVOLUTION` | 궤도 10바퀴 돌려 `총 틱 ÷ 10` |

#### C. `setup.py 4번`이 재주는 값 — 궤도의 핵심 ★

| 값 | 기본 | 의미 |
|---|---|---|
| `TRACK_SLIP_FACTOR` | 1.3 | **회전 정밀도를 좌우.** 180도 명령 대비 실제 회전각 비율 |
| `DISTANCE_CALIBRATION_FACTOR` | 1.0 | 실제 이동거리 ÷ 엔코더 추정거리 |
| `MOTOR_MIN_DUTY` | 0.18 | 궤도가 겨우 움직이는 듀티. **궤도는 0.25~0.40** |

> `TRACK_SLIP_FACTOR` 가 틀리면 유턴과 헤드랜드 선회가 전부 어긋나
> 밭 밖으로 나갑니다. 궤도에서 가장 중요한 값입니다.

#### D. `setup.py 1번`으로 확인하며 넣는 값

| 값 | 의미 |
|---|---|
| `TOF_NOMINAL_WALL_DISTANCE_MM` | 고랑 중앙에서 좌우 ToF 평균 |
| `TOF_OUT_OF_RANGE_MM` | 위 값의 3~4배. 이보다 멀면 "벽 없음" |
| `ENTRANCE_PROBE_TOO_CLOSE_MM` | (고랑폭 − 궤도폭) ÷ 2 보다 **작게** |

#### E. `setup.py 3번`으로 맞추는 값

| 값 | 의미 |
|---|---|
| `SOIL_HSV_LOWER` / `UPPER` | 흙 색상 범위. **맑을 때·흐릴 때 각각** |

#### F. 별도 작업이 필요한 값

`CAMERA_MATRIX` / `DIST_COEFFS` — 체스보드 20장으로 `cv2.calibrateCamera`.
**건너뛰면 마커 거리 추정이 부정확해 입구 정렬이 계속 실패합니다.**

### 8-4. 카메라 캘리브레이션 — 건너뛰면 정렬이 계속 실패합니다
기본 설정은 **내부 코너 9×6**, 즉 인쇄된 무늬가 **10×7칸**인 보드입니다.
한 칸의 크기를 자로 재어 `--square-mm`에 넣으세요.

```bash
cd farm_robot
python3 tools/webcam_checkerboard_calibration.py --square-mm 25 --auto
```

보드를 화면 중앙에만 두지 말고 모서리·가장자리·가까운 거리·먼 거리와
여러 기울임으로 움직이세요. `20/20`이 되면 자동 계산합니다.
수동 모드에서는 `SPACE`로 수집하고 `C` 또는 `ENTER`로 계산합니다.

결과는 `calibration/webcam_0.json`에 저장되며 ArUco 검출기와
`bringup_monitor.py`가 자동으로 읽습니다. 웹캠 해상도·초점·줌을 바꾸면
다시 보정해야 합니다. 평균 재투영 오차가 `1.2 px` 이상이면 샘플을 다시 찍으세요.

### 8-5. 저속 단계 주행
`BASE_SPEED = 0.2`, `LOG_LEVEL = "DEBUG"` 로 두고 아래 순서로 넓혀 가세요.
각 단계마다 **비상 전원 차단 스위치에 손을 올려두세요.**

1. 물 없이 빈 고랑 1개 왕복
2. 물 넣고 고랑 1개
3. 고랑 2개 (헤드랜드 이동 포함)
4. 전체 밭

---

## 9. 마커 배치

고랑 하나에 팻말이 **1개** 붙습니다. **입구에만** 붙이고, 출구에는 없습니다.

| 위치 | 마커 ID |
|---|---|
| N번 고랑 **입구** | `N` |
| HOME (1번 고랑 입구) | `0` |
| **마지막 고랑 입구에 추가** | **`249` (END)** |

고랑 3개면 입구 팻말 `1`,`2`,`3` / HOME 팻말에 `0` /
3번 고랑 입구 팻말에 `249` 를 추가합니다.
**출구에는 아무것도 붙이지 않습니다.**

입구 팻말 ID = 고랑 번호입니다. 복귀 중 팻말 하나만 봐도 지금 몇 번 고랑
앞인지 바로 알 수 있습니다.

`config.py` 의 `furrow_marker_id(N)` 으로 확인할 수 있습니다.

**팻말 위치는 자유입니다.** 고랑이든 이랑이든, 왼쪽이든 오른쪽이든
로봇이 잘 볼 수 있는 곳에 세우면 됩니다. 코드는 팻말이 어디 있는지
모르고, 중심선은 비전이 직접 찾습니다.
다만 로봇 주행 경로를 막지 않아야 하고, 중심선에서 **0.15~0.20m** 정도로
지켜 주세요 (한계선과 근거는 3장 "팻말을 얼마나 옆으로 박아도 되는가" 참고).

### 인쇄

**`DICT_4X4_250`** 으로 인쇄하세요.

> [수정] 예전 설정은 `DICT_4X4_50` 이었는데 이 딕셔너리는 ID 0~49 만
> 존재합니다. 그런데 END 마커가 998 로 잡혀 있어 **인쇄 자체가 불가능**
> 했습니다. 그래서 `DICT_4X4_250` 으로
> 바꿨습니다. 4x4 는 5x5/6x6 보다 비트 수가 적어 **같은 크기에서 더
> 멀리서 검출**됩니다. 헤드랜드를 따라 복귀하며 옆 고랑 팻말을 60도
> 비스듬히(= `cos60` 배로 납작하게) 봐야 하므로 원거리 검출력이
> 중요합니다.

### 마커 크기 — 고랑 길이에 맞추세요

검출 가능 거리는 대략 이렇습니다.

```
검출거리 ≈ 마커 한 변(m) × 이미지 폭(px) / (2 × 마커 픽셀크기 하한)
```

`0.10m` 마커 + `640px` 폭이면 실질 한계가 **6~8m** 정도입니다.
고랑이 10m 이상이면 **0.15~0.20m 로 크게 인쇄하세요.**

고랑 안에서는 마커가 보이지 않으므로 비전과 ToF 만으로 달립니다.
입구 마커는 진입 정렬과 복귀 판정에 쓰입니다.

### 설치

- 평평한 판에 붙이세요. **휘면 자세 추정이 망가집니다.**
- 모든 팻말의 **높이를 서로 맞추세요.** (복귀 중 두 고랑의 팻말을 동시에
  보고 헤드랜드 방향을 계산합니다)
- 카메라 높이와 비슷한 높이가 가장 좋습니다.
- 팻말 위치는 자유입니다. 로봇이 잘 볼 수 있는 곳이면 됩니다.

### 팻말은 30도 기울여 박으세요 ★

팻말을 고랑에 **정면으로(수직으로)** 박으면, 로봇이 헤드랜드를 따라 옆으로
이동할 때 마커를 거의 옆면으로 보게 되어 검출되지 않습니다. 그러면 HOME
복귀 때 참고할 절대 기준점이 없어 엔코더로 칸 수를 세는 추측항법에 의존하게
되고, 바퀴가 한 번 미끄러지면 오차가 **누적**됩니다.

**전부 밭 끝(END) 쪽을 향해 30도 기울이세요.** 그러면 마커 하나가 두 방향에서
다 보입니다.

- 고랑에 정면 진입할 때 → 30도 비스듬히 → 검출 가능
- 헤드랜드를 따라 복귀할 때 → 60도 비스듬히 → 검출 가능

입구 팻말 ID = 고랑 번호이므로, 복귀 중 마커 `3` 이 보이면 "지금 3번 고랑
앞이다"를 알 수 있고, 매 정거장마다 **절대 위치와 방위로 리셋**됩니다.

#### 왜 하필 30도인가 (계산 근거)

| 기울기 | 정면 접근 가능 거리 | 헤드랜드 검출 거리 |
|---|---|---|
| 0도 | 0.30 ~ 4.0m | 0.8m ← 복귀 때 안 보임 |
| 20도 | 0.30 ~ 4.0m | 3.6m |
| **30도** | **0.40 ~ 4.0m** | **15m+** ← 최적 |
| 45도 | 0.70 ~ 4.0m | 15m+ ← 정면 근접이 빠듯 |
| 50도 | 0.95 ~ 4.0m | 15m+ ← 정면 근접 실패 |

고랑을 빠져나온 로봇은 게이트 앞 **0.5~0.7m** 에 서게 됩니다. 45도면 이
거리에서 마커가 시야각을 벗어납니다(시뮬레이션에서 HOME 을 못 찾고
밭 밖으로 나가는 것을 재현했습니다). 30도면 0.4m 까지 검출됩니다.

> **기울이는 방향이 일정해야 합니다.** 반대로 기울이면 복귀할 때 마커
> 뒷면만 보입니다.
>
> 30도로 보면 마커가 `cos30 = 0.87`배로 납작해 보입니다. 검출 거리가 약
> 15% 줄므로 `MARKER_SIZE_M` 을 **0.15 이상**으로 키우세요.

### END 마커(249)의 의미 ★

**마지막 고랑의 입구 팻말에 붙입니다.** 뜻은 이렇습니다.

> "이 고랑이 마지막이다. **이 고랑까지 물을 준 뒤** 임무를 끝내라."

즉 END 마커를 본다고 그 자리에서 돌아서는 게 아닙니다. 평소처럼 진입 →
살수 → 유턴 → 복귀를 마친 **다음에** HOME 으로 향합니다.

고랑이 3개라면 3번 고랑 입구 팻말에 `6`, `7`, 그리고 `249` 를 함께 붙입니다.
(왼쪽 팻말이든 오른쪽 팻말이든 상관없지만, 로봇이 정면 접근할 때 잘 보이는
쪽에 두세요.)

END 마커가 없으면 로봇은 다음 고랑을 찾다가 상한에 걸려 `SAFE_HALT` 로
멈춥니다. "마커가 안 보인다"는 완료가 아니라 **모름**이기 때문입니다.
이는 의도된 설계입니다 — 밭이 아직 남았는데 마커가 흙에 가려진 것일 수도
있으니까요.

## 10. 자동 실행 (전원만 넣으면 동작)

**8장을 모두 마치고 손으로 실행한 임무가 최소 한 번 성공한 뒤에만** 등록하세요.

**(1) 유닛 파일 작성**
```bash
sudo nano /etc/systemd/system/farm-robot.service
```
아래 내용을 붙여넣고 저장(Ctrl+O, Enter, Ctrl+X)하세요.
사용자 이름이 `pi` 가 아니면 `User=`, `Group=`, 경로 3곳을 바꾸세요.

```ini
[Unit]
Description=Farm Irrigation Robot
After=multi-user.target
Wants=multi-user.target

[Service]
Type=simple
User=pi
Group=pi
SupplementaryGroups=gpio i2c video

WorkingDirectory=/home/pi/farm_robot
ExecStart=/home/pi/farm_robot/.venv/bin/python3 /home/pi/farm_robot/main.py

# 로그가 버퍼에 갇히지 않고 즉시 journal 로 나가게 한다
Environment=PYTHONUNBUFFERED=1

# 안전 종료: main.py 가 SIGTERM 을 받아 모터/펌프를 정리한다
KillSignal=SIGTERM
TimeoutStopSec=15
KillMode=mixed
SendSIGKILL=yes

# 재시작 정책: SAFE_HALT(종료코드 2)는 재시작하지 않는다 (아래 표 참고)
Restart=on-failure
RestartPreventExitStatus=2
RestartSec=10
StartLimitIntervalSec=300
StartLimitBurst=3

[Install]
WantedBy=multi-user.target
```

**(2) 등록**
```bash
sudo systemctl daemon-reload
sudo systemctl enable farm-robot     # 부팅 시 자동 실행
sudo systemctl start farm-robot      # 지금 시작
```

```bash
sudo systemctl status farm-robot     # 상태
journalctl -u farm-robot -f          # 실시간 로그
sudo systemctl stop farm-robot       # 안전 정지 (모터·펌프 정리됨)
sudo systemctl disable farm-robot    # 자동 실행 해제
```

### 재시작 정책
`main.py` 는 결과를 종료 코드로 알립니다.

| 코드 | 의미 | systemd 동작 |
|------|------|--------------|
| 0 | 임무 정상 완료 | 재시작 안 함 |
| 2 | `SAFE_HALT` (사람 확인 필요) | **재시작 안 함** |
| 1 | 예기치 못한 오류 | 10초 후 재시작 (최대 3회) |

`Restart=always` 로 두면 안 됩니다. `SAFE_HALT` 는 "로봇이 스스로 판단하기를
포기하고 사람을 기다리는 상태"이며, 자동 재시작은 같은 사고를 무한 반복합니다.

---

## 11. 문제 해결

| 증상 | 확인할 것 |
|------|-----------|
| `ModuleNotFoundError: sensors` | 프로젝트 루트에서 실행했는지 (`cd ~/farm_robot`) |
| `ModuleNotFoundError: picamera2` | venv 를 `--system-site-packages` 로 만들었는지 |
| 사전 점검 실패 (ToF) | `i2cdetect -y 1` 로 0x30/0x31 확인, XSHUT 배선 |
| 고랑 진입 3초 후 SAFE_HALT | ToF가 벽을 못 봄. 장착 각도/높이, `TOF_OUT_OF_RANGE_MM` |
| 입구에서 15초 후 SAFE_HALT | 정렬 수렴 실패. 카메라 캘리브레이션, `MOTOR_MIN_DUTY` |
| "비전 신뢰도 부족" 반복 | `tools/setup.py` 3번으로 HSV 재튜닝 |
| 로봇이 오차를 키우며 벽으로 감 | **부호 문제**. `tools/setup.py` 2번 재실행 후 `SIGN_*` 뒤집기 |
| 제어 루프 주기 초과 경고 | `CAMERA_RESOLUTION` 을 (480,360)으로 또는 `CONTROL_LOOP_HZ` 를 10으로 |
| Pi가 갑자기 리부팅 | 모터/펌프 전원을 Pi에서 뽑고 있음. 전원 분리 필수 |

---

## 12. 알려진 한계

1. **바퀴 미끄러짐에 한계가 있음** — 자체 점검으로 검증한 범위는
   **좌우 대칭 미끄러짐 2% 이내**입니다. (마커가 입구에만 있는 배치 기준) 그 너머에서는
   폭주하지 않고 `SAFE_HALT` 로 멈추지만 임무를 마치지 못합니다.
   근본 해결은 **IMU(자이로) 추가**입니다. 회전 각도를 엔코더가 아니라
   자이로로 재면 미끄러짐과 무관해집니다. 무른 흙이 많은 밭이라면
   MPU-6050 급(1만원 이하) 추가를 권합니다.
2. **장애물 감지 없음** — 사람/동물/돌이 앞에 있어도 모릅니다.
   전방 ToF나 범퍼 스위치 추가를 강력히 권장합니다.
   **첫 주행에는 반드시 사람이 따라다니세요.**
3. **궤도 미끄러짐** — Mega의 A/B 쿼드러처는 모터 출력축 회전은 정확히
   재지만 흙 위에서 궤도가 미끄러진 거리는 알 수 없습니다. 정밀 회전에는
   IMU 융합이 필요합니다.
4. **추측항법 드리프트** — 회전이 많으면 헤딩 오차가 누적됩니다.
   근본 해결은 IMU(자이로) 융합입니다.
5. **비전이 색상 임계값 기반** — 조명 변화에 약합니다.
   그늘/역광이 심한 밭이라면 깊이 기반이 근본적입니다.
6. **밭 밖으로 나가는 것을 막는 물리 장치가 없음** — 소프트웨어 상한뿐입니다.

`selftest.py` 는 상태 전이·부호·워치독·인터록을 검증할 뿐,
바퀴 미끄러짐·조명 변화·모터 응답 지연은 검증하지 못합니다.
**"여기서 통과 = 실기에서 잘 달린다"가 아니라 "여기서 실패 = 실기에서는
확실히 실패한다"** 로 읽으세요.

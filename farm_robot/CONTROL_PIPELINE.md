# 제어 파이프라인 — 임무 시퀀스 · 센서 · 구동 계층

2026-08-31 기준. 설치·운용 절차는 `README.md`, 모델 학습 근거는 `../reports/`.

---

## 1. 임무 시퀀스 (합의된 운용 순서)

```text
[시작]
  │  ODOMETRY_BACKEND="encoder_imu" 면 정지 상태에서 자이로 바이어스 보정
  ▼
① 직선 주행으로 마커 탐색            SEARCH_AND_ALIGN (SEARCH_MODE="creep")
  │   밭 방위 유지하며 저속 직진, "조금 가고 → 멈춰서 촬영" 반복
  │   3m 안에 못 찾으면 → 그 자리에서 제자리 회전 탐색으로 전환
  │   그래도 없으면 → 헤드랜드로 한 칸 이동 후 재탐색
  ▼ 마커 발견
② 마커 앞에서 정렬                   SEARCH_AND_ALIGN (계속)
  │   마커 방위로 접근 → 정지 거리까지 → 진입각을 제자리 회전으로 보정
  │   ※ 마커는 "입구 위치"만 알려준다. 중심선은 알려주지 않는다.
  ▼ 정렬 완료 (_field_heading 갱신)
③ 비전으로 중심선 추정 + 추종        TRAVEL_INTO_FURROW (펌프 ON)
  │   비전(주) + 1D ToF ×2(보조 25%) + 엔코더 헤딩(최후 폴백)
  ▼ 좌우 ToF 양쪽 벽 소실 (연속 3틱 + 최소 주행거리)
④ 유턴                               TURN_AROUND (제자리 180°)
  ▼
⑤ 복귀 주행                          TRAVEL_BACK_TO_ENTRANCE (펌프 ON)
  ▼ 입구 도달
⑥ 고랑 이탈 → 다음 고랑 판단          EXIT_FURROW → EVALUATE_MISSION
  │
  ├─ 물 부족 / END 마커 확인  → HOME 복귀 (4절)
  └─ 계속                     → HEADLAND_TRANSIT → ① 로 반복
```

**시퀀스와 코드가 다른 점은 없다.** 단, ①의 "직선 주행 탐색"은 이번에
새로 추가한 것이다(기존 코드는 처음부터 제자리 회전 탐색이었다).
`SEARCH_MODE="rotate"` 로 두면 레거시 동작으로 되돌아간다.

> **왜 직선 탐색 뒤에 회전 탐색을 남겨 두는가**
> 팻말이 진행 방향 앞이 아니라 옆/뒤에 있으면 직진만으로는 영원히 못 찾는다.
> 반대로 직선 탐색을 무한정 늘리면 밭 안으로 계속 들어가 버린다. 그래서
> **직진은 거리 상한(3m)으로 끊고, 그 자리에서 한 바퀴 훑은 뒤,
> 헤드랜드로 위치를 옮긴다.** 모든 단계에 상한이 있어 폭주하지 않는다.

---

## 2. 한 틱(20 Hz)의 데이터 흐름 — 고랑 내부 주행

```text
Camera (BGR)
   │  비동기 제출 ≤5 fps  (VISION_ONNX_SUBMIT_INTERVAL_SEC)
   ▼
[비전 백엔드: VISION_BACKEND]                    별도 스레드, 최신 프레임만
   "hsv"            흙 HSV 밴드 무게중심 → 직선 피팅      (기본, 안전)
   "onnx_boundary"  2-클래스 경계 모델 + Hough
   "ccrdnet"        3-클래스 (OTHER/STRUCTURE/NAV_BAND)  ← CRDLD/CCRDNet
   ▼
VisionLineResult(normalized_error, heading_error, confidence, coverage)
   │
   ▼                          ┌───────────────────────────────┐
LineFollower.step()  20 Hz    │ ToFPair.read()  (틱당 1회 샘플) │
   │                          └───────────────────────────────┘
   │  오차 사다리 (위가 우선):
   │   1) conf ≥ VISION_MIN_CONFIDENCE
   │        error = 비전오차 + VISION_HEADING_WEIGHT × 헤딩오차
   │        ToF 유효 시 → error = 0.75×비전 + 0.25×ToF   ← ToF 로 중앙 맞춤
   │   2) 비전 탈락 → ToF 좌우 대칭 오차 단독
   │   3) ToF 도 무효 → 엔코더/IMU 헤딩 유지
   ▼
PID (와인드업 방지) → steer   (부호 규약: 양수 = 오른쪽 조향)
   ▼
구동 계층 drive(base_speed, steer)                   ← 3절
```

---

## 3. 구동 계층 (Mega Motion V2)

상위(FSM/LineFollower)는 **무차원 명령만** 주고, `MegaMotion`이 출력축
RPM 또는 상대 회전각으로 변환한다. 엔코더와 PID는 USB로 연결된 Mega가
소유한다.

```text
FSM / LineFollower       drive(base_speed, steer)       [-1, 1]
        │
        ▼
MegaMotion (Pi)          DRIVE left_rpm right_rpm       연속 주행
control/mega_motion.py   MOVE seq left_deg right_deg    정밀 회전
        │ USB CDC 115200
        ▼
Arduino Mega             1 kHz PID · 4체배 엔코더 · 20 kHz PWM
agrix_motor_mega.ino     400 ms DRIVE watchdog
```

- `DRIVE`는 20 Hz 제어 틱마다 갱신한다. 별도 background heartbeat를 보내지
  않으므로 Pi 제어 루프가 멎으면 Mega가 400 ms 안에 정지한다.
- `MOVE`를 기다리는 동안에만 200 ms heartbeat를 보내며 `DONE seq`까지
  블로킹한다. 90°/180° 회전은 이 경로를 사용한다.
- `STATE`는 Mega가 20 Hz 자동 전송하고 Pi 오도메트리에 좌우 출력축 각도
  증분을 주입한다.
- `actuators/motor_driver.py`는 Gazebo/내부 시뮬레이션 전용으로 남겨 둔다.

---

## 4. HOME 복귀 (물 부족 / 임무 종료)

### 트리거

| 트리거 | 감지 위치 | 동작 |
|---|---|---|
| 물 부족 (진입 중) | `TRAVEL_INTO_FURROW` | 고랑 끝까지 가지 않고 **즉시 유턴** → 복귀 |
| 물 부족 (그 외) | `EVALUATE_MISSION` | 이번 고랑 마치고 HOME 으로 |
| END 마커(249) | `SEARCH_AND_ALIGN` | **이 고랑까지 급수한 뒤** HOME 으로 |

### 수위 소스: Nano USB (`WATER_SOURCE="nano_usb"`)

Nano 가 USB 시리얼로 `empty` / `ok` 를 **주기적으로(권장 1초)** 보낸다.
1회성 이벤트만 보내면 그 순간 케이블이 빠졌을 때 로봇이 영영 모른다.

`sensors/nano_link.py` 의 3중 방어:

```text
디바운스   같은 상태가 WATER_LEVEL_DEBOUNCE_SEC 유지될 때만 확정
래치       한 번 '부족' 확정되면 재시작 전까지 유지 (WATER_EMPTY_LATCHED)
           → 물 출렁임으로 empty/ok 가 번갈아 와서 복귀를 취소하고
             밭 한가운데서 다시 급수를 시작하는 사고를 막는다
링크 감시  NANO_LINK_STALE_SEC 무수신 = '판단 불가'
           → 마지막 확정 상태 유지 (부족으로 단정하지 않음) + 경고 로그
```

인터페이스가 기존 GPIO 센서와 같아서 FSM 수정이 필요 없다.

### 복귀 방법 — "칸 수를 세어 되돌아간다"

HOME 은 **1번 고랑 입구**(마커 0)다. 로봇은 자기가 몇 번 고랑에 있는지 알고
있으므로 되돌아갈 칸 수를 **직접 계산**한다.

```text
남은 이동 횟수 = (현재 고랑 번호) − 1

  예) 5번 고랑에서 물이 떨어짐 → 헤드랜드로 4칸 되돌아감
      각 정거장에서 마커를 한 번만 훑어본다 (전체 360도 탐색 안 함)
      → 고랑 12개 밭 기준 404초 → 80초 (5배 단축)
```

- 오도메트리가 어긋나 **예상보다 일찍** HOME 마커가 보이면 즉시 정렬한다.
- **늦게** 도착하면 남은 허용 횟수(`지나온 고랑 수 + 3`) 안에서 계속 찾는다.
  왔던 거리보다 더 멀리 가는 일은 없다.
- 이 방식의 정확도는 **회전 정밀도**에 달려 있다 → IMU 오도메트리 권장(5절).

---

## 5. 오도메트리 / 엔코더 / IMU

### 엔코더: 쿼드러처 A/B (13 PPR × 4분주)

```text
ENCODER_QUAD_PINS = {"left_a":…, "left_b":…, "right_a":…, "right_b":…}
ENCODER_DECODE    = 4     # A/B 모든 에지. 1 로 낮추면 인터럽트 1/4
ENCODER_PPR       = 13.0
ENCODER_GEAR_RATIO= 1.0   # ★ 모터축→바퀴축 감속비 실측 필요
→ 바퀴 1회전당 카운트 = 13 × 4 × 기어비
```

- **위상으로 방향을 직접 안다** → 모터 명령으로 방향을 추정하던 단일 채널
  방식의 오차(관성·미끄러짐·브레이크 중 역회전)가 사라진다.
- `ENCODER_QUAD_PINS = None`(현재)이면 기존 단일 채널 경로로 동작한다.
  **핀맵을 받으면 채우기만 하면 된다.**
- Pi 의 Python 인터럽트가 버거우면 `ENCODER_DECODE = 1` 로 낮춘다.

### 오도메트리 백엔드

| `ODOMETRY_BACKEND` | 거리 | 방향(theta) |
|---|---|---|
| `"encoder"` (기본) | 엔코더 | 엔코더 좌우 차동 |
| `"encoder_imu"` | 엔코더 | **IMU 자이로 적분** |

궤도(트랙) 차체는 회전할 때 지면을 비비며 미끄러지므로, 엔코더 차동으로 잰
회전각은 흙 상태에 따라 크게 틀어진다(README 알려진 한계 1·4번). IMU 는
미끄러짐과 무관하게 실제로 돈 각도를 재므로 **유턴·헤드랜드 선회·HOME 복귀
칸 수 계산이 안정된다.**

- 시작 시 정지 상태에서 자이로 바이어스를 실측한다(`_state_init` 에서 자동).
- IMU 가 없거나 통신이 끊기면 **그 순간부터 엔코더 theta 로 자동 폴백**한다.
- 다른 IMU 로 바꾸려면 `read_yaw_rate() -> rad/s (CCW 양수)` 만 맞추면 된다.

---

## 6. 주요 설정 (`config.py`)

```python
SEARCH_MODE       = "creep"      # "rotate" = 레거시 제자리 회전 탐색
ODOMETRY_BACKEND  = "encoder"    # → "encoder_imu"
ODOMETRY_SOURCE   = "mega_usb"   # 실기 Mega STATE 사용
MEGA_DRIVE_MAX_RPM = 80.0        # 상위 명령 1.0의 출력축 RPM
WATER_SOURCE      = "gpio"       # → "nano_usb"
VISION_BACKEND    = "hsv"        # → "ccrdnet"
```

부호가 반대로 동작하면 **코드가 아니라** `SIGN_*` 를 뒤집는다
(`SIGN_VISION_ERROR`, `SIGN_IMU_YAW`, `SIGN_LEFT_ENCODER`, …).

---

## 7. Pi 4 예산 / 남은 실측

| 항목 | 값 |
|---|---|
| 제어 루프 | 20 Hz, 비전과 완전 비동기 (블로킹 없음) |
| 추론 제출 | ≤5 fps, 최신 프레임 1장만 (백로그 없음) |
| 결과 수명 | 1.0초 초과 시 무효 → ToF/헤딩 강등 |
| 추론 비용 | 37.6 M MACs, ONNX 163 KB. PC CPU 3스레드 6.7 ms/frame |

**핀맵 확정 후 실측할 것**
1. Mega의 A/B 위상과 `MOTOR*_FORWARD_SIGN` 확인 (`tools/setup.py` 2번)
2. `MEGA_DRIVE_MAX_RPM`, PID 게인, 400 ms watchdog 정지 확인
3. IMU 배선 후 `ODOMETRY_BACKEND="encoder_imu"`, 180° 회전 정밀도 재측정
4. 비전 지연(mean/P95)·CPU·온도 벤치 → 통과 후에만 `VISION_BACKEND="ccrdnet"`
5. `SEARCH_CREEP_MAX_DISTANCE_M` 을 실제 팻말 간격에 맞게 조정

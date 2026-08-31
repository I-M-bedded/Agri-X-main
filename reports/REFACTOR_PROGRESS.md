# 리팩토링 진행 상태 (중단/재개용 메모)

최종 갱신: 2026-08-31. 세션이 끊겨도 이 파일만 보면 이어서 작업할 수 있다.

## 완료 (검증됨)

| 항목 | 파일 | 검증 |
|---|---|---|
| 쿼드러처 A/B 엔코더 (13PPR×4분주) | `sensors/quadrature_encoder.py` | PC 폴백 OK |
| 오도메트리 리팩토링 + IMU 융합 버전 | `sensors/odometry.py` | selftest 144/146 |
| IMU(MPU-6050) 요레이트 모듈 | `sensors/imu.py` | 없을 때 자동 폴백 확인 |
| 바퀴속도 폐루프 구동(PI) | `actuators/closed_loop_drive.py` | drive/stop/cleanup OK |
| Nano USB 수위 링크 | `sensors/nano_link.py` | 포트 없을 때 폴백 확인 |
| config 신규 블록 13-1/2/3 | `config.py` | import OK |
| FSM 팩토리 스위치 3종 | `navigation/mission_state_machine.py` | selftest OK |
| CCRDNet/CRDLD 비전 백엔드 | `sensors/ccrdnet_line_detector.py` | 6.7ms/frame(PC) |

selftest 실패 2건(SIGTERM 핸들러, hardware_safety_fixes 예외)은 **Windows에서만**
나는 기존 아티팩트다. 리팩토링과 무관 (리팩토링 전에도 동일하게 실패).

| 직선(creep) 마커 탐색 + 회전 폴백 | `navigation/mission_state_machine.py` | 전체 임무 sim 통과 |
| 시퀀스/HOME복귀/구동계층 문서 | `farm_robot/CONTROL_PIPELINE.md` | — |
| 고랑 폭 40cm 반영 | `config.py` (200mm) | selftest 144/146 |
| 비전 실측오차 폐루프 실험 | `tools/vision_error_model.py`, `tools/sim_vision_experiment.py` | `reports/sim_closed_loop_report.md` |
| 기존 로깅 버그 2건 수정 | `mission_state_machine.py` | 오차주입으로 발견 |

## 남은 작업

0. **[최우선/코드] `_field_heading` 재기준 검증 추가** — 드리프트가 임계값을
   넘으면 채택 거부. 현재는 무조건 채택이라 비전 오차 하나가 임무 전체를
   망친다 (근거: `reports/sim_closed_loop_report.md` 4절, 중앙 드리프트 40°)
0-1. **[코드] `VISION_MIN_CONFIDENCE` 상향** — 절대값 0.70 이 아니라
   "상위 약 20% 백분위" 기준으로. 현재 0.25 는 완주율 6.7%
1. 핀맵 확정 → `ENCODER_QUAD_PINS` 채우고 A/B 위상·부호 확인
2. `ENCODER_GEAR_RATIO`, `MAX_WHEEL_SPEED_MPS` 실측 → `DRIVE_MODE="closed_loop"`
3. IMU 배선 → `ODOMETRY_BACKEND="encoder_imu"`, 180° 회전 정밀도 재측정
4. Pi 4 비전 지연(mean/P95)·CPU·온도 벤치 → 통과 후 `VISION_BACKEND="ccrdnet"`
5. `SEARCH_CREEP_MAX_DISTANCE_M` 을 실제 팻말 간격에 맞게 조정
6. Nano 펌웨어: `empty`/`ok` 를 **1초 주기로 재전송**하도록 (1회성 이벤트 금지)

## 전환 스위치 (전부 config.py, 기본값은 안전한 레거시 경로)

```python
SEARCH_MODE       = "creep"      # -> "rotate"       (레거시 제자리 회전 탐색)
ODOMETRY_BACKEND  = "encoder"    # -> "encoder_imu"  (IMU 헤딩)
DRIVE_MODE        = "open_loop"  # -> "closed_loop"  (바퀴속도 PI)
WATER_SOURCE      = "gpio"       # -> "nano_usb"     (Nano 시리얼)
VISION_BACKEND    = "hsv"        # -> "ccrdnet"      (CRDLD/CCRDNet 모델)
ENCODER_QUAD_PINS = None         # -> 핀맵 dict      (쿼드러처 활성)
```

전체 설계 설명은 `farm_robot/CONTROL_PIPELINE.md` 참고.

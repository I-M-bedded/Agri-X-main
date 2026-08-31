# Segmentation 중심선과 기존 FSM 연동 검토

## 결론

기존 `MissionStateMachine`의 왕복·물 부족·HOME 복귀 상태 전이는 유지한다. 비전 모델은
고랑 내부의 조향 오차만 제공하며, 행 끝 판정·장애물 안전·귀환 위치 추정을 대신하지
않는다. 공개 데이터 B0가 아직 현장 게이트를 통과하지 못했으므로 기본 설정도
`VISION_BACKEND="hsv"`로 유지한다.

## 마스크에서 모터 명령까지

```text
camera frame
  -> SegFormer-B0 ONNX (최대 5 FPS 비동기 worker)
  -> class-0 thin row-line mask
  -> HoughLinesP + 중복 선 병합
  -> 화면 중앙에 가까운 인접 경계 2개
  -> 두 경계의 중간 중심선
  -> lateral error + heading error
  -> 기존 LineFollower PID
  -> MotorDriver.drive(base_speed, steer)
```

AI 추론은 20Hz FSM 스레드에서 직접 기다리지 않는다. 최신 프레임 하나만 worker에
제출하고 마지막 완료 결과를 사용한다. 결과가 1초보다 오래되거나, 하늘 오염·넓은 blob·
과도한 측면 오차·잘못된 선 개수 조건을 만족하면 confidence를 0으로 만들어 기존
ToF+encoder heading hold로 폴백한다.

## 기존 미션 동작과의 관계

- 고랑 진입: 입구 ArUco로 정렬한 뒤 segmentation 조향을 시작한다.
- 고랑 내부: 비전이 주 조향, 좌우 1D ToF는 중심 교차검증과 폴백이다.
- 고랑 끝: 양쪽 ToF wall loss를 연속 확인한다. 비전 mask만으로 끝이라고 판단하지 않는다.
- 끝 도착: pump zone을 끄고 180도 회전한 뒤 같은 고랑을 역방향으로 따라 입구로 복귀한다.
- 물 부족: 즉시 pump lockout, 최소 안전 거리 이후 유턴, 입구 복귀 후 HOME 목표로 전환한다.
- 정상 완료: 마지막 고랑 왕복을 끝낸 뒤 HOME 목표로 전환한다.

현재 코드의 중요한 전제는 “고랑 안에서 좌우 ToF가 적어도 한 번 벽을 본다”는 것이다.
실제 두둑 높이와 센서 장착 높이가 맞지 않으면 vision이 좋아도 `SAFE_HALT`가 발생한다.
이 조건은 소프트웨어로 완화하기 전에 실제 장착 상태에서 확인해야 한다.

## Raspberry Pi 4B용 HOME 방식

SLAM이나 전체 영상 breadcrumb를 쓰지 않고 **1차원 위상 지도 + 저주기 ArUco 재보정**을
쓴다. 밭의 구조가 “평행한 N개 고랑과 하나의 headland”이므로 이 방식이 계산량과
실패 모드가 가장 작다.

1. 로봇은 현재 고랑 번호와 왕복 완료 여부를 FSM 상태로 보관한다.
2. 물이 고랑 안에서 떨어지면 같은 고랑을 되짚어 입구까지 돌아온다.
3. headland에서 `(현재 고랑 번호 - 1)`칸을 encoder 거리와 heading hold로 이동한다.
4. 이동 중 매 4 tick에만 입구 ArUco를 확인해 현재 번호와 남은 칸 수를 보정한다.
5. HOME marker 0이 유효 거리에서 보이면 최종 정렬한다.
6. 이동 상한, 시간 제한, marker/odometry 불일치가 해소되지 않으면 추측 주행 대신
   `SAFE_HALT`한다.

이 방법은 지도 메모리가 O(고랑 수)가 아니라 사실상 현재 번호 몇 개뿐이고, 영상 특징점
지도·loop closure·LiDAR SLAM이 없어 Pi CPU 부담이 작다. 순수 encoder만 쓰면 미끄럼이
누적되므로 ArUco 재보정은 제거하면 안 된다. 향후 IMU yaw를 heading hold에 더하는 것은
저비용 개선이지만 HOME 방식 자체를 바꿀 필요는 없다.

## 배포 순서

1. `vision_model_experiments/FIELD_FINETUNING.md`의 현장 게이트 통과
2. 승인 B0를 ONNX로 export하고 `farm_robot/models/furrow_line.onnx`에 복사
3. 정지된 바퀴 상태에서 overlay/조향 부호 확인
4. pump를 끈 저속 tether test에서 ToF 폴백과 stale-result 정지 확인
5. 그 다음에만 `VISION_BACKEND="onnx_boundary"` 설정

현재 공개 데이터 checkpoint를 이 경로에 복사하거나 설정을 켜면 안 된다.

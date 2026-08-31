# Gazebo 물리 시뮬 — 좁게 정의된 목적

2D 시뮬(`farm_robot/tools/simulation.py`)이 **답하지 못한 것만** 본다.
밭을 사실적으로 재현하는 것이 목적이 아니다.

## 왜 필요해졌나

2D 시뮬의 마커 모델은 "면각 70° 하드컷" 하나뿐이라, 실제 검출기가 겪는
해상도·원근·포즈 모호성을 전혀 모르고 있었다. 실제로 이 때문에 **결론이 한 번
뒤집혔다**:

> `USE_MARKER_TILT_FOR_FIELD_HEADING`(팻말 각도로 밭 방위 유도)이 2D 시뮬에서는
> **오차 0.00°** 로 검증됐는데, 실제 OpenCV 검출기로 재보니 10cm 마커 2m 에서
> 원근 단서가 **0.00px** 이라 추정 자체가 불가능했다(`tools/aruco_angle_bench.py`).
> 시뮬이 마커 yaw 를 참값 그대로 넘겨준 **순환 검증**이었다.

## 이 시뮬이 답할 질문 (그 외는 목표 아님)

1. **마커가 카메라에 실제로 잡히는가** — 각도/거리/모션블러/해상도 포함
2. 탱크 섀시가 이랑을 밟았을 때 **충돌·전복**이 나는가 (종횡비 2.5 우려)
3. 1D ToF 가 이랑 벽을 **실제로 어떻게 읽는가** (경사면 반사)
4. 위 셋과 비전을 **융합**했을 때 폐루프가 도는가

밭의 생물학적 사실성(작물 종류, 생장 단계)은 목표가 아니다.
**일정 간격으로 반복되는 언덕 + 마커 + 충돌 판정**이면 충분하다.

## 구성

```text
sim_gazebo/
├── scripts/make_world.py     월드 생성 (farm_robot/config.py 측량값을 그대로 읽음)
├── scripts/make_robot.py     AGV 모델 생성 (차체/카메라/ToF 사양)
├── worlds/field.sdf          생성물: 이랑 N+1개 + 팻말
├── models/agv/               생성물: 탱크 섀시 근사 로봇
├── models/markers/           ArUco 텍스처 (생성·자체검증 완료 9/9)
└── models/ground_textures/   우리 데이터셋에서 뽑은 흙 텍스처 (도메인 랜덤화용)
```

**측량값 단일 출처**: 월드는 `farm_robot/config.py` 의
`FIELD_ROW_SPACING_M`, `TOF_NOMINAL_WALL_DISTANCE_MM`, `MARKER_SIZE_M`,
`MARKER_POST_LATERAL_OFFSET_M`, `MARKER_POST_TILT_DEG`, `CAMERA_RESOLUTION`
을 읽어 만든다. 2D 시뮬과 Gazebo 가 서로 다른 밭을 보는 사고를 막는다.

## 생성

```bash
python sim_gazebo/scripts/make_world.py --furrows 4 --length 6
python sim_gazebo/scripts/make_robot.py
```

## 실행 (WSL2 Ubuntu + Docker)

이 PC에는 Docker Desktop + WSL2 Ubuntu + RTX 4070 SUPER 가 있으므로
Gazebo Harmonic 컨테이너로 돌리는 것이 가장 빠르다.

```bash
docker run --rm -it \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD/sim_gazebo:/sim" \
  -e GZ_SIM_RESOURCE_PATH=/sim/models \
  --gpus all \
  gazebo:harmonic \
  gz sim -v4 /sim/worlds/field.sdf
```

> **아직 이 PC에서 실행 검증하지 않았다.** SDF 는 XML 파싱과 구조 검사만
> 통과한 상태다. 첫 실행에서 플러그인 이름/리소스 경로를 손봐야 할 수 있다.

## 왜 Isaac Sim 이 아닌가

Isaac 은 렌더 품질이 좋지만 이 4가지 질문에 답하는 데 필요한 것보다 훨씬
무겁다(설치·에셋·학습 곡선). Gazebo 는 이미 있는 Docker/WSL2 로 당일 돌릴 수
있고, ArUco 검출·충돌·레이센서는 동일하게 얻는다.
렌더 사실성이 병목이라는 증거가 나오면 그때 Isaac 을 검토한다.

## 탱크 섀시에 대한 정직한 한계

무한궤도를 그대로 풀지 않고 **차동구동 + 넓은 접지 + 높은 마찰**로 근사한다.
따라서 **회전 정밀도·미끄러짐 수치를 이 시뮬에서 그대로 가져다 쓰면 안 된다.**
(그 값은 실기 `tools/setup.py` 4번으로 실측한다)
여기서 믿을 것은 **마커 가시성, 충돌/전복, ToF 반사** 세 가지다.

## 전복 판정에 직결되는 실측 필요 값

`scripts/make_robot.py` 상단:
- `MASS_KG = 8.0` — 배터리/물탱크 포함 실제 질량
- `COM_HEIGHT_M = 0.18` — **무게중심 높이. 전복 여부를 지배한다**
- `RIDGE_HEIGHT_M = 0.15` (make_world.py) — 실제 이랑 높이

차체 높이 50cm / 폭 20cm = 종횡비 2.5 라 무게중심이 높으면 쉽게 넘어간다.
이 세 값이 틀리면 전복 결론이 통째로 의미 없다.

## 다음 단계

1. 컨테이너에서 월드가 실제로 뜨는지 확인, 플러그인 경로 수정
2. `/camera` 토픽을 받아 **실제 ArUco 검출률**을 각도/거리별로 측정
   (2D 시뮬의 70° 하드컷을 이 실측 곡선으로 교체)
3. 지면 텍스처를 `models/ground_textures/` 에서 무작위 교체 → 도메인 랜덤화
4. ToF 레이 값을 `farm_robot` 의 `ToFPair` 계약으로 브릿지 → 융합 폐루프 검증

# -*- coding: utf-8 -*-
"""
sim_gazebo/scripts/make_world.py
---------------------------------
Gazebo(gz-sim) 월드 SDF 를 **farm_robot/config.py 측량값 그대로** 생성한다.

목적 (좁게 정의됨)
  밭을 사실적으로 재현하는 것이 아니다. 2D 시뮬이 답하지 못한 것만 본다:
    1) **마커가 카메라에 실제로 잡히는가** (각도/거리/모션블러/해상도)
    2) 탱크 섀시가 이랑을 **밟았을 때 충돌/전복** 이 나는가
    3) 1D ToF 가 이랑 벽을 실제로 어떻게 읽는가
  따라서 지형은 "일정 간격으로 반복되는 언덕(이랑)" 이면 충분하다.

두 시뮬이 어긋나지 않게 config 를 단일 출처로 쓴다:
    FIELD_ROW_SPACING_M, TOF_NOMINAL_WALL_DISTANCE_MM(=고랑 반폭),
    MARKER_SIZE_M, MARKER_POST_LATERAL_OFFSET_M, MARKER_POST_TILT_DEG,
    CAMERA_RESOLUTION

    python sim_gazebo/scripts/make_world.py --furrows 4 --length 6
"""

import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "farm_robot"))

import config as C  # noqa: E402

RIDGE_HEIGHT_M = 0.15        # 이랑 높이 (전복/충돌 판정에 직접 영향) ★실측 필요
RIDGE_TOP_RATIO = 0.5        # 사다리꼴 윗면 폭 / 밑면 폭


def ridge(index, cx, length, half_gap):
    """이랑 하나 = **사다리꼴 프리즘 메시**.

    상자+기울인 판 조합은 경사면과 윗면이 어긋나 틈이 생기고 회전 부호를
    틀리기 쉬웠다(실제로 '사선으로 박힌 기둥'처럼 보이는 버그가 났다).
    단면을 그대로 밀어낸 메시 하나면 그 문제가 사라지고 형상도 현실적이다.
    메시는 scripts/make_ridge_mesh.py 가 생성한다(흙 텍스처 포함).
    """
    return f'''
      <link name="ridge_{index}">
        <pose>{cx} 0 0 0 0 0</pose>
        <collision name="c">
          <geometry><mesh><uri>/sim/models/ridge/ridge.obj</uri></mesh></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>0.9</mu2></ode></friction></surface>
        </collision>
        <visual name="v">
          <geometry><mesh><uri>/sim/models/ridge/ridge.obj</uri></mesh></geometry>
        </visual>
      </link>'''


def marker_post(marker_id, x, y, tilt_deg, size, height=0.45):
    """팻말 = 기둥 + ArUco 판. 카메라 높이 근처에 세운다.

    [주의] PBR metal 워크플로는 metalness 기본값이 1 이라 환경맵이 없으면
      판이 **새까맣게** 렌더된다. metalness 0 / roughness 1 로 낮추고
      ambient/diffuse 폴백을 함께 준다.
    """
    yaw = -math.pi / 2 + math.radians(tilt_deg)   # 2D 시뮬과 같은 규약
    pole_h = height - size / 2.0
    return f'''
    <model name="post_{marker_id}">
      <static>true</static>
      <link name="pole">
        <pose>{x:.3f} {y:.3f} {pole_h/2:.3f} 0 0 0</pose>
        <collision name="c"><geometry><cylinder><radius>0.015</radius>
          <length>{pole_h:.3f}</length></cylinder></geometry></collision>
        <visual name="v"><geometry><cylinder><radius>0.015</radius>
          <length>{pole_h:.3f}</length></cylinder></geometry>
          <material><ambient>0.25 0.25 0.25 1</ambient>
            <diffuse>0.35 0.35 0.35 1</diffuse></material></visual>
      </link>
      <link name="plate">
        <pose>{x:.3f} {y:.3f} {height:.3f} 0 0 {yaw:.4f}</pose>
        <collision name="c"><geometry><box><size>{size:.3f} 0.008 {size:.3f}</size></box></geometry></collision>
        <visual name="v">
          <geometry><box><size>{size:.3f} 0.008 {size:.3f}</size></box></geometry>
          <material>
            <ambient>1 1 1 1</ambient>
            <diffuse>1 1 1 1</diffuse>
            <specular>0 0 0 1</specular>
            <pbr><metal>
              <albedo_map>/sim/models/markers/aruco_{marker_id}.png</albedo_map>
              <metalness>0.0</metalness>
              <roughness>1.0</roughness>
            </metal></pbr>
          </material>
        </visual>
      </link>
    </model>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--furrows", type=int, default=4)
    ap.add_argument("--length", type=float, default=6.0, help="고랑 길이(m)")
    ap.add_argument("--robot-x", type=float, default=-1.0,
                    help="로봇 시작 x (기본: 첫 이랑보다 왼쪽)")
    ap.add_argument("--robot-y", type=float, default=-1.1)
    ap.add_argument("--marker-size", type=float, default=None,
                    help="팻말 한 변(m). 미지정 시 config.MARKER_SIZE_M")
    ap.add_argument("--marker-tilt", type=float, default=None,
                    help="팻말 설치각(도). 미지정 시 config.MARKER_POST_TILT_DEG")
    ap.add_argument("--out", default=os.path.join(_ROOT, "sim_gazebo", "worlds",
                                                  "field.sdf"))
    args = ap.parse_args()

    # 정책 스윕용 오버라이드. config 는 여전히 기본값의 단일 출처다.
    mk_size = C.MARKER_SIZE_M if args.marker_size is None else args.marker_size
    mk_tilt = (C.MARKER_POST_TILT_DEG if args.marker_tilt is None
               else args.marker_tilt)

    half_gap = C.TOF_NOMINAL_WALL_DISTANCE_MM / 1000.0     # 고랑 반폭
    spacing = C.FIELD_ROW_SPACING_M
    off = C.MARKER_POST_LATERAL_OFFSET_M

    # 고랑 k 중심 = (k-1)*spacing.  이랑은 고랑 사이에 놓는다.
    ridges = []
    for i in range(-1, args.furrows + 1):
        cx = (i - 0.5) * spacing
        ridges.append(ridge(i, cx, args.length, half_gap))

    # [중요] HOME 은 1번 팻말과 **겹치지 않게** 옆에 세운다. 정확히 같은 좌표에
    #   두면 카메라 화면에서 두 마커가 겹쳐 그려져 서로의 패턴을 망가뜨린다
    #   (farm_robot/tools/rendered_aruco.py 로 실제 검출기 확인).
    # HOME(0)도 **다른 팻말과 같은 규칙**으로 놓는다: 고랑 k 팻말이
    #   (k-1)*spacing + off 이므로, k=0 자리는 -spacing + off.
    #   (예전에는 -off 로 두어 혼자 다른 규칙이었다)
    # [배치 규약]  [이랑0+마커0] [고랑1] [이랑1+마커1] [고랑2] [이랑2+마커2] ...
    #   - 마커 k 는 **이랑 k 의 중심**:      x = (k - 0.5) * spacing
    #   - 고랑 k 는 이랑 k-1 과 이랑 k 사이:  중심 x = (k - 1) * spacing
    #   -> 마커 0 의 이랑을 **지나쳐야** 첫 진짜 고랑(고랑 1)이 나온다.
    #      마커 0 은 HOME 기준점일 뿐 진입 지점이 아니다.
    posts = []
    for k in range(0, args.furrows + 1):
        mid = C.HOME_MARKER_ID if k == 0 else C.furrow_marker_id(k)
        posts.append(marker_post(mid, (k - 0.5) * spacing, 0.0,
                                 mk_tilt, mk_size))
    # END 는 마지막 고랑의 오른쪽 이랑에 함께 붙인다
    posts.append(marker_post(C.FIELD_END_MARKER_ID,
                             (args.furrows - 0.5) * spacing, 0.0,
                             mk_tilt, mk_size))

    sdf = f'''<?xml version="1.0"?>
<!-- 자동 생성: sim_gazebo/scripts/make_world.py
     측량값 출처 = farm_robot/config.py (두 시뮬이 어긋나지 않게 단일 출처)
       고랑 간격 {spacing}m / 고랑 반폭 {half_gap}m / 이랑 높이 {RIDGE_HEIGHT_M}m
       팻말 {mk_size*100:.0f}cm, 오프셋 {off}m, 기울기 {mk_tilt}도 -->
<sdf version="1.10">
  <world name="field">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>

    <!-- [중요] 조명이 약하면 마커의 '흰 셀'이 어두운 회색으로 렌더돼
         ArUco 가 요구하는 흑백 대비가 나오지 않는다(실제로 검출 0건이었다).
         환경광을 충분히 주고, 팻말 면(-y 방향)을 비추는 보조광을 둔다. -->
    <scene>
      <ambient>0.75 0.75 0.75 1</ambient>
      <background>0.75 0.80 0.88 1</background>
      <shadows>true</shadows>
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>-0.3 0.4 -0.9</direction>
    </light>
    <!-- 팻말 정면(-y)을 비추는 보조광: 마커 대비 확보용 -->
    <light type="directional" name="fill_front">
      <cast_shadows>false</cast_shadows>
      <pose>0 -20 8 0 0 0</pose>
      <diffuse>0.6 0.6 0.6 1</diffuse>
      <direction>0 1 -0.35</direction>
    </light>

    <model name="ground">
      <static>true</static>
      <link name="l">
        <collision name="c"><geometry><plane><normal>0 0 1</normal>
          <size>200 200</size></plane></geometry></collision>
        <visual name="v"><geometry><plane><normal>0 0 1</normal>
          <size>200 200</size></plane></geometry>
          <material><ambient>0.4 0.32 0.24 1</ambient><diffuse>0.5 0.4 0.3 1</diffuse></material>
        </visual>
      </link>
    </model>

    <model name="ridges">
      <static>true</static>{''.join(ridges)}
    </model>
{''.join(posts)}

    <!-- AGV: 첫 이랑보다 왼쪽 헤드랜드에서 밭 안쪽(+y)을 바라보고 출발 -->
    <include>
      <uri>/sim/models/agv</uri>
      <name>agv</name>
      <pose>{args.robot_x} {args.robot_y} 0 0 0 1.5708</pose>
    </include>
  </world>
</sdf>
'''
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(sdf)
    print(f"월드 생성: {args.out}")
    print(f"  고랑 {args.furrows}개, 길이 {args.length}m, 이랑 {args.furrows+1}개")
    print(f"  팻말 {len(posts)}개 (HOME/{args.furrows}고랑/END) "
          f"크기 {mk_size*100:.0f}cm 기울기 {mk_tilt:.0f}도")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
sim_gazebo/scripts/make_robot.py
---------------------------------
AGV(탱크 섀시) 모델 SDF 생성.

형상 (전체 30 x 20 x 50 cm)
    지상고 2cm
    하부 헐(hull)   30 x 20 x 18 cm   - 무게중심이 여기 있다
    좌우 트랙 판     30 x 3 x 8 cm     - 궤도처럼 보이게(구동은 바퀴가 한다)
    상부 마스트      8 x 8 x 30 cm     - 카메라를 45cm 로 올리는 기둥
    카메라           전면, 높이 45cm, 640x480, 화각 62도
    ToF 좌/우        높이 12cm (이랑 높이 15cm 보다 낮아야 벽을 본다)

[이전 버그] base_link 높이를 바퀴 반지름(3.3cm)으로 잡아 50cm 상자의 중심이
  z=5.3cm 에 놓였고, 결과적으로 차체 **절반이 지면 아래**에 묻혀 있었다.
  이제 각 부품 높이를 지면 기준으로 명시한다.

[탱크 근사에 대한 한계]
  무한궤도를 그대로 풀지 않고 차동구동(DiffDrive) + 높은 마찰로 근사한다.
  **회전 미끄러짐 수치를 이 시뮬에서 가져다 쓰면 안 된다.**
  여기서 믿을 것은 마커 가시성, 충돌, ToF 반사다.
"""

import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "farm_robot"))

import config as C  # noqa: E402

# 차체: 길이 30cm x **폭 18cm**(실기 확정값).
#   [주의] 이 보고서의 측정 수치는 전부 **폭 20cm** 로 얻은 것입니다.
#   18cm 는 좌우 여유가 각 1cm 넓어지는 방향이므로 결과는 개선 쪽으로만
#   움직입니다(reports/gazebo_policy_sweep.md 9절에 예상 영향 정리).
BODY_L, BODY_W = 0.30, 0.18
HULL_H = 0.18                 # 하부 헐 높이
MAST_H = 0.30                 # 상부 마스트 높이 (헐 위)
CLEARANCE = 0.02              # 지상고
CAM_HEIGHT_M = 0.45
# ToF 장착 높이: **바퀴 바로 위**. 바퀴 반지름 3.3cm 이므로 바퀴 상단이 6.6cm.
#   그 바로 위(8cm)에 둔다. 이랑 높이 15cm 보다 충분히 낮아야 경사면을 본다.
#   (예전 12cm 는 바퀴에서 너무 떨어져 있었다)
TOF_HEIGHT_M = 0.08
# ToF 센서 타입: "lidar"(CPU 레이캐스트) | "gpu_lidar"(렌더 기반)
#   소프트웨어 렌더링 환경에서는 gpu_lidar 가 값을 내지 않는다.
SENSOR_TYPE = "gpu_lidar"
MASS_KG = 8.0                 # ★ 대략값 (전복 판정이 목적이 아니므로 정밀 불필요)
HFOV_DEG = 62.0

HULL_Z = CLEARANCE + HULL_H / 2.0          # 헐 중심 높이 = 0.11
MAST_Z = CLEARANCE + HULL_H + MAST_H / 2.0  # 마스트 중심 = 0.35


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "sim_gazebo", "models",
                                                  "agv", "model.sdf"))
    args = ap.parse_args()

    w, h = C.CAMERA_RESOLUTION
    hfov = math.radians(HFOV_DEG)
    sep, rad = C.TRACK_WIDTH_M, C.WHEEL_RADIUS_M

    ixx = MASS_KG * (BODY_W**2 + HULL_H**2) / 12.0
    iyy = MASS_KG * (BODY_L**2 + HULL_H**2) / 12.0
    izz = MASS_KG * (BODY_L**2 + BODY_W**2) / 12.0

    def box_part(name, sx, sy, sz, px, py, pz, rgb, collide=True):
        # [수정] 트랙 판은 **시각용**으로만 둔다(충돌 없음).
        #   지면에 닿게 두면 하중을 바퀴와 나눠 받아 바퀴가 헛돌고,
        #   엔코더 오도메트리가 실제 이동의 4~5배를 보고했다
        #   (실측: 명령 -86도 회전 -> 실제 -21도, 전진 1.05m -> 실제 0.22m).
        #   지면에서 띄우면 이번엔 헐 모서리가 박혀 아예 못 간다.
        #   -> 접지는 바퀴 2개 + 무마찰 캐스터 2개로 정리한다.
        col = "" if not collide else f'''
      <collision name="col_{name}">
        <pose>{px:.3f} {py:.3f} {pz:.3f} 0 0 0</pose>
        <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
      </collision>'''
        return col + f'''
      <visual name="vis_{name}">
        <pose>{px:.3f} {py:.3f} {pz:.3f} 0 0 0</pose>
        <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
        <material><ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse></material>
      </visual>'''

    def caster(name, px, r=0.02, lift=0.006):
        """앞뒤 피치를 막는 무마찰 받침. 견인력은 전부 바퀴가 낸다.

        [수정] 캐스터 바닥을 바퀴와 같은 높이(z=0)에 두면 하중이 캐스터로
          몰려 바퀴가 헛돈다(실측: 바퀴 1.06m 회전에 실제 이동 0.35m).
          6mm 띄워 평지에서는 **바퀴만 접지**하게 한다.
        """
        return f'''
      <collision name="col_{name}">
        <pose>{px:.3f} 0 {r + lift - HULL_Z:.4f} 0 0 0</pose>
        <geometry><sphere><radius>{r:.3f}</radius></sphere></geometry>
        <surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction>
          <bounce><restitution_coefficient>0</restitution_coefficient></bounce></surface>
      </collision>'''

    # base_link 원점은 **헐 중심**(지면에서 HULL_Z). 나머지는 그 기준 상대좌표.
    parts = box_part("hull", BODY_L, BODY_W, HULL_H, 0, 0, 0, "0.20 0.35 0.50")
    for sgn, nm in ((1, "l"), (-1, "r")):
        parts += box_part(f"track_{nm}", BODY_L, 0.03, 0.08,
                          0, sgn * (BODY_W / 2 + 0.015), -HULL_H / 2 + 0.02,
                          "0.10 0.10 0.12", collide=False)
    parts += box_part("mast", 0.08, 0.08, MAST_H, 0, 0,
                      MAST_Z - HULL_Z, "0.30 0.30 0.34")
    parts += caster("caster_f", BODY_L / 2 - 0.03)
    parts += caster("caster_r", -(BODY_L / 2 - 0.03))

    def wheel(name, y):
        """바퀴 링크. **링크 자체는 회전시키지 않는다.**

        [수정] 예전에는 링크를 roll +90도로 눕히고 조인트 축을 (0,-1,0) 으로
          줬는데, SDF 에서 축은 기본적으로 **자식 링크 프레임** 기준이라
          눕힌 링크에서 (0,-1,0) 은 결국 **연직축**을 가리켰다.
          그 결과 바퀴는 돌지만 차체는 제자리에서 옆으로 미끄러지기만 했다
          (실측: 오도메트리 3.45m 전진 보고, 실제 y 변화 0.00m).
          링크를 세워두면 축 (0,1,0) 이 그대로 좌우축이라 모호함이 없다.
          충돌은 구, 보이는 것만 원통으로 눕힌다.
        """
        return f'''
    <link name="{name}">
      <pose>0 {y:.4f} {rad:.4f} 0 0 0</pose>
      <inertial><mass>0.4</mass>
        <inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      <collision name="c">
        <geometry><sphere><radius>{rad:.4f}</radius></sphere></geometry>
        <surface><friction><ode><mu>2.0</mu><mu2>2.0</mu2>
          <slip1>0.0</slip1><slip2>0.0</slip2></ode></friction>
          <contact><ode><kp>1e6</kp><kd>100</kd></ode></contact></surface>
      </collision>
      <visual name="v">
        <pose>0 0 0 {math.pi/2:.4f} 0 0</pose>
        <geometry><cylinder><radius>{rad:.4f}</radius><length>0.05</length></cylinder></geometry>
        <material><ambient>0.05 0.05 0.05 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material>
      </visual>
    </link>
    <joint name="{name}_joint" type="revolute">
      <parent>base_link</parent><child>{name}</child>
      <axis><xyz expressed_in="__model__">0 1 0</xyz>
        <limit><lower>-1e16</lower><upper>1e16</upper></limit></axis>
    </joint>'''

    def tof(name, sgn):
        # gpu_lidar 는 렌더 기반이라 소프트웨어 렌더(llvmpipe)에서
        # 헤더만 오고 ranges 가 비었다. CPU 레이캐스트("lidar")로 대체.
        return f'''
      <sensor name="tof_{name}" type="{SENSOR_TYPE}">
        <pose>0 {sgn * (BODY_W/2 + 0.03):.3f} {TOF_HEIGHT_M - HULL_Z:.3f} 0 0 {sgn * math.pi/2:.4f}</pose>
        <update_rate>20</update_rate><topic>tof_{name}</topic>
        <lidar>
          <scan><horizontal><samples>1</samples>
            <min_angle>0</min_angle><max_angle>0</max_angle></horizontal></scan>
          <range><min>0.02</min><max>{C.TOF_OUT_OF_RANGE_MM/1000.0:.2f}</max>
            <resolution>0.001</resolution></range>
        </lidar>
        <always_on>1</always_on>
      </sensor>'''

    sdf = f'''<?xml version="1.0"?>
<!-- 자동 생성: sim_gazebo/scripts/make_robot.py
     전체 {BODY_L*100:.0f}x{BODY_W*100:.0f}x{(CLEARANCE+HULL_H+MAST_H)*100:.0f}cm (폭 18cm 실기값)
     헐 {HULL_H*100:.0f}cm + 마스트 {MAST_H*100:.0f}cm, 카메라 {CAM_HEIGHT_M*100:.0f}cm
     ※ 탱크 궤도는 차동구동 근사. 회전 미끄러짐 수치는 신뢰하지 말 것. -->
<sdf version="1.10">
  <model name="agv">
    <link name="base_link">
      <pose>0 0 {HULL_Z:.4f} 0 0 0</pose>
      <inertial>
        <pose>0 0 -0.02 0 0 0</pose>
        <mass>{MASS_KG}</mass>
        <inertia><ixx>{ixx:.4f}</ixx><iyy>{iyy:.4f}</iyy><izz>{izz:.4f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>{parts}

      <sensor name="front_camera" type="camera">
        <pose>{BODY_L/2:.3f} 0 {CAM_HEIGHT_M - HULL_Z:.3f} 0 0 0</pose>
        <update_rate>20</update_rate><topic>camera</topic>
        <camera>
          <horizontal_fov>{hfov:.5f}</horizontal_fov>
          <image><width>{w}</width><height>{h}</height><format>R8G8B8</format></image>
          <clip><near>0.05</near><far>50</far></clip>
          <distortion><k1>0</k1><k2>0</k2><k3>0</k3><p1>0</p1><p2>0</p2></distortion>
        </camera>
        <always_on>1</always_on>
      </sensor>
{tof("left", 1)}
{tof("right", -1)}
    </link>
{wheel("wheel_left", sep/2)}
{wheel("wheel_right", -sep/2)}

    <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
      <left_joint>wheel_left_joint</left_joint>
      <right_joint>wheel_right_joint</right_joint>
      <wheel_separation>{sep}</wheel_separation>
      <wheel_radius>{rad}</wheel_radius>
      <topic>cmd_vel</topic>
      <odom_topic>odom</odom_topic>
      <frame_id>odom</frame_id><child_frame_id>base_link</child_frame_id>
    </plugin>
  </model>
</sdf>
'''
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(sdf)
    with open(os.path.join(os.path.dirname(args.out), "model.config"), "w",
              encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n<model>\n  <name>agv</name>\n'
                '  <version>1.0</version>\n  <sdf version="1.10">model.sdf</sdf>\n'
                '</model>\n')
    print(f"로봇 모델 생성: {args.out}")
    print(f"  헐 중심 z={HULL_Z:.3f}m (바닥 {CLEARANCE:.2f}m), "
          f"마스트 위 {CLEARANCE+HULL_H+MAST_H:.2f}m, 카메라 {CAM_HEIGHT_M}m")
    print(f"  ToF z={TOF_HEIGHT_M}m (이랑 높이 0.15m 보다 낮음 = 벽을 본다)")


if __name__ == "__main__":
    main()

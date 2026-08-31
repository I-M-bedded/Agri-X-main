# -*- coding: utf-8 -*-
"""
sim_gazebo/scripts/make_ridge_mesh.py
--------------------------------------
이랑(두둑) 단면을 **사다리꼴 프리즘 메시**로 생성한다.

왜 메시인가
  상자 + 기울인 판 조합으로 근사하면 (a) 경사면과 윗면이 어긋나 틈이 생기고
  (b) 회전 축/부호를 틀리기 쉽고 (c) 시각적으로 '판때기'처럼 보인다.
  실제로 그 방식에서 '사선으로 박힌 기둥' 처럼 보이는 버그가 났다.
  단면을 그대로 밀어낸(extrude) 메시 하나면 셋 다 사라진다.

단면 (고랑을 가로지르는 x-z 평면)
      /---- top_w ----\\        z = height
     /                 \\
    /                   \\
   +------  base_w  -----+      z = 0

흙 텍스처는 우리 데이터셋에서 뽑은 것을 쓴다(models/ground_textures/).
"""

import argparse
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.dirname(_HERE)


def write_ridge_obj(path, base_w, top_w, height, length, tex_repeat_y=6.0):
    """사다리꼴 프리즘 OBJ + MTL 생성. y 축(고랑 방향)으로 밀어낸다.

    [중요] **법선(vn)을 반드시 쓴다.** 없으면 DART 충돌 로더가
      "normal count 0 != vertex count" 로 서브메시를 통째로 무시해서
      **이랑에 충돌이 아예 없어진다**(실제로 이 오류를 겪었다).
    """
    hb, ht = base_w / 2.0, top_w / 2.0
    verts = [
        (-hb, 0.0, 0.0), (hb, 0.0, 0.0), (hb, length, 0.0), (-hb, length, 0.0),
        (-ht, 0.0, height), (ht, 0.0, height),
        (ht, length, height), (-ht, length, height),
    ]
    uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, tex_repeat_y), (0.0, tex_repeat_y)]

    dx, dz = hb - ht, height
    n = math.hypot(dz, dx) or 1.0
    normals = [
        (0.0, 0.0, 1.0),            # 1 윗면
        (0.0, 0.0, -1.0),           # 2 바닥
        (dz / n, 0.0, dx / n),      # 3 +x 경사면
        (-dz / n, 0.0, dx / n),     # 4 -x 경사면
        (0.0, -1.0, 0.0),           # 5 y=0 끝면
        (0.0, 1.0, 0.0),            # 6 y=L 끝면
    ]
    faces = [
        ((5, 6, 7, 8), 1),
        ((4, 3, 2, 1), 2),
        ((2, 3, 7, 6), 3),
        ((1, 5, 8, 4), 4),
        ((1, 2, 6, 5), 5),
        ((3, 4, 8, 7), 6),
    ]

    mtl_name = os.path.splitext(os.path.basename(path))[0] + ".mtl"
    lines = [
        f"# 이랑 사다리꼴 프리즘 (base {base_w}m, top {top_w}m, "
        f"h {height}m, len {length}m)",
        f"mtllib {mtl_name}",
    ]
    lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in verts]
    lines += [f"vt {u:.4f} {w:.4f}" for u, w in uvs]
    lines += [f"vn {a:.4f} {b:.4f} {c:.4f}" for a, b, c in normals]
    lines.append("usemtl soil")
    for vs, ni in faces:
        lines.append("f " + " ".join(f"{a}/{i + 1}/{ni}"
                                     for i, a in enumerate(vs)))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    mtl = [
        "newmtl soil",
        "Ka 0.400 0.320 0.240",
        "Kd 0.700 0.580 0.450",
        "Ks 0.000 0.000 0.000",
        "Ns 1",
        "map_Kd ../ground_textures/soil_pilot.png",
    ]
    with open(os.path.join(os.path.dirname(path), mtl_name), "w",
              encoding="utf-8") as f:
        f.write("\n".join(mtl) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=float, default=0.6, help="이랑 밑면 폭(m)")
    ap.add_argument("--top", type=float, default=0.30, help="이랑 윗면 폭(m)")
    ap.add_argument("--height", type=float, default=0.15)
    ap.add_argument("--length", type=float, default=6.0)
    args = ap.parse_args()

    out_dir = os.path.join(_SIM, "models", "ridge")
    os.makedirs(out_dir, exist_ok=True)
    obj = os.path.join(out_dir, "ridge.obj")
    write_ridge_obj(obj, args.base, args.top, args.height, args.length)
    slope_deg = math.degrees(math.atan2(args.height, (args.base - args.top) / 2))
    print(f"이랑 메시 생성: {obj}")
    print(f"  밑면 {args.base}m -> 윗면 {args.top}m, 높이 {args.height}m, "
          f"길이 {args.length}m, 경사 {slope_deg:.0f}도")


if __name__ == "__main__":
    main()

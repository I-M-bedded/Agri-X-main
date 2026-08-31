# -*- coding: utf-8 -*-
"""
tools/sim_design_pilot.py
--------------------------
두 진입 설계를 **하나로 확정하기 위한** 파일럿 실험.

  설계 A (측량 기반)  : 팻말이 고랑 중심선에서 얼마나 떨어졌는지를 로봇이
                        알고 있고, 그 값으로 **고랑 중심을 계산**해 겨냥한다.
                        (config.MARKER_ON_RIDGE_CENTER = True)
  설계 B (측량 무관)  : 팻말이 어디 있든 상관없이 팻말 방위로 접근한 뒤
                        비전/ToF 로 중심을 찾는다. (기존 원칙)

두 설계를 아래 축에서 비교한다.
  - IMU 유무      : 없으면 회전각을 엔코더로만 잰다(미끄러짐에 취약)
  - 궤도 미끄러짐 : 0% / 3%  (IMU 의 값어치는 미끄러질 때만 드러난다)
  - 팻말 각도     : 0 / 30 / 45도
  - 시작 위치     : 헤드랜드 근방 여러 지점 (밭쪽 응시)

무엇을 재는가
  완주율        MISSION_COMPLETE 비율
  이랑밟음      고랑 주행 중 |횡편차| > 여유 인 시행 비율
  유턴+복귀     TURN_AROUND 와 TRAVEL_BACK_TO_ENTRANCE 를 실제로 거친 비율
                (= "고랑 끝나면 유턴해 다시 끝까지 주행"이 되는가)
  HOME 오차     종료 시 HOME(x=0)에서 벗어난 거리

    python tools/sim_design_pilot.py
    python tools/sim_design_pilot.py --quick
"""

import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from tools.sim_vision_experiment import run_trial  # noqa: E402

START_X = (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2)
START_Y = (-0.6, -1.2)


def set_design(survey_based: bool):
    """설계 A/B 를 런타임에 전환한다(모듈이 from-import 하므로 양쪽 모두)."""
    import navigation.mission_state_machine as msm

    config.MARKER_ON_RIDGE_CENTER = survey_based
    msm.MARKER_ON_RIDGE_CENTER = survey_based


def run_cell(survey_based, use_imu, slip, tilt, seeds=(0, 1)):
    set_design(survey_based)
    out = []
    for x in START_X:
        for y in START_Y:
            for sd in seeds:
                out.append(run_trial(
                    seed=sd, n_furrows=3, start_x=x, start_y=y,
                    start_theta_deg=90, marker_tilt_deg=tilt,
                    use_imu=use_imu, slip=slip,
                ))
    n = len(out)
    home = [r["home_error_m"] for r in out]
    return {
        "n": n,
        "complete": sum(r["complete"] for r in out) / n,
        "ridge": sum(1 for r in out if r["ridge_hit_ticks"] > 0) / n,
        "loop": sum(1 for r in out if r["turned_around"] and r["drove_back"]) / n,
        "home_med": st.median(home),
        "home_p90": sorted(home)[int(0.9 * (n - 1))],
        "furrows": sum(r["completed_furrows"] for r in out) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="팻말 각도 30도만")
    args = ap.parse_args()

    tilts = (30,) if args.quick else (0, 30, 45)
    print("=" * 100)
    print("진입 설계 확정 파일럿:  A=측량 기반  vs  B=측량 무관")
    print(f"  셀당 {len(START_X) * len(START_Y) * 2}시행 (시작 x {len(START_X)}종 "
          f"x y {len(START_Y)}종 x 시드 2종), 고랑 3개, 밭쪽 응시 출발")
    print("=" * 100)

    rows = []
    for tilt in tilts:
        print(f"\n[팻말 각도 {tilt}도]")
        print(f"  {'설계':<14s} {'IMU':<5s} {'미끄러짐':<8s} | {'완주':>6s} "
              f"{'이랑밟음':>8s} {'유턴+복귀':>9s} {'HOME중앙':>9s} {'HOME P90':>9s}")
        print("  " + "-" * 88)
        for survey in (True, False):
            for imu in (False, True):
                for slip in (0.0, 0.03):
                    r = run_cell(survey, imu, slip, tilt)
                    rows.append((tilt, survey, imu, slip, r))
                    print(f"  {'A 측량기반' if survey else 'B 측량무관':<14s} "
                          f"{'있음' if imu else '없음':<5s} {slip*100:5.0f}%   | "
                          f"{r['complete']*100:5.1f}% {r['ridge']*100:7.1f}% "
                          f"{r['loop']*100:8.1f}% {r['home_med']*100:8.1f}cm "
                          f"{r['home_p90']*100:8.1f}cm")

    # --- 축별 요약 ---
    print("\n" + "=" * 100)
    print("축별 평균 (모든 팻말 각도/미끄러짐 통합)")
    print("=" * 100)
    for label, key in (("설계", 1), ("IMU", 2), ("미끄러짐", 3)):
        print(f"\n  [{label}]")
        vals = sorted({r[key] for r in rows}, key=str)
        for v in vals:
            sel = [r[4] for r in rows if r[key] == v]
            name = {True: "A 측량기반" if key == 1 else "있음",
                    False: "B 측량무관" if key == 1 else "없음"}.get(v, f"{v}")
            print(f"    {name:<12s} 완주 {st.mean(s['complete'] for s in sel)*100:5.1f}%  "
                  f"이랑밟음 {st.mean(s['ridge'] for s in sel)*100:5.1f}%  "
                  f"유턴+복귀 {st.mean(s['loop'] for s in sel)*100:5.1f}%  "
                  f"HOME중앙 {st.mean(s['home_med'] for s in sel)*100:5.1f}cm")
    return rows


if __name__ == "__main__":
    main()

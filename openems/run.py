# -*- coding: utf-8 -*-
"""
물리 검증 진입점. 진화/강화학습 방향은 ../main.py 에서만 수정한다.

MODE
  s11     : 사각 패치 S11 (openEMS 기준해)
  mesh    : λ/N 메쉬를 여러 단계로 쪼개 수렴 확인 (자동 재메쉬가 아니라 여기 루프)
  compare : 같은 사각 패치를 voxel FDTD 와 겹쳐 S11 오차 확인
  pattern : S11 + NF2FF 지향성 (한 방향 이득이 실제 나오는지)
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 이 폴더에서 고치는 것: 어떤 검증을 돌릴지
# ---------------------------------------------------------------------------
MODE = "s11"  # "s11" | "mesh" | "compare" | "pattern"
MESH_DIVS = (10, 20, 40)  # mesh 모드: 파장 대비 셀 수. 클수록 촘촘
COMPARE_BACKEND = "cpu"  # compare 때 voxel 쪽 "cpu" | "cuda"
# ---------------------------------------------------------------------------


def _save_s11(path: Path, freqs, curves: list[tuple[str, np.ndarray]], title: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, y in curves:
        ax.plot(freqs / 1e9, y, lw=2, label=name)
    ax.axhline(-10.0, color="r", ls="--", lw=1.2, label="-10 dB")
    ax.axvline(7.25, color="gray", ls=":")
    ax.set_ylim([-40, 0])
    ax.set_xlabel("GHz")
    ax.set_ylabel("S11 (dB)")
    ax.grid(True, ls=":")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path)


def mode_s11(mesh_div=20, nf2ff=False, tag="s11"):
    from openems.sim import F0, run_fdtd

    r = run_fdtd(mesh_div=mesh_div, nf2ff=nf2ff, sim_tag=tag)
    _save_s11(
        OUT / f"{tag}.png",
        r["freqs"],
        [("openEMS", r["s11_db"])],
        f"openEMS patch  λ/{mesh_div}  min={r['min_db']:.1f} dB @ {r['min_hz']/1e9:.2f} GHz",
    )
    np.savetxt(
        OUT / f"{tag}.csv",
        np.column_stack([r["freqs"], r["s11_db"]]),
        delimiter=",",
        header="freq_Hz,S11_dB",
        comments="",
    )
    print(f"min S11 = {r['min_db']:.2f} dB @ {r['min_hz']/1e9:.3f} GHz")
    if r["Dmax"] is not None:
        print(f"Dmax = {r['Dmax']:.2f} ({10*np.log10(r['Dmax']+1e-15):.1f} dBi)")
    variation = float(np.max(r["s11_db"]) - np.min(r["s11_db"]))
    if variation < 3.0:
        print("FAIL: S11가 평탄합니다. 가진/타임스텝이 잘렸을 수 있습니다.")
        return 2
    return 0


def mode_mesh():
    """메쉬를 더 잘게 다시 계산. openEMS 자체 AMR 이 아니라 이 루프가 한다."""
    from openems.sim import run_fdtd

    rows = []
    curves = []
    freqs_ref = None
    for div in MESH_DIVS:
        r = run_fdtd(mesh_div=int(div), sim_tag=f"mesh_{div}")
        rows.append((div, r["mesh_res_mm"], r["min_db"], r["min_hz"]))
        curves.append((f"λ/{div}", r["s11_db"]))
        freqs_ref = r["freqs"]
        print(f"  λ/{div}: min={r['min_db']:.2f} dB @ {r['min_hz']/1e9:.3f} GHz")
    _save_s11(OUT / "mesh_study.png", freqs_ref, curves, "openEMS mesh convergence")
    np.savetxt(
        OUT / "mesh_study.csv",
        np.array(rows),
        delimiter=",",
        header="lambda_div,mesh_mm,min_S11_dB,min_Hz",
        comments="",
    )
    if len(rows) >= 2:
        df = abs(rows[-1][3] - rows[-2][3]) / 1e9
        ds = abs(rows[-1][2] - rows[-2][2])
        print(f"가장 촘촘한 두 단계 차이: {df:.3f} GHz, {ds:.2f} dB")
        if df > 0.15 or ds > 3.0:
            print("아직 수렴 전. MESH_DIVS 를 더 올려 다시 돌리세요.")
        else:
            print("메쉬는 이 구간에서 대체로 수렴.")
    return 0


def mode_compare():
    """같은 사각 패치: openEMS (참에 가까움) vs voxel (진화가 쓰는 값)."""
    from openems.sim import PATCH_LX, PATCH_LY, run_fdtd

    sys.path.insert(0, str(ROOT))
    if COMPARE_BACKEND == "cuda":
        from solver_cuda import VoxelFDTD, rectangle_mask
    else:
        from solver_cpu import VoxelFDTD, rectangle_mask

    r = run_fdtd(mesh_div=20, sim_tag="compare_oem")
    metal = rectangle_mask(51, PATCH_LX, PATCH_LY)
    # 1 mm 픽셀에서 x=-2.2 mm → 인덱스 23
    feed = (23, 25)
    metal[feed] = 1
    solver = VoxelFDTD()
    try:
        solver.set_feed(*feed)
        s11_v, mindb, minhz, nused = solver.run(metal, r["freqs"])
    finally:
        solver.close()
    _save_s11(
        OUT / "compare_voxel.png",
        r["freqs"],
        [("openEMS", r["s11_db"]), ("voxel", s11_v)],
        "openEMS vs voxel  (offset-feed rectangle)",
    )
    err = s11_v - r["s11_db"]
    print(
        f"openEMS min {r['min_db']:.2f} dB @ {r['min_hz']/1e9:.3f} GHz\n"
        f"voxel   min {mindb:.2f} dB @ {minhz/1e9:.3f} GHz  steps={nused}\n"
        f"S11 차이 RMS {float(np.sqrt(np.mean(err**2))):.2f} dB  "
        f"골 주파수 차 {(minhz-r['min_hz'])/1e6:.1f} MHz"
    )
    np.savetxt(
        OUT / "compare_voxel.csv",
        np.column_stack([r["freqs"], r["s11_db"], s11_v]),
        delimiter=",",
        header="freq_Hz,openEMS_dB,voxel_dB",
        comments="",
    )
    return 0


def mode_pattern():
    return mode_s11(mesh_div=20, nf2ff=True, tag="pattern")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mode = MODE.lower().strip()
    print(f"openEMS validate  MODE={mode}")
    if mode == "s11":
        return mode_s11()
    if mode == "mesh":
        return mode_mesh()
    if mode == "compare":
        return mode_compare()
    if mode == "pattern":
        return mode_pattern()
    raise ValueError(f"MODE 는 s11 | mesh | compare | pattern : {mode!r}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

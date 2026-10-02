# -*- coding: utf-8 -*-
"""
Physics validation entry point. Evolution/RL settings are edited only in ../main.py.

MODE
  s11     : rectangle patch S11 (openEMS reference)
  mesh    : refine the λ/N mesh over several levels to check convergence (this loop, not automatic remeshing)
  compare : overlay the same rectangle patch with voxel FDTD to check S11 error
  pattern : S11 + NF2FF directivity (whether one-direction gain actually shows up)
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
# What to change in this folder: which validation to run
# ---------------------------------------------------------------------------
MODE = "s11"  # "s11" | "mesh" | "compare" | "pattern"
MESH_DIVS = (10, 20, 40)  # mesh mode: cells per wavelength. Larger = finer
COMPARE_BACKEND = "cpu"  # voxel side for compare: "cpu" | "cuda"
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
        print("FAIL: S11 is flat. Excitation or timesteps may be truncated.")
        return 2
    return 0


def mode_mesh():
    """Recompute with a finer mesh. Done by this loop, not openEMS's own AMR."""
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
        print(f"diff between two finest steps: {df:.3f} GHz, {ds:.2f} dB")
        if df > 0.15 or ds > 3.0:
            print("Not converged yet. Raise MESH_DIVS and rerun.")
        else:
            print("Mesh roughly converged in this range.")
    return 0


def mode_compare():
    """Same rectangle patch: openEMS (close to truth) vs voxel (what evolution uses)."""
    from openems.sim import PATCH_LX, PATCH_LY, run_fdtd

    sys.path.insert(0, str(ROOT))
    if COMPARE_BACKEND == "cuda":
        from solver_cuda import VoxelFDTD, rectangle_mask
    else:
        from solver_cpu import VoxelFDTD, rectangle_mask

    r = run_fdtd(mesh_div=20, sim_tag="compare_oem")
    metal = rectangle_mask(51, PATCH_LX, PATCH_LY)
    # at 1 mm pixels, x=-2.2 mm → index 23
    from core import GATE_FEED_MM, ij_from_mm

    feed = ij_from_mm(*GATE_FEED_MM)
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
        f"S11 diff RMS {float(np.sqrt(np.mean(err**2))):.2f} dB  "
        f"dip freq diff {(minhz-r['min_hz'])/1e6:.1f} MHz"
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
    raise ValueError(f"MODE must be s11 | mesh | compare | pattern : {mode!r}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

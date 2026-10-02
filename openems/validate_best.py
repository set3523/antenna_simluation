# -*- coding: utf-8 -*-
"""
Recompute the GA leader (or a given .npy) design with openEMS and overlay it on the voxel S11.

  python -m openems.validate_best                 # auto-pick the top score from run_byproduct/log.csv
  python -m openems.validate_best path/to/x.npy   # a specific mask

Matched to the same structure as voxel:
  - lower layer = only the 9×11.7 mm rectangle (core.lower_layer_metal) is metal (not full GND)
  - upper layer = 51×51 bitmap
  - feed = voxel has a z-directed port at a grid 'vertex' (lower-left corner of pixel 25).
           openEMS pixels are centered at (i-25) mm, so that vertex is at (-0.5, -0.5) mm.
Output: openems/out/validate_best.png, validate_best.csv
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openems.env import register  # noqa: E402

register()

from CSXCAD import ContinuousStructure  # noqa: E402
from openEMS import openEMS  # noqa: E402
from openEMS.physical_constants import C0, EPS0  # noqa: E402

from openems.sim import (  # noqa: E402
    END_CRITERIA,
    FEED_R,
    NR_TS,
    SIM_BOX,
    SUBSTRATE_CELLS_Z,
    SUBSTRATE_EPS_R,
    SUBSTRATE_TAN_DELTA,
    SUBSTRATE_THICKNESS,
    SUBSTRATE_XY,
    UNIT,
    add_bitmap_patch,
)

F_LO, F_HI = 3.0e9, 9.0e9
FEED_XY_MM = (-0.5, -0.5)


def pick_best_npy() -> Path:
    log = ROOT / "run_byproduct" / "log.csv"
    best, best_sc = None, -1e18
    with open(log, encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            try:
                sc = float(r["score"])
            except (KeyError, ValueError):
                continue
            if sc > best_sc:
                best_sc, best = sc, r["npy"]
    if best is None:
        raise RuntimeError(f"no candidate found in {log}")
    p = ROOT / "run_byproduct" / "metal" / Path(best.replace("\\", "/")).name
    print(f"top score score={best_sc:.3f}  {p.name}")
    return p


def run_openems(upper: np.ndarray, lower: np.ndarray, mesh_div: int = 20, tag: str = "validate_best"):
    f0 = 0.5 * (F_LO + F_HI)
    fc = 0.5 * (F_HI - F_LO)
    mesh_res = C0 / F_HI / UNIT / float(mesh_div)
    kappa = SUBSTRATE_TAN_DELTA * 2 * np.pi * f0 * EPS0 * SUBSTRATE_EPS_R
    sim_path = os.path.join(tempfile.gettempdir(), f"antenna_openems_{tag}")
    os.makedirs(sim_path, exist_ok=True)

    FDTD = openEMS(NrTS=NR_TS, EndCriteria=END_CRITERIA)
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(["MUR"] * 6)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(UNIT)
    mesh.AddLine("x", [-SIM_BOX[0] / 2.0, SIM_BOX[0] / 2.0])
    mesh.AddLine("y", [-SIM_BOX[1] / 2.0, SIM_BOX[1] / 2.0])
    mesh.AddLine("z", [-SIM_BOX[2] / 3.0, SIM_BOX[2] * 2.0 / 3.0])

    top, n_top = add_bitmap_patch(CSX, upper, SUBSTRATE_THICKNESS, SUBSTRATE_THICKNESS)
    FDTD.AddEdges2Grid(dirs="xy", properties=top)
    # lower layer uses the same pixel rule (only the name differs)
    bot = CSX.AddMetal("lower")
    cx = cy = upper.shape[0] // 2
    n_bot = 0
    for i in range(lower.shape[0]):
        for j in range(lower.shape[1]):
            if lower[i, j]:
                x1, y1 = (i - cx) - 0.5, (j - cy) - 0.5
                bot.AddBox([x1, y1, 0.0], [x1 + 1.0, y1 + 1.0, 0.0], priority=10)
                n_bot += 1
    FDTD.AddEdges2Grid(dirs="xy", properties=bot)

    sub = CSX.AddMaterial("FR4", epsilon=SUBSTRATE_EPS_R, kappa=kappa)
    sub.AddBox(
        [-SUBSTRATE_XY / 2.0, -SUBSTRATE_XY / 2.0, 0.0],
        [SUBSTRATE_XY / 2.0, SUBSTRATE_XY / 2.0, SUBSTRATE_THICKNESS],
        priority=0,
    )
    mesh.AddLine("z", np.linspace(0.0, SUBSTRATE_THICKNESS, SUBSTRATE_CELLS_Z + 1))

    fx, fy = FEED_XY_MM
    port = FDTD.AddLumpedPort(1, FEED_R, [fx, fy, 0.0], [fx, fy, SUBSTRATE_THICKNESS], "z", 1.0,
                              priority=5, edges2grid="xy")
    mesh.SmoothMeshLines("all", mesh_res, 1.4)
    print(f"[openEMS] {tag}  top {n_top}px  bottom {n_bot}px  mesh λ/{mesh_div}@9GHz  feed=({fx},{fy}) mm")
    FDTD.Run(sim_path, verbose=1, cleanup=True)
    freqs = np.linspace(F_LO, F_HI, 121)
    port.CalcPort(sim_path, freqs)
    s11 = port.uf_ref / port.uf_inc
    return freqs, 20.0 * np.log10(np.abs(s11) + 1e-15)


def run_voxel(upper: np.ndarray, freqs: np.ndarray, backend: str = "cuda"):
    import main
    from core import FEED_MM, ij_from_mm

    if backend == "cuda":
        from solver_cuda import VoxelFDTD
    else:
        from solver_cpu import VoxelFDTD
    s = VoxelFDTD()
    try:
        s.set_feed(*ij_from_mm(*FEED_MM))
        s.set_sim(float(main.FDTD_T_SEC), float(main.FDTD_CFL), 0.5 * (F_LO + F_HI), 0.5 * (F_HI - F_LO))
        s11, *_ = s.run(upper, freqs)
    finally:
        s.close()
    return s11


def main_cli():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core import lower_layer_metal

    p = Path(sys.argv[1]) if len(sys.argv) > 1 else pick_best_npy()
    upper = np.load(p).astype(np.uint8)
    lower = lower_layer_metal().astype(np.uint8)
    freqs, s_oe = run_openems(upper, lower)
    try:
        s_vx = run_voxel(upper, freqs, "cuda")
    except Exception as e:  # fall back to CPU without CUDA
        print("CUDA voxel failed → CPU:", e)
        s_vx = run_voxel(upper, freqs, "cpu")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT / "validate_best.csv", np.c_[freqs, s_oe, s_vx], delimiter=",",
               header="freq_Hz,openEMS_S11_dB,voxel_S11_dB", comments="")
    band = s_oe < -10.0
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(freqs / 1e9, s_oe, lw=2.4, label="openEMS")
    ax.plot(freqs / 1e9, s_vx, lw=2.0, ls="--", label="voxel (custom FDTD)")
    ax.axhline(-10, color="r", ls=":", lw=1.2)
    ax.set_ylim(-40, 1)
    ax.set_xlabel("GHz")
    ax.set_ylabel("S11 (dB)")
    ax.grid(ls=":")
    ax.legend()
    ax.set_title(f"{p.name}  openEMS <-10dB fraction {band.mean()*100:.0f}%")
    fig.tight_layout()
    fig.savefig(OUT / "validate_best.png", dpi=140)
    diff = np.abs(s_oe - s_vx)
    print(f"saved: {OUT / 'validate_best.png'}")
    print(f"openEMS: min {s_oe.min():.1f} dB @ {freqs[s_oe.argmin()]/1e9:.2f} GHz, fraction below -10dB {band.mean()*100:.0f}%")
    print(f"voxel  : min {s_vx.min():.1f} dB @ {freqs[s_vx.argmin()]/1e9:.2f} GHz")
    print(f"curve diff: mean {diff.mean():.1f} dB, max {diff.max():.1f} dB")


if __name__ == "__main__":
    main_cli()

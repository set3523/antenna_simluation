# -*- coding: utf-8 -*-
"""v4 NTFF (custom FDTD far field) validation: compare the gate reference rectangle patch (full GND) with openEMS NF2FF.

  python -m openems.validate_ntff            # CUDA voxel
  python -m openems.validate_ntff cpu

Compared: per-frequency max directivity Dmax (dBi), +z (broadside) directivity, total efficiency (radiated/incident power).
Resonance differs by ~2.7% between the two solvers, so efficiency may differ near resonance. Directivity within 1 dB passes.
Output: openems/out/validate_ntff.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FREQS = np.array([6.5e9, 7.0e9, 7.5e9])
D_TOL_DB = 1.0


def voxel_ref(backend: str):
    import ctypes

    import ntff
    from core import GATE_FEED_MM, GATE_PATCH_LY_MM, PATCH_LX_MM as PATCH_LX, ij_from_mm  # same 9.0 as sim.PATCH_LX

    if backend == "cuda":
        from solver_cuda import VoxelFDTD, rectangle_mask
    else:
        from solver_cpu import VoxelFDTD, rectangle_mask
    metal = rectangle_mask(51, PATCH_LX, GATE_PATCH_LY_MM)
    feed = ij_from_mm(*GATE_FEED_MM)
    metal[feed] = 1
    s = VoxelFDTD()
    try:
        s.set_lower_full_ground(True)
        s.set_feed(*feed)
        fr = np.ascontiguousarray(FREQS, dtype=np.float64)
        s._dll.voxel_ntff_set(s._ctx, fr.size, fr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        s.run(metal, np.linspace(3e9, 9e9, 61))
        ff = ntff.far_field(s)
    finally:
        s.close()
    return ff


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    import ntff
    from openems.sim import run_fdtd

    ff = voxel_ref(backend)
    r = run_fdtd(mesh_div=20, nf2ff=True, sim_tag="ntff_ref", nf2ff_freqs=FREQS,
                 f_start=5e9, f_stop=9e9, n_freq=161)
    lines = ["freq  | Dmax voxel / openEMS (dBi) | +z voxel / openEMS (dBi) | total eff voxel / openEMS"]
    ok = True
    for q, f in enumerate(FREQS):
        dv = 10 * np.log10(ff["d_peak_f"][q])
        do = 10 * np.log10(r["Dmax_f"][q])
        near = np.argsort(-ntff._DIRS[:, 2])[:5]  # 5 directions closest to +z
        bv = 10 * np.log10(max(float(ff["D"][q][near].mean()), 1e-12))
        bo = 10 * np.log10(max(r.get("D_boresight_f", [np.nan] * 3)[q], 1e-12))
        ev = ff["eff"][q]
        eo = r.get("eff_f", [np.nan] * 3)[q]
        ok &= abs(dv - do) <= D_TOL_DB
        lines.append(f"{f/1e9:.2f} | {dv:6.2f} / {do:6.2f} | {bv:6.2f} / {bo:6.2f} | {ev:5.2f} / {eo:5.2f}")
    lines.append("PASS" if ok else f"directivity diff > {D_TOL_DB} dB, check needed")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validate_ntff.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

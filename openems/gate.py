# -*- coding: utf-8 -*-
"""Called by main before evolution. Refines the openEMS mesh and checks that the reference solution still holds."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Start coarse at λ/10 and stop when two consecutive levels agree. Otherwise refine.
MESH_DIVS = (10, 16, 20, 28, 40)
FREQ_CONV_HZ = 0.15e9
S11_CONV_DB = 3.0
VOXEL_FREQ_TOL_HZ = 0.35e9
# Since complex S11: dip depth must also be within this of openEMS to pass (blocks fake dips that match magnitude only)
VOXEL_DEPTH_TOL_DB = 6.0
# v4: dips deeper than −20 dB are under 1% reflection, so values swing a lot with the grid → clip here before comparing
VOXEL_DEPTH_FLOOR_DB = -20.0


def _save_s11(path: Path, freqs, curves, title: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, y in curves:
        ax.plot(np.asarray(freqs) / 1e9, y, lw=2, label=name)
    ax.axhline(-10.0, color="r", ls="--", lw=1.2)
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


def _resonance_ok(r) -> bool:
    s11 = np.asarray(r["s11_db"])
    return float(np.max(s11) - np.min(s11)) >= 3.0 and float(r["min_db"]) < -5.0


def _voxel_patch(backend: str, freqs: np.ndarray):
    from openems.sim import PATCH_LX

    if backend.lower() == "cuda":
        from solver_cuda import VoxelFDTD, rectangle_mask
    else:
        from solver_cpu import VoxelFDTD, rectangle_mask

    from core import GATE_PATCH_LY_MM, GATE_FEED_MM, ij_from_mm

    metal = rectangle_mask(51, PATCH_LX, GATE_PATCH_LY_MM)
    feed = ij_from_mm(*GATE_FEED_MM)
    metal[feed] = 1
    solver = VoxelFDTD()
    try:
        # openEMS: full GND + a single upper rectangle patch. Separate from the 2-layer (lower rectangle) evolution.
        if hasattr(solver, "set_lower_full_ground"):
            solver.set_lower_full_ground(True)
        solver.set_feed(*feed)
        return solver.run(metal, freqs)
    finally:
        solver.close()


def ensure_before_optimize(backend: str) -> dict:
    """
    Converge the openEMS mesh on the rectangle patch reference, then compare the dip frequency with the evolution voxel.
    On failure, raise so evolution does not start.
    """
    from openems.sim import run_fdtd

    OUT.mkdir(parents=True, exist_ok=True)
    from core import FEED_MM, GATE_FEED_MM

    print("Validation start (openEMS mesh → voxel compare). No evolution until it passes.")
    print(
        f"  gate reference patch feed={GATE_FEED_MM} mm  |  "
        f"evolution voxel feed={FEED_MM} mm (red cell in plot)"
    )

    prev = None
    chosen = None
    curves = []
    freqs_ref = None
    log = []
    for div in MESH_DIVS:
        r = run_fdtd(mesh_div=int(div), sim_tag=f"gate_{div}")
        log.append(r)
        curves.append((f"λ/{div}", r["s11_db"]))
        freqs_ref = r["freqs"]
        print(
            f"  openEMS λ/{div}  min={r['min_db']:.1f} dB @ {r['min_hz']/1e9:.3f} GHz"
        )
        if not _resonance_ok(r):
            print("  → no resonance. Recomputing with a finer mesh.")
            prev = r
            continue
        if prev is not None and _resonance_ok(prev):
            df = abs(r["min_hz"] - prev["min_hz"])
            ds = abs(r["min_db"] - prev["min_db"])
            print(f"  → vs previous step Δf={df/1e6:.0f} MHz, ΔS11={ds:.1f} dB")
            if df <= FREQ_CONV_HZ and ds <= S11_CONV_DB:
                chosen = r
                print(f"  mesh converged: λ/{div}")
                break
        prev = r
        chosen = r

    if chosen is None or not _resonance_ok(chosen):
        raise RuntimeError(
            "No resonance found on openEMS rectangular patch. "
            "Check port, excitation, install path. Evolution aborted."
        )
    if freqs_ref is not None:
        _save_s11(OUT / "gate_mesh.png", freqs_ref, curves, "openEMS mesh (auto)")
        np.savetxt(
            OUT / "gate_mesh.csv",
            np.array([[x["mesh_div"], x["min_db"], x["min_hz"]] for x in log]),
            delimiter=",",
            header="lambda_div,min_S11_dB,min_Hz",
            comments="",
        )

    last_two = log[-2:] if len(log) >= 2 else log
    if len(last_two) == 2:
        df = abs(last_two[1]["min_hz"] - last_two[0]["min_hz"])
        if df > FREQ_CONV_HZ and chosen["mesh_div"] == MESH_DIVS[-1]:
            raise RuntimeError(
                f"Dip frequency still shifts {df/1e6:.0f} MHz even at mesh λ/{MESH_DIVS[-1]}. "
                "Evolution aborted."
            )

    print("Comparing same rectangular patch with voxel...")
    s11_v, mindb, minhz, nused = _voxel_patch(backend, chosen["freqs"])
    _save_s11(
        OUT / "gate_compare.png",
        chosen["freqs"],
        [("openEMS", chosen["s11_db"]), ("voxel", s11_v)],
        "openEMS vs voxel",
    )
    dfv = abs(minhz - chosen["min_hz"])
    print(
        f"  openEMS {chosen['min_db']:.1f} dB @ {chosen['min_hz']/1e9:.3f} GHz\n"
        f"  voxel   {mindb:.1f} dB @ {minhz/1e9:.3f} GHz  steps={nused}\n"
        f"  dip freq diff {dfv/1e6:.0f} MHz"
    )
    if dfv > VOXEL_FREQ_TOL_HZ:
        raise RuntimeError(
            f"voxel resonance differs from openEMS by {dfv/1e6:.0f} MHz. "
            "Aborting, evolution would chase a fake S11."
        )
    ddb = abs(max(mindb, VOXEL_DEPTH_FLOOR_DB) - max(chosen["min_db"], VOXEL_DEPTH_FLOOR_DB))
    print(f"  dip depth diff {ddb:.1f} dB (below {VOXEL_DEPTH_FLOOR_DB:g} dB treated equal, tol {VOXEL_DEPTH_TOL_DB:.1f} dB)")
    if ddb > VOXEL_DEPTH_TOL_DB:
        raise RuntimeError(
            f"voxel dip depth differs from openEMS by {ddb:.1f} dB. "
            "S11 magnitude unreliable, evolution aborted."
        )
    print("Validation passed → starting evolution")
    return {"openems": chosen, "voxel_min_hz": minhz, "voxel_min_db": mindb}

# -*- coding: utf-8 -*-
"""v4 build check: run the same design on CUDA and CPU and confirm S11, total efficiency, cone and directivity match (1~2 min).

  python check_ntff.py                      # v3 GA leader (rectangle patch if missing)
  python check_ntff.py path/to/mask.npy
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run(backend: str, metal: np.ndarray, freqs: np.ndarray):
    import main
    import opt_ga

    cfg = main.make_config()
    if backend == "cuda":
        from solver_cuda import VoxelFDTD
    else:
        from solver_cpu import VoxelFDTD
    s = VoxelFDTD()
    try:
        s.set_feed(*cfg.feed)
        sc, worst, s11, *_rest = opt_ga.evaluate(s, metal, freqs, cfg)
        faces, leave = _rest[5], _rest[6]
        parts = opt_ga.gain_parts(s11, freqs, cfg, metal=metal, faces=faces, leave=leave)
    finally:
        s.close()
    return sc, s11, parts


def main_cli():
    default = ROOT.parent / "v3" / "run_byproduct" / "metal" / "g038_b003_kid017.npy"
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    if p.is_file():
        metal = np.load(p).astype(np.uint8)
        print("mask:", p)
    else:
        from core import lower_layer_metal

        metal = lower_layer_metal().astype(np.uint8)
        print("mask: rectangular patch (no v3 top-1 file)")
    freqs = np.linspace(3e9, 9e9, 81)
    res = {}
    for b in ("cuda", "cpu"):
        sc, s11, parts = run(b, metal, freqs)
        cone_db, eta, eta_db, _beam, _r, _cu, _dap, _c2, face6_db, leave_db, leave_r, _em = parts
        res[b] = (s11, leave_r, cone_db, face6_db)
        print(f"[{b:4s}] score {sc:7.2f}  S11 min {s11.min():6.1f} dB @ {freqs[s11.argmin()]/1e9:.2f} GHz  "
              f"total eff {leave_r:.3f}  cone {cone_db:5.2f} dB  axial dir {face6_db:5.2f} dBi")
    d_s11 = float(np.max(np.abs(res["cuda"][0] - res["cpu"][0])))
    d_eff = abs(res["cuda"][1] - res["cpu"][1])
    d_cone = abs(res["cuda"][2] - res["cpu"][2])
    ok = d_s11 < 1.0 and d_eff < 0.02 and d_cone < 0.3
    print(f"diff: S11 max {d_s11:.2f} dB, total eff {d_eff:.3f}, cone {d_cone:.2f} dB  →  {'match (OK)' if ok else 'MISMATCH, report before running'}")


if __name__ == "__main__":
    main_cli()

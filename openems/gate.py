# -*- coding: utf-8 -*-
"""main 이 진화 전에 호출. openEMS 메쉬를 맞춰 가며 기준해가 살아 있는지 확인한다."""
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

# 성긴 λ/10 부터 시작해, 연속 두 단계가 같으면 멈춤. 아니면 더 촘촘히.
MESH_DIVS = (10, 16, 20, 28, 40)
FREQ_CONV_HZ = 0.15e9
S11_CONV_DB = 3.0
VOXEL_FREQ_TOL_HZ = 0.35e9


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
    from openems.sim import PATCH_LX, PATCH_LY

    if backend.lower() == "cuda":
        from solver_cuda import VoxelFDTD, rectangle_mask
    else:
        from solver_cpu import VoxelFDTD, rectangle_mask

    metal = rectangle_mask(51, PATCH_LX, PATCH_LY)
    feed = (23, 25)
    metal[feed] = 1
    solver = VoxelFDTD()
    try:
        solver.set_feed(*feed)
        return solver.run(metal, freqs)
    finally:
        solver.close()


def ensure_before_optimize(backend: str) -> dict:
    """
    사각 패치 기준해로 openEMS 메쉬를 수렴시킨 뒤, 진화용 voxel 과 골 주파수를 비교.
    실패하면 예외를 던져 진화를 시작하지 않는다.
    """
    from openems.sim import run_fdtd

    OUT.mkdir(parents=True, exist_ok=True)
    print("검증 시작 (openEMS 메쉬 → voxel 비교). 통과 전에 진화하지 않음.")

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
            print("  → 공진이 없음. 메쉬를 더 촘촘히 다시 계산.")
            prev = r
            continue
        if prev is not None and _resonance_ok(prev):
            df = abs(r["min_hz"] - prev["min_hz"])
            ds = abs(r["min_db"] - prev["min_db"])
            print(f"  → 이전 단계와 Δf={df/1e6:.0f} MHz, ΔS11={ds:.1f} dB")
            if df <= FREQ_CONV_HZ and ds <= S11_CONV_DB:
                chosen = r
                print(f"  메쉬 수렴: λ/{div}")
                break
        prev = r
        chosen = r

    if chosen is None or not _resonance_ok(chosen):
        raise RuntimeError(
            "openEMS 사각 패치에서 공진을 못 찾았습니다. "
            "포트/가진/설치 경로를 확인하세요. 진화 중단."
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
                f"메쉬를 λ/{MESH_DIVS[-1]} 까지 잘랐는데도 골 주파수가 {df/1e6:.0f} MHz 흔들립니다. "
                "진화 중단."
            )

    print("voxel 과 같은 사각 패치 비교...")
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
        f"  골 주파수 차 {dfv/1e6:.0f} MHz"
    )
    if dfv > VOXEL_FREQ_TOL_HZ:
        raise RuntimeError(
            f"voxel 공진이 openEMS 와 {dfv/1e6:.0f} MHz 차이. "
            "진화가 가짜 S11 을 쫓게 되어 중단."
        )
    if mindb < chosen["min_db"] - 15.0:
        print(
            "  참고: voxel dip 이 openEMS 보다 훨씬 깊음 (크기만 맞춘 S11). "
            "골 위치는 맞아서 진화는 진행."
        )
    print("검증 통과 → 진화 시작")
    return {"openems": chosen, "voxel_min_hz": minhz, "voxel_min_db": mindb}

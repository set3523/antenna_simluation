# -*- coding: utf-8 -*-
"""openEMS FDTD 한 번 돌리기. 메쉬 밀도·비트맵·NF2FF 는 인자로만."""
from __future__ import annotations

import os
import tempfile

import numpy as np

from openems.env import register

register()

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0

# 기판 / 패치 검증 기하. 진화 설정(main.py)과 분리.
F0 = 7.25e9
FC = 1.25e9
SUBSTRATE_EPS_R = 4.4
SUBSTRATE_TAN_DELTA = 0.02
SUBSTRATE_XY = 50.0
SUBSTRATE_THICKNESS = 1.5
SUBSTRATE_CELLS_Z = 4
PATCH_LX = 9.0
PATCH_LY = 11.7
FEED_X = -2.2
FEED_Y = 0.0
FEED_R = 50.0
SIM_BOX = np.array([100.0, 100.0, 80.0])
NR_TS = 50000
END_CRITERIA = 1e-4
UNIT = 1e-3
METAL_THICKNESS_MM = 0.0  # 0 = 영두께 PEC. 제작 검증 때 0.035
PIX = 51
PITCH = 1.0


def metal_z_spans():
    t = float(METAL_THICKNESS_MM)
    z_gnd = (0.0, 0.0)
    z_patch = (SUBSTRATE_THICKNESS, SUBSTRATE_THICKNESS)
    if t > 0.0:
        z_gnd = (-t, 0.0)
        z_patch = (SUBSTRATE_THICKNESS, SUBSTRATE_THICKNESS + t)
    return z_gnd, z_patch


def pixel_to_mm(px: int, py: int):
    cx = cy = PIX // 2
    return (px - cx) * PITCH, (py - cy) * PITCH


def add_bitmap_patch(CSX, metal: np.ndarray, z0: float, z1: float):
    patch = CSX.AddMetal("patch")
    metal = np.asarray(metal, dtype=np.uint8)
    cx = cy = PIX // 2
    n = 0
    for i in range(PIX):
        for j in range(PIX):
            if not metal[i, j]:
                continue
            x1 = (i - cx) * PITCH - PITCH / 2.0
            x2 = x1 + PITCH
            y1 = (j - cy) * PITCH - PITCH / 2.0
            y2 = y1 + PITCH
            patch.AddBox([x1, y1, z0], [x2, y2, z1], priority=10)
            n += 1
    return patch, n


def run_fdtd(
    mesh_div: int = 20,
    metal=None,
    feed_xy=None,
    nf2ff: bool = False,
    sim_tag: str = "s11",
    f_start: float | None = None,
    f_stop: float | None = None,
    n_freq: int = 401,
):
    """
    mesh_div : 최고주파수 파장 대비 셀 수 (λ/N). 클수록 촘촘.
    metal    : None 이면 검증용 사각 패치, 아니면 (51,51) 비트맵.
    feed_xy  : mm. None 이면 사각 패치는 (-2.2, 0), 비트맵은 픽셀 중앙.
    """
    register()
    mesh_res = C0 / (F0 + FC) / UNIT / float(mesh_div)
    kappa = SUBSTRATE_TAN_DELTA * 2 * np.pi * F0 * EPS0 * SUBSTRATE_EPS_R
    z_gnd, z_patch = metal_z_spans()

    if feed_xy is None:
        if metal is None:
            feed_xy = (FEED_X, FEED_Y)
        else:
            feed_xy = (0.0, 0.0)
    fx, fy = float(feed_xy[0]), float(feed_xy[1])

    sim_path = os.path.join(tempfile.gettempdir(), f"antenna_openems_{sim_tag}")
    os.makedirs(sim_path, exist_ok=True)

    FDTD = openEMS(NrTS=NR_TS, EndCriteria=END_CRITERIA)
    FDTD.SetGaussExcite(F0, FC)
    FDTD.SetBoundaryCond(["MUR", "MUR", "MUR", "MUR", "MUR", "MUR"])

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(UNIT)
    mesh.AddLine("x", [-SIM_BOX[0] / 2.0, SIM_BOX[0] / 2.0])
    mesh.AddLine("y", [-SIM_BOX[1] / 2.0, SIM_BOX[1] / 2.0])
    mesh.AddLine("z", [-SIM_BOX[2] / 3.0, SIM_BOX[2] * 2.0 / 3.0])

    if metal is None:
        patch = CSX.AddMetal("patch")
        patch.AddBox(
            [-PATCH_LX / 2.0, -PATCH_LY / 2.0, z_patch[0]],
            [PATCH_LX / 2.0, PATCH_LY / 2.0, z_patch[1]],
            priority=10,
        )
        n_copper = None
    else:
        patch, n_copper = add_bitmap_patch(CSX, metal, z_patch[0], z_patch[1])
    FDTD.AddEdges2Grid(dirs="xy", properties=patch, metal_edge_res=mesh_res / 2.0)

    substrate = CSX.AddMaterial("FR4", epsilon=SUBSTRATE_EPS_R, kappa=kappa)
    substrate.AddBox(
        [-SUBSTRATE_XY / 2.0, -SUBSTRATE_XY / 2.0, 0.0],
        [SUBSTRATE_XY / 2.0, SUBSTRATE_XY / 2.0, SUBSTRATE_THICKNESS],
        priority=0,
    )
    mesh.AddLine("z", np.linspace(0.0, SUBSTRATE_THICKNESS, SUBSTRATE_CELLS_Z + 1))

    gnd = CSX.AddMetal("gnd")
    gnd.AddBox(
        [-SUBSTRATE_XY / 2.0, -SUBSTRATE_XY / 2.0, z_gnd[0]],
        [SUBSTRATE_XY / 2.0, SUBSTRATE_XY / 2.0, z_gnd[1]],
        priority=10,
    )
    FDTD.AddEdges2Grid(dirs="xy", properties=gnd)
    if METAL_THICKNESS_MM > 0.0:
        mesh.AddLine("z", [z_gnd[0], z_patch[1]])

    port = FDTD.AddLumpedPort(
        1,
        FEED_R,
        [fx, fy, 0.0],
        [fx, fy, SUBSTRATE_THICKNESS],
        "z",
        1.0,
        priority=5,
        edges2grid="xy",
    )

    calc_nf2ff = None
    if nf2ff:
        calc_nf2ff = FDTD.CreateNF2FFBox(
            start=[-35.0, -35.0, -5.0],
            stop=[35.0, 35.0, 15.0],
        )

    mesh.SmoothMeshLines("all", mesh_res, 1.4)
    print(f"[openEMS] {sim_tag}  mesh λ/{mesh_div} ({mesh_res:.2f} mm)  feed=({fx:.2f},{fy:.2f})")
    FDTD.Run(sim_path, verbose=1, cleanup=True)

    if f_start is None:
        f_start = max(1e9, F0 - FC)
    if f_stop is None:
        f_stop = F0 + FC
    freqs = np.linspace(f_start, f_stop, n_freq)
    port.CalcPort(sim_path, freqs)
    s11 = port.uf_ref / port.uf_inc
    s11_db = 20.0 * np.log10(np.abs(s11) + 1e-15)
    min_idx = int(np.argmin(s11_db))

    out = {
        "freqs": freqs,
        "s11_db": s11_db,
        "min_db": float(s11_db[min_idx]),
        "min_hz": float(freqs[min_idx]),
        "mesh_div": mesh_div,
        "mesh_res_mm": mesh_res,
        "n_copper": n_copper,
        "sim_path": sim_path,
        "Dmax": None,
        "D_boresight": None,
    }
    if calc_nf2ff is not None:
        theta = np.arange(0.0, 181.0, 5.0)
        phi = np.arange(0.0, 360.0, 10.0)
        nf = calc_nf2ff.CalcNF2FF(sim_path, np.array([F0]), theta, phi)
        dmax = np.array(nf.Dmax, dtype=float).reshape(-1)
        out["Dmax"] = float(np.max(dmax))
        print(f"[openEMS] NF2FF  Dmax={out['Dmax']:.2f}  (linear)")
    return out

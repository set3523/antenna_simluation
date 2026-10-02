# -*- coding: utf-8 -*-
"""Masks, connectivity, seeds. The minimal unit shared by learning, circle and GA."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solver_cpu import rectangle_mask

PIX = 51
CENTER = PIX // 2
# mm (board center=0). Port for evolution and voxel FDTD, red cell in xy plots.
FEED_MM = (0.0, 0.0)
# openEMS gate only (rectangle patch mesh check) — resonance and match confirmed only at (-2.2,0).
GATE_FEED_MM = (-2.2, 0.0)
GATE_PATCH_LY_MM = 11.7
PATCH_LX_MM = 9.0
PATCH_LY_MM = 11.7
GND_MARK_MM = (0.0, 0.0)


def ij_from_mm(x_mm: float, y_mm: float) -> tuple[int, int]:
    return CENTER + int(round(float(x_mm))), CENTER + int(round(float(y_mm)))


def mm_from_ij(i: int, j: int) -> tuple[float, float]:
    return float(i - CENTER), float(j - CENTER)

DIFFUSE_K = np.array(
    [[0.05, 0.10, 0.05], [0.10, 0.40, 0.10], [0.05, 0.10, 0.05]], dtype=np.float64
)


@dataclass
class Individual:
    metal: np.ndarray  # upper layer 51×51 (chart/index map). Lower rectangle is fixed in FDTD
    mode: int  # -1: no symmetry enforced
    score: float | None = None
    s11: np.ndarray | None = None
    air: np.ndarray | None = None
    cut: np.ndarray | None = None
    worst: float | None = None
    d_use: float | None = None
    eta: float | None = None  # band-average 1-|S11|²
    eta_min: float | None = None  # η at the worst frequency in band (plot/debug)
    d_ap: float | None = None
    d_peak: float | None = None
    beam_db: float | None = None
    leave: float | None = None
    leave_db: float | None = None
    faces: np.ndarray | None = None
    field_extra: dict | None = None
    src: str = "ga"
    chart_uv: tuple | None = None  # chart plane (r cos θ, r sin θ)
    chart_s: int | None = None  # spiral start order
    chart_theta: float | None = None
    chart_index: int | None = None  # mask ID, dedupe only
    chart_kind: str = "seed"
    chart_r: float | None = None
    chart_root: int = 0
    feasible: bool | None = None
    eval_ck: bytes | None = None  # metal+mode fingerprint at FDTD time
    port_ij: tuple[int, int] | None = None  # feed_search: voxel port (i,j). metal is the fixed patch


def is_feasible(ind: Individual, cfg=None) -> bool:
    """Only FDTD-evaluated individuals can be chart, peaks, Best or parent candidates. No hard cut on copper/S11."""
    _ = cfg
    return ind.s11 is not None and getattr(ind, "worst", None) is not None


def clone_ind(ind: Individual) -> Individual:
    return Individual(
        metal=ind.metal.copy(),
        mode=ind.mode,
        score=ind.score,
        s11=None if ind.s11 is None else np.array(ind.s11, copy=True),
        air=None if ind.air is None else np.array(ind.air, copy=True),
        cut=None if ind.cut is None else np.array(ind.cut, copy=True),
        worst=ind.worst,
        d_use=ind.d_use,
        eta=ind.eta,
        eta_min=getattr(ind, "eta_min", None),
        d_ap=getattr(ind, "d_ap", None),
        d_peak=getattr(ind, "d_peak", None),
        beam_db=getattr(ind, "beam_db", None),
        leave=getattr(ind, "leave", None),
        leave_db=getattr(ind, "leave_db", None),
        faces=None if getattr(ind, "faces", None) is None else np.array(ind.faces, copy=True),
        field_extra=None
        if getattr(ind, "field_extra", None) is None
        else {k: np.array(v, copy=True) for k, v in ind.field_extra.items()},
        src=ind.src,
        chart_uv=ind.chart_uv,
        chart_s=getattr(ind, "chart_s", None),
        chart_theta=getattr(ind, "chart_theta", None),
        chart_r=getattr(ind, "chart_r", None),
        chart_index=getattr(ind, "chart_index", None),
        chart_kind=getattr(ind, "chart_kind", "seed"),
        chart_root=int(getattr(ind, "chart_root", 0) or 0),
        feasible=getattr(ind, "feasible", None),
        eval_ck=getattr(ind, "eval_ck", None),
        port_ij=getattr(ind, "port_ij", None),
    )


def effective_port(ind: Individual, cfg_feed: tuple[int, int]) -> tuple[int, int]:
    p = getattr(ind, "port_ij", None)
    return p if p is not None else cfg_feed


def fixed_patch_metal(lx_mm: float = PATCH_LX_MM, ly_mm: float = PATCH_LY_MM) -> np.ndarray:
    """Same lower rectangle as gate (9×11.7 mm). Same as solver metal_lo, for plots and feed scan."""
    return rectangle_mask(PIX, float(lx_mm), float(ly_mm))


def lower_layer_metal(lx_mm: float = PATCH_LX_MM, ly_mm: float = PATCH_LY_MM) -> np.ndarray:
    return fixed_patch_metal(lx_mm, ly_mm)


def patch_port_sites(metal: np.ndarray | None = None) -> list[tuple[int, int]]:
    g = fixed_patch_metal() if metal is None else np.asarray(metal, dtype=np.uint8)
    ys, xs = np.nonzero(g)
    return [(int(i), int(j)) for i, j in zip(ys, xs, strict=True)]


def seed_fixed_patch_port(i: int, j: int, lx_mm: float = PATCH_LX_MM, ly_mm: float = PATCH_LY_MM) -> Individual:
    g = fixed_patch_metal(lx_mm, ly_mm)
    i, j = int(i), int(j)
    if g[i, j] == 0:
        raise ValueError(f"port ({i},{j}) is outside the patch")
    return Individual(g, mode=-1, src="feed", port_ij=(i, j))


def port_manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def apply_symmetry(core_src: np.ndarray, mode: int) -> np.ndarray:
    ind = np.zeros((PIX, PIX), dtype=np.uint8)
    if mode < 0:
        ind[:] = core_src
        return ind
    half = CENTER + 1
    if mode == 1:
        for i in range(half):
            for j in range(half):
                v = int(core_src[CENTER + i, CENTER + j])
                ind[CENTER + i, CENTER + j] = v
                ind[CENTER - i, CENTER + j] = v
                ind[CENTER + i, CENTER - j] = v
                ind[CENTER - i, CENTER - j] = v
    elif mode == 4:
        for i in range(PIX):
            for j in range(half):
                v = int(core_src[i, CENTER + j])
                ind[i, CENTER + j] = v
                ind[i, CENTER - j] = v
    elif mode == 3:
        for i in range(half):
            for j in range(PIX):
                v = int(core_src[CENTER + i, j])
                ind[CENTER + i, j] = v
                ind[CENTER - i, j] = v
    else:
        return apply_symmetry(core_src, 1)
    return ind


def keep_connected(metal: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    out = np.zeros_like(metal)
    if metal[seed] == 0:
        metal = metal.copy()
        metal[seed] = 1
    stack = [seed]
    seen = np.zeros_like(metal, dtype=bool)
    seen[seed] = True
    out[seed] = 1
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < PIX and 0 <= nj < PIX and not seen[ni, nj] and metal[ni, nj]:
                seen[ni, nj] = True
                out[ni, nj] = 1
                stack.append((ni, nj))
    return out


def repair_ga(ind: Individual, feed: tuple[int, int]) -> Individual:
    """GA: feed and symmetry, no connectivity repair, cut diagonal contacts."""
    from circle.diagonal import enforce_feed_symmetry, strip_diagonal_contacts

    mode = int(ind.mode)
    metal = enforce_feed_symmetry(ind.metal.astype(np.uint8), feed, mode)
    metal, n_cut = strip_diagonal_contacts(metal, feed)
    metal[int(feed[0]), int(feed[1])] = 1
    out = Individual(metal=metal, mode=ind.mode, src=ind.src)
    out.ga_diag_cut = int(n_cut)  # type: ignore[attr-defined]
    out.port_ij = getattr(ind, "port_ij", None)
    out.chart_uv = getattr(ind, "chart_uv", None)
    out.chart_s = getattr(ind, "chart_s", None)
    out.chart_theta = getattr(ind, "chart_theta", None)
    out.chart_index = getattr(ind, "chart_index", None)
    out.chart_kind = getattr(ind, "chart_kind", "seed")
    out.chart_root = int(getattr(ind, "chart_root", 0) or 0)
    out.chart_r = getattr(ind, "chart_r", None)
    return out


def repair(ind: Individual, feed: tuple[int, int]) -> Individual:
    return repair_ga(ind, feed)


def repair_upper(ind: Individual, feed: tuple[int, int], *, connect: bool) -> Individual:
    """GA repair. If connect=True (main.CONNECT_REPAIR), remove upper copper not 4-connected to the feed."""
    out = repair_ga(ind, feed)
    if connect:
        n_cut = getattr(out, "ga_diag_cut", 0)
        out.metal = keep_connected(out.metal.astype(np.uint8), (int(feed[0]), int(feed[1]))).astype(np.uint8)
        out.ga_diag_cut = n_cut  # type: ignore[attr-defined]
    return out


def seed_rectangle(feed: tuple[int, int]) -> Individual:
    """openEMS/gate and rectangle anchor only. Not used for warmup Best (upper layer = map search)."""
    g = rectangle_mask(PIX, PATCH_LX_MM, PATCH_LY_MM)
    g[feed] = 1
    return repair(Individual(g, mode=-1), feed)


def seed_random(rng: np.random.Generator, feed: tuple[int, int], mode: int) -> Individual:
    if rng.random() < 0.5:
        p = float(rng.uniform(0.06, 0.42))
        g = (rng.random((PIX, PIX)) < p).astype(np.uint8)
    else:
        g = np.zeros((PIX, PIX), dtype=np.uint8)
        g[feed] = 1
        n_add = int(rng.integers(50, 800))
        for _ in range(n_add):
            ys, xs = np.nonzero(g)
            k = int(rng.integers(0, len(xs)))
            i, j = int(ys[k]), int(xs[k])
            di = int(rng.integers(-2, 3))
            dj = int(rng.integers(-2, 3))
            ni = int(np.clip(i + di, 0, PIX - 1))
            nj = int(np.clip(j + dj, 0, PIX - 1))
            g[ni, nj] = 1
        if rng.random() < 0.35:
            g[rng.random((PIX, PIX)) < 0.1] = 0
    g[feed] = 1
    nmin = 80
    while int(g.sum()) < nmin:
        ys, xs = np.nonzero(g)
        k = int(rng.integers(0, len(xs)))
        i, j = int(ys[k]), int(xs[k])
        di = int(rng.integers(-2, 3))
        dj = int(rng.integers(-2, 3))
        g[int(np.clip(i + di, 0, PIX - 1)), int(np.clip(j + dj, 0, PIX - 1))] = 1
    return repair(Individual(g, mode), feed)


def diffuse_grow(
    metal: np.ndarray,
    rng: np.random.Generator,
    rate: float,
    steps: int,
    kernel: np.ndarray | None = None,
) -> np.ndarray:
    k = DIFFUSE_K if kernel is None else np.asarray(kernel, dtype=np.float64)
    g = metal.astype(np.float64)
    for _ in range(max(0, steps)):
        pad = np.pad(g, 1, mode="constant")
        p = np.zeros_like(g)
        for di in range(3):
            for dj in range(3):
                p += k[di, dj] * pad[di : di + PIX, dj : dj + PIX]
        grow = (rng.random(g.shape) < rate * p) & (g < 0.5)
        g = np.clip(g + grow.astype(np.float64), 0.0, 1.0)
    out = (g >= 0.5).astype(np.uint8)
    out |= metal.astype(np.uint8)
    return out


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(np.asarray(a, dtype=np.uint8) != np.asarray(b, dtype=np.uint8)))

# -*- coding: utf-8 -*-
"""Mask ↔ index ID (for storage and dedupe). Chart coords live in circle.spiral_chart."""
from __future__ import annotations

import math

import numpy as np

from core import Individual, PIX

N_BITS = PIX * PIX

# --- mask ↔ index (dedupe only, not chart coords) ---


def mask_to_index(metal: np.ndarray) -> int:
    flat = np.asarray(metal, dtype=np.uint8).ravel()
    n = 0
    for k in range(N_BITS):
        if flat[k]:
            n |= 1 << k
    return int(n)


def index_to_mask(index: int) -> np.ndarray:
    index = int(index) & ((1 << N_BITS) - 1)
    flat = np.zeros(N_BITS, dtype=np.uint8)
    for k in range(N_BITS):
        if (index >> k) & 1:
            flat[k] = 1
    return flat.reshape(PIX, PIX)


def sync_chart_coords(ind: Individual) -> Individual:
    ind.chart_index = mask_to_index(ind.metal)
    return ind


def mask_to_uv(metal: np.ndarray) -> tuple[int, int]:
    """Legacy index split u + v*2^1301. Not chart coords, for Reveal and diagnostics only."""
    idx = mask_to_index(metal)
    width = 1 << 1301
    return int(idx % width), int(idx // width)


# --- spiral chart (preferred) ---

from circle.spiral_chart import (  # noqa: E402
    SpiralChart,
    attach_spiral_sample,
    chart_neighbor_rs8,
    seed_chart_metal,
    verify_mask_matches_rs,
)


def attach_sample_uv(ind: Individual, u: float, v: float) -> Individual:
    """Records planar (r cos θ, r sin θ); legacy name."""
    ind.chart_uv = (float(u), float(v))
    ind.chart_index = mask_to_index(ind.metal)
    return ind


def uv_local(u: float, v: float, anchor: tuple) -> tuple[float, float]:
    au, av = float(anchor[0]), float(anchor[1])
    return float(u) - au, float(v) - av


def uv_global(lu: float, lv: float, anchor: tuple) -> tuple[float, float]:
    au, av = float(anchor[0]), float(anchor[1])
    return au + float(lu), av + float(lv)


def chart_dist_plane(u: float, v: float) -> float:
    return float(math.hypot(float(u), float(v)))


def chart_dist(uv, anchor) -> float:
    """Legacy: ignores anchor, planar radius."""
    _ = anchor
    if uv is None:
        return float("inf")
    return chart_dist_plane(uv[0], uv[1])

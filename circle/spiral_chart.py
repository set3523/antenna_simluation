# -*- coding: utf-8 -*-
"""
Feed-connected chart (v3): every (r, θ) mask is 4-connected to the feed.

  r = independent half-plane pixels flipped growing out from the feed (actual cells incl. mirror ≈ 2r)
  θ = growth direction. Used as half-plane direction φ = θ/2 ∈ [0, π).

Order (order_s): the cell right above the feed (fi, fj+1) is always index 0,
the rest sorted by ascending key = cheb * (1 + LAMBDA*|angle − φ|).
key increases monotonically along each ray, so the first r cells form a feed-centered star-shaped blob.
mask_from_rs is deterministic: (seed XOR range) → diagonal-contact trim → keep only feed-connected.
So chart coords → mask is 1:1 (deterministic), and the shape fed to FDTD is always attached to the feed.
The existing SpiralChart API (M, s_limit, s_from_theta, mask_from_rs, neighbors_rs …) is unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from core import CENTER, PIX, Individual, apply_symmetry, hamming, keep_connected

from circle.index_map import mask_to_index

from circle.diagonal import has_diagonal_contact, strip_diagonal_contacts

# Direction preference strength. Larger gives thin long branches, smaller is closer to a disk.
LAMBDA = 1.5


def seed_chart_metal(feed: tuple[int, int], *, symmetry_mode: int = 4) -> np.ndarray:
    """Chart seed: feed + left-right symmetry (default mode 4)."""
    g = np.zeros((PIX, PIX), dtype=np.uint8)
    g[int(feed[0]), int(feed[1])] = 1
    if symmetry_mode >= 0:
        g = apply_symmetry(g, symmetry_mode)
        g[int(feed[0]), int(feed[1])] = 1
    return g


def _half_plane_pixels(feed: tuple[int, int]) -> list[tuple[int, float, int, int]]:
    fi, fj = int(feed[0]), int(feed[1])
    rows = []
    for i in range(PIX):
        for j in range(PIX):
            if j <= fj or (i, j) == (fi, fj):
                continue
            cheb = int(max(abs(i - fi), abs(j - fj)))
            ang = float(math.atan2(j - fj, i - fi))  # (0, π)
            rows.append((cheb, ang, i, j))
    return rows


def build_independent_order(feed: tuple[int, int]) -> list[tuple[int, int]]:
    """Legacy: Chebyshev ↑, angle ↑ (φ ignored). For M computation and compatibility."""
    rows = sorted(_half_plane_pixels(feed), key=lambda t: (t[0], t[1]))
    return [(t[2], t[3]) for t in rows]


def build_direction_order(feed: tuple[int, int], phi: float) -> list[tuple[int, int]]:
    fi, fj = int(feed[0]), int(feed[1])
    anchor = (fi, fj + 1)
    rows = []
    for cheb, ang, i, j in _half_plane_pixels(feed):
        if (i, j) == anchor:
            continue
        dang = abs(ang - phi)
        key = cheb * (1.0 + LAMBDA * dang)
        rows.append((key, cheb, dang, i, j))
    rows.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4]))
    return [anchor] + [(t[3], t[4]) for t in rows]


def mirror_j(j: int, fj: int) -> int:
    return int(2 * fj - j)


def toggle_pixel(metal: np.ndarray, i: int, j: int, fj: int) -> None:
    metal[i, j] = 1 - int(metal[i, j])
    mj = mirror_j(j, fj)
    if mj != j and 0 <= mj < PIX:
        metal[i, mj] = 1 - int(metal[i, mj])


@dataclass
class SpiralChart:
    seed_base: np.ndarray
    feed: tuple[int, int]
    r_max: int
    order: list[tuple[int, int]]
    _orders: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_seed(cls, seed_metal: np.ndarray, feed: tuple[int, int], r_max: int = 130) -> SpiralChart:
        base = np.asarray(seed_metal, dtype=np.uint8).copy()
        order = build_independent_order(feed)
        return cls(seed_base=base, feed=(int(feed[0]), int(feed[1])), r_max=int(r_max), order=order)

    @property
    def M(self) -> int:
        return len(self.order)

    @property
    def s_limit(self) -> int:
        return max(0, self.M - int(self.r_max))

    def s_from_theta(self, theta: float) -> int:
        if self.s_limit <= 0:
            return 0
        th = float(theta) % (2.0 * math.pi)
        s = int(round(th / (2.0 * math.pi) * self.s_limit))
        return int(np.clip(s, 0, self.s_limit))

    def theta_from_s(self, s: int) -> float:
        if self.s_limit <= 0:
            return 0.0
        return (float(s) / float(self.s_limit)) * 2.0 * math.pi

    def phi_from_s(self, s: int) -> float:
        """Growth direction (half-plane angle, 0 to π)."""
        return 0.5 * self.theta_from_s(s)

    def order_for_s(self, s: int) -> list[tuple[int, int]]:
        s = int(s)
        o = self._orders.get(s)
        if o is None:
            o = build_direction_order(self.feed, self.phi_from_s(s))
            self._orders[s] = o
        return o

    def plane_xy(self, r: float, theta: float) -> tuple[float, float]:
        return float(r * math.cos(theta)), float(r * math.sin(theta))

    def mask_from_rs(self, r: int, s: int) -> np.ndarray:
        r = int(r)
        s = int(s)
        if r < 0 or r > self.r_max:
            raise ValueError(f"r out of range: {r}")
        if s < 0 or s > self.s_limit:
            raise ValueError(f"s out of range: s={s} s_limit={self.s_limit}")
        g = self.seed_base.copy()
        fi, fj = self.feed
        if r > 0:
            order = self.order_for_s(s)
            for k in range(min(r, len(order))):
                i, j = order[k]
                toggle_pixel(g, i, j, fj)
        g[fi, fj] = 1
        g, _ = strip_diagonal_contacts(g, self.feed)
        g = keep_connected(g, self.feed).astype(np.uint8)
        return g

    def individual_from_polar(
        self,
        r: float,
        theta: float,
        *,
        mode: int = 4,
        src: str = "chart",
        reject_diagonal: bool = True,
    ) -> Individual | None:
        r_int = int(round(float(r)))
        s = self.s_from_theta(theta)
        try:
            metal = self.mask_from_rs(r_int, s)
        except ValueError:
            return None
        if reject_diagonal and has_diagonal_contact(metal):
            return None
        ind = Individual(metal, mode=mode, src=src)
        attach_spiral_sample(ind, r_int, s, theta, self)
        return ind

    def neighbors_rs(self, r: int, s: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for dr in (-1, 0, 1):
            for ds in (-1, 0, 1):
                if dr == 0 and ds == 0:
                    continue
                nr, ns = int(r) + dr, int(s) + ds
                if nr < 0 or nr > self.r_max:
                    continue
                if ns < 0 or ns > self.s_limit:
                    continue
                out.append((nr, ns))
        return out

    def independent_hamming(self, r: int) -> int:
        return int(r)

    def total_hamming_from_seed(self, r: int) -> int:
        return int(2 * r)


def attach_spiral_sample(
    ind: Individual,
    r: int,
    s: int,
    theta: float,
    spiral: SpiralChart,
) -> Individual:
    ux, vy = spiral.plane_xy(float(r), float(theta))
    ind.chart_r = float(r)
    ind.chart_s = int(s)
    ind.chart_theta = float(theta)
    ind.chart_uv = (ux, vy)
    ind.chart_index = mask_to_index(ind.metal)
    return ind


def chart_neighbor_rs8(r: int, s: int, spiral: SpiralChart) -> list[tuple[int, int, float]]:
    """8 neighbors of (r,s) + angle θ."""
    out: list[tuple[int, int, float]] = []
    for nr, ns in spiral.neighbors_rs(r, s):
        out.append((nr, ns, spiral.theta_from_s(ns)))
    return out


def verify_mask_matches_rs(metal: np.ndarray, spiral: SpiralChart, r: int, s: int) -> bool:
    try:
        ref = spiral.mask_from_rs(int(r), int(s))
    except ValueError:
        return False
    return hamming(ref, metal) == 0

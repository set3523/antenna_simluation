# -*- coding: utf-8 -*-
"""마스크·연결·시드. 학습/원/GA 가 같이 쓰는 최소 단위."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solver_cpu import rectangle_mask

PIX = 51
CENTER = PIX // 2

DIFFUSE_K = np.array(
    [[0.05, 0.10, 0.05], [0.10, 0.40, 0.10], [0.05, 0.10, 0.05]], dtype=np.float64
)


@dataclass
class Individual:
    metal: np.ndarray
    mode: int  # -1: 대칭 강제 없음
    score: float | None = None
    s11: np.ndarray | None = None
    air: np.ndarray | None = None
    worst: float | None = None
    d_use: float | None = None
    eta: float | None = None
    src: str = "ga"


def clone_ind(ind: Individual) -> Individual:
    return Individual(
        metal=ind.metal.copy(),
        mode=ind.mode,
        score=ind.score,
        s11=None if ind.s11 is None else np.array(ind.s11, copy=True),
        air=None if ind.air is None else np.array(ind.air, copy=True),
        worst=ind.worst,
        d_use=ind.d_use,
        eta=ind.eta,
        src=ind.src,
    )


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


def repair(ind: Individual, feed: tuple[int, int]) -> Individual:
    metal = keep_connected(ind.metal.astype(np.uint8), feed)
    if ind.mode >= 0:
        metal = apply_symmetry(metal, ind.mode)
        metal = keep_connected(metal, feed)
    metal[feed] = 1
    return Individual(metal=metal, mode=ind.mode, src=ind.src)


def seed_rectangle(feed: tuple[int, int]) -> Individual:
    g = rectangle_mask(PIX, 9.0, 11.7)
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

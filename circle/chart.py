# -*- coding: utf-8 -*-
"""PES 차트 원 위의 한 점 → 마스크. 구리 실루엣 원이 아님."""
from __future__ import annotations

import numpy as np

from core import PIX, Individual, repair


def hamming_ring(center: np.ndarray, r: float, rng: np.random.Generator, feed) -> np.ndarray:
    """B 축이 없을 때: 중심에서 약 r 칸을 뒤집음."""
    g = np.asarray(center, dtype=np.uint8).copy()
    nflip = int(max(1, round(r)))
    flat = g.ravel()
    n = flat.size
    nflip = min(nflip, n - 1)
    idx = rng.choice(n, size=nflip, replace=False)
    fi = int(feed[0]) * PIX + int(feed[1])
    idx = np.array([int(k) for k in idx if int(k) != fi], dtype=np.int64)
    if len(idx) == 0:
        return g
    flat[idx] = 1 - flat[idx]
    g = flat.reshape(PIX, PIX)
    g[feed] = 1
    return g


def sample_on_ring(
    center: np.ndarray,
    r: float,
    theta: float,
    rng: np.random.Generator,
    feed,
    mode: int = -1,
    b=None,
) -> Individual:
    metal0 = np.asarray(center, dtype=np.float64)
    if b is not None and b.ready():
        delta = b.ring_delta(r, theta)
        logits = 6.0 * (metal0 - 0.5) + 2.2 * delta
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -12.0, 12.0)))
        g = (rng.random((PIX, PIX)) < p).astype(np.uint8)
        src = "circleB"
    else:
        g = hamming_ring(center, r, rng, feed)
        src = "circleH"
    kid = repair(Individual(g, mode, src=src), feed)
    kid.src = src
    return kid

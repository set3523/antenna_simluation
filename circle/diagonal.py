# -*- coding: utf-8 -*-
"""Diagonal contact (corner-only touch) detection, GA trimming, chart rejection."""
from __future__ import annotations

import numpy as np

from core import PIX

_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def diagonal_contact_pairs(metal: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Copper-copper diagonal neighbor pairs whose two orthogonal cells are both air."""
    m = np.asarray(metal, dtype=np.uint8)
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for i in range(PIX):
        for j in range(PIX):
            if m[i, j] == 0:
                continue
            for di, dj in _DIAG:
                ni, nj = i + di, j + dj
                if not (0 <= ni < PIX and 0 <= nj < PIX):
                    continue
                if m[ni, nj] == 0:
                    continue
                if m[i + di, j] or m[i, j + dj]:
                    continue
                key = (min(i, ni), min(j, nj), max(i, ni), max(j, nj))
                if key in seen:
                    continue
                seen.add(key)
                out.append(((i, j), (ni, nj)))
    return out


def has_diagonal_contact(metal: np.ndarray) -> bool:
    return len(diagonal_contact_pairs(metal)) > 0


def _mirror_j(j: int, axis_j: int) -> int:
    return int(2 * axis_j - j)


def strip_diagonal_contacts(
    metal: np.ndarray,
    feed: tuple[int, int],
    *,
    axis_j: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    For each diagonal-contact pair, set one cell to air (feed excluded). Trim the mirror (j) too.
    Cell to trim: the one farther from the feed by Manhattan distance.
    """
    fi, fj = int(feed[0]), int(feed[1])
    ax = int(fj if axis_j is None else axis_j)
    g = np.asarray(metal, dtype=np.uint8).copy()
    n_cut = 0

    def dist(p: tuple[int, int]) -> int:
        return abs(p[0] - fi) + abs(p[1] - fj)

    while True:
        pairs = diagonal_contact_pairs(g)
        if not pairs:
            break
        a, b = pairs[0]
        if a == (fi, fj):
            rem = b
        elif b == (fi, fj):
            rem = a
        elif dist(a) != dist(b):
            rem = a if dist(a) > dist(b) else b
        else:
            rem = max(a, b)
        if rem == (fi, fj):
            break
        ri, rj = rem
        g[ri, rj] = 0
        n_cut += 1
        mj = _mirror_j(rj, ax)
        if mj != rj and 0 <= mj < PIX and (ri, mj) != (fi, fj):
            g[ri, mj] = 0
            n_cut += 1
        g[fi, fj] = 1
    return g, n_cut


def enforce_feed_symmetry(metal: np.ndarray, feed: tuple[int, int], mode: int) -> np.ndarray:
    from core import apply_symmetry

    g = np.asarray(metal, dtype=np.uint8).copy()
    g[int(feed[0]), int(feed[1])] = 1
    if mode >= 0:
        g = apply_symmetry(g, mode)
        g[int(feed[0]), int(feed[1])] = 1
    return g

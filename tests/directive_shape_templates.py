# -*- coding: utf-8 -*-
"""Upper layer 51×51 pixels: directive templates in the Vivaldi, bowtie and tapered-horn family (1 mm/px).

Reference baseline: use "Vivaldi-like" example shapes as FDTD candidates.
The evolution goal is a map that beats this family.
"""
from __future__ import annotations

import numpy as np

from core import PIX, Individual, repair_upper


def _set(g: np.ndarray, i: int, j: int) -> None:
    if 0 <= i < PIX and 0 <= j < PIX:
        g[i, j] = 1


def finalize_metal(g: np.ndarray, feed: tuple[int, int], *, connect: bool = False) -> np.ndarray:
    ind = Individual(np.asarray(g, dtype=np.uint8), mode=-1)
    out = repair_upper(ind, feed, connect=connect)
    return np.asarray(out.metal, dtype=np.uint8)


def metal_vivaldi_exponential(
    feed: tuple[int, int],
    *,
    axis: str = "x",
    steps: int = 18,
    w0: float = 1.4,
    w1: float = 10.0,
    slot: int = 2,
) -> np.ndarray:
    """Exponential flare plus center slot (planar Vivaldi approximation)."""
    g = np.zeros((PIX, PIX), dtype=np.uint8)
    fi, fj = int(feed[0]), int(feed[1])
    g[fi, fj] = 1
    steps = int(np.clip(steps, 3, 24))
    w0 = max(float(w0), 0.8)
    w1 = max(float(w1), w0 + 0.5)
    alpha = np.log(w1 / w0) / max(steps - 1, 1)
    half_slot = max(0, int(slot) // 2)

    for sign in (-1, 1):
        for k in range(1, steps + 1):
            half_w = int(round(0.5 * w0 * np.exp(alpha * (k - 1))))
            half_w = max(half_w, half_slot + 1)
            if axis == "x":
                i = fi + sign * k
                for dj in range(-half_w, half_w + 1):
                    if abs(dj) <= half_slot:
                        continue
                    _set(g, i, fj + dj)
            else:
                j = fj + sign * k
                for di in range(-half_w, half_w + 1):
                    if abs(di) <= half_slot:
                        continue
                    _set(g, fi + di, j)

    for dj in range(-half_slot - 1, half_slot + 2):
        _set(g, fi, fj + dj)
    if axis == "x":
        for di in range(-1, 2):
            _set(g, fi + di, fj)
    else:
        for dj in range(-1, 2):
            _set(g, fi, fj + dj)
    return g


def metal_bowtie(
    feed: tuple[int, int],
    *,
    arm: int = 16,
    flare: float = 0.55,
    axis: str = "x",
) -> np.ndarray:
    """Two opposing triangular arms (bowtie / tapered dipole face)."""
    g = np.zeros((PIX, PIX), dtype=np.uint8)
    fi, fj = int(feed[0]), int(feed[1])
    _set(g, fi, fj)
    arm = int(np.clip(arm, 4, 22))
    for k in range(1, arm + 1):
        w = 1 + int(round(flare * k))
        for sign in (-1, 1):
            if axis == "x":
                i = fi + sign * k
                for dj in range(-w, w + 1):
                    _set(g, i, fj + dj)
            else:
                j = fj + sign * k
                for di in range(-w, w + 1):
                    _set(g, fi + di, j)
    return g


def metal_tapered_horn(
    feed: tuple[int, int],
    *,
    axis: str = "x",
    length: int = 18,
    w_throat: int = 2,
    w_mouth: int = 11,
) -> np.ndarray:
    """Linear tapered horn (simpler than Vivaldi)."""
    g = np.zeros((PIX, PIX), dtype=np.uint8)
    fi, fj = int(feed[0]), int(feed[1])
    _set(g, fi, fj)
    length = int(np.clip(length, 4, 24))
    w_throat = max(1, int(w_throat))
    w_mouth = max(w_throat + 1, int(w_mouth))
    for sign in (-1, 1):
        for k in range(1, length + 1):
            t = k / max(length, 1)
            half = int(round(0.5 * (w_throat * (1 - t) + w_mouth * t)))
            half = max(half, 1)
            if axis == "x":
                i = fi + sign * k
                for dj in range(-half, half + 1):
                    _set(g, i, fj + dj)
            else:
                j = fj + sign * k
                for di in range(-half, half + 1):
                    _set(g, fi + di, j)
    return g


def metal_cpw_flare(
    feed: tuple[int, int],
    *,
    steps: int = 16,
    gap0: int = 2,
    gap1: int = 8,
    axis: str = "y",
) -> np.ndarray:
    """Flare where a CPW slot opens up (Vivaldi family)."""
    g = np.zeros((PIX, PIX), dtype=np.uint8)
    fi, fj = int(feed[0]), int(feed[1])
    _set(g, fi, fj)
    steps = int(np.clip(steps, 3, 22))
    for sign in (-1, 1):
        for k in range(1, steps + 1):
            t = k / max(steps, 1)
            gap = int(round(gap0 * (1 - t) + gap1 * t))
            gap = max(gap, 1)
            if axis == "y":
                j = fj + sign * k
                for dj in (-gap - 2, -gap - 1, gap + 1, gap + 2):
                    _set(g, fi, j + dj)
                _set(g, fi - 1, j)
                _set(g, fi + 1, j)
            else:
                i = fi + sign * k
                for di in (-gap - 2, -gap - 1, gap + 1, gap + 2):
                    _set(g, i + di, fj)
                _set(g, i, fj - 1)
                _set(g, i, fj + 1)
    return g


def directive_template_catalog(feed: tuple[int, int]) -> list[tuple[str, np.ndarray]]:
    """Name, upper-layer metal (close to raw before repair; finished by finalize_metal)."""
    specs: list[tuple[str, np.ndarray]] = []

    for axis in ("x", "y"):
        for steps in (14, 18, 22):
            for w1 in (8.0, 11.0, 14.0):
                name = f"vivaldi_{axis}_s{steps}_w{int(w1)}"
                specs.append(
                    (
                        name,
                        metal_vivaldi_exponential(
                            feed, axis=axis, steps=steps, w1=w1, slot=2
                        ),
                    )
                )

    for axis in ("x", "y"):
        for arm in (12, 16, 20):
            specs.append((f"bowtie_{axis}_a{arm}", metal_bowtie(feed, arm=arm, axis=axis)))

    for axis in ("x", "y"):
        for length in (14, 18):
            specs.append(
                (
                    f"horn_{axis}_L{length}",
                    metal_tapered_horn(feed, axis=axis, length=length, w_mouth=10),
                )
            )

    for axis in ("x", "y"):
        specs.append((f"cpw_flare_{axis}", metal_cpw_flare(feed, axis=axis, steps=16)))

    out: list[tuple[str, np.ndarray]] = []
    seen: set[bytes] = set()
    for name, raw in specs:
        m = finalize_metal(raw, feed, connect=False)
        if int(m.sum()) < 20:
            continue
        key = m.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append((name, m))
    return out

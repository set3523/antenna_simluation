# -*- coding: utf-8 -*-
"""chart_redesign_spec §5: spiral chart checks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core import CENTER, FEED_MM, PIX, apply_symmetry, hamming, ij_from_mm
from circle.diagonal import has_diagonal_contact, strip_diagonal_contacts
from circle.spiral_chart import SpiralChart, seed_chart_metal


def _feed():
    return ij_from_mm(*FEED_MM)


def _spiral(r_max: int = 130) -> SpiralChart:
    feed = _feed()
    seed = seed_chart_metal(feed, symmetry_mode=4)
    return SpiralChart.from_seed(seed, feed, r_max)


def _indep_hamming(seed: np.ndarray, mask: np.ndarray, order, feed) -> int:
    fi, fj = feed
    n = 0
    for i, j in order:
        if seed[i, j] != mask[i, j]:
            n += 1
    return n


def test_hamming_r_and_2r():
    sp = _spiral()
    feed = sp.feed
    order = sp.order
    for r in (1, 4, 12, 24, 40):
        for s in (0, 17, 33, sp.s_limit // 2):
            if s + r > sp.M:
                continue
            m = sp.mask_from_rs(r, s)
            assert _indep_hamming(sp.seed_base, m, order, feed) == r
            assert hamming(sp.seed_base, m) == 2 * r


def test_overlap_r_plus_one():
    sp = _spiral()
    for s in (0, 5, 20, 50):
        for r in (1, 8, 15):
            if s + r + 1 > sp.M:
                continue
            a = sp.mask_from_rs(r, s)
            b = sp.mask_from_rs(r + 1, s)
            assert hamming(a, b) == 2


def test_angle_neighbor_hamming_two():
    sp = _spiral()
    for r in (2, 6, 12):
        for s in (0, 3, 10):
            if s + r > sp.M or s + 1 + r > sp.M:
                continue
            a = sp.mask_from_rs(r, s)
            b = sp.mask_from_rs(r, s + 1)
            assert _indep_hamming(a, b, sp.order, sp.feed) == 2


def test_unique_masks_in_chart():
    sp = _spiral()
    seen: set[bytes] = set()
    for r in range(1, min(25, sp.r_max + 1)):
        step = max(1, sp.s_limit // 8) if sp.s_limit else 1
        for s in range(0, sp.s_limit + 1, step):
            if s + r > sp.M:
                continue
            raw = sp.mask_from_rs(r, s).tobytes()
            assert raw not in seen
            seen.add(raw)


def test_symmetry_feed_axis():
    sp = _spiral()
    fi, fj = sp.feed
    for r in (0, 4, 12):
        for s in (0, 7, 19):
            if s + r > sp.M:
                continue
            m = sp.mask_from_rs(r, s)
            assert m[fi, fj] == 1
            mir = apply_symmetry(m, 4)
            assert np.array_equal(m, mir)


def test_no_diagonal_on_evaluated_chart_masks():
    """Only masks sent to FDTD have zero diagonals (reject_diagonal=True)."""
    sp = _spiral()
    n_ok = 0
    for r in range(1, 30):
        for s in range(0, sp.s_limit + 1, 3):
            if s + r > sp.M:
                continue
            th = sp.theta_from_s(s)
            ind = sp.individual_from_polar(float(r), th, reject_diagonal=True)
            if ind is None:
                continue
            assert not has_diagonal_contact(ind.metal)
            n_ok += 1
    assert n_ok > 20


def test_ga_strip_diagonal():
    feed = _feed()
    g = seed_chart_metal(feed, symmetry_mode=4)
    g[feed[0] + 2, feed[1] + 3] = 1
    g[feed[0] + 3, feed[1] + 4] = 1
    if has_diagonal_contact(g):
        g2, n = strip_diagonal_contacts(g, feed)
        assert not has_diagonal_contact(g2)
        assert g2[feed[0], feed[1]] == 1
        assert n >= 1


def test_s_plus_r_within_M():
    sp = _spiral()
    rng = np.random.default_rng(0)
    for _ in range(200):
        r = int(rng.integers(0, sp.r_max + 1))
        s = int(rng.integers(0, sp.s_limit + 1))
        if s + r <= sp.M:
            sp.mask_from_rs(r, s)


def test_visual_spiral_masks():
    """§5.8: sample PNGs for r=4,12,24,130."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")
    out = Path(__file__).resolve().parents[1] / "run_out" / "_spiral_chart_test"
    out.mkdir(parents=True, exist_ok=True)
    sp = _spiral(130)
    thetas = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    for r in (4, 12, 24, 130):
        if r > sp.r_max:
            continue
        fig, axes = plt.subplots(1, len(thetas), figsize=(3 * len(thetas), 3))
        if len(thetas) == 1:
            axes = [axes]
        for ax, th in zip(axes, thetas):
            s = sp.s_from_theta(float(th))
            if s + r > sp.M:
                ax.set_visible(False)
                continue
            m = sp.mask_from_rs(r, s)
            ax.imshow(m, origin="lower", cmap="copper")
            ax.set_title(f"r={r} s={s}")
            ax.axis("off")
        fig.savefig(out / f"spiral_r{r:03d}.png", dpi=100)
        plt.close(fig)


def test_legacy_index_vs_spiral_diagnosis(capsys):
    """§0: Hamming distance between the old index disk and spiral r (index= u+v·2^1301)."""
    from circle.index_map import index_to_mask, mask_to_index, mask_to_uv

    CHART_WIDTH = 1 << 1301
    feed = _feed()
    seed = seed_chart_metal(feed, symmetry_mode=4)
    anchor = mask_to_uv(seed)
    sp = _spiral()

    def legacy_mask_at(r: float, theta: float) -> np.ndarray:
        au, av = anchor
        u = int(au) + int(round(r * float(np.cos(theta))))
        v = int(av) + int(round(r * float(np.sin(theta))))
        idx = u + v * CHART_WIDTH
        return index_to_mask(idx)

    print("\n=== §0 diagnostics: old index disk vs spiral chart ===")
    print(f"anchor(mask_to_uv)={anchor}  M={sp.M}  r_max={sp.r_max}")
    for r in (4.0, 8.0, 24.0):
        for k, th in enumerate((0.0, 0.7, 1.4)):
            leg = legacy_mask_at(r, th)
            h_legacy = hamming(seed, leg)
            s = sp.s_from_theta(th)
            ri = int(round(r))
            if ri <= sp.r_max and s + ri <= sp.M:
                spi = sp.mask_from_rs(ri, s)
                h_sp = hamming(seed, spi)
                h_indep = _indep_hamming(seed, spi, sp.order, feed)
            else:
                h_sp = h_indep = -1
            print(
                f"  r={r:g} θ={th:.2f}  legacy_H={h_legacy}  "
                f"spiral_H={h_sp}  spiral_indep={h_indep}"
            )
    assert hamming(seed, seed) == 0

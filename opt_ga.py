# -*- coding: utf-8 -*-
"""Genetic algorithm. Evolution settings (fitness, symmetry, mutation) come from the Config passed in by main."""
from __future__ import annotations

import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from core import (
    PIX,
    CENTER,
    GND_MARK_MM,
    Individual,
    ij_from_mm,
    mm_from_ij,
    clone_ind,
    apply_symmetry,
    keep_connected,
    repair,
    repair_upper,
    seed_rectangle,
    seed_random,
    seed_fixed_patch_port,
    fixed_patch_metal,
    lower_layer_metal,
    patch_port_sites,
    effective_port,
    port_manhattan,
    hamming,
    is_feasible,
    DIFFUSE_K,
    diffuse_grow,
)
from circle.roots import keep_metric_parents, pick_breed_pair, pick_metric_pair
from learn import OffspringModel
from learn.viz import save_reveal

_LIVE = None
_LIVE_ROWS = 6


def _enable_vt() -> bool:
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes

        h = ctypes.windll.kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if not ctypes.windll.kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        return bool(ctypes.windll.kernel32.SetConsoleMode(h, mode.value | 0x0004))
    except Exception:
        return False


class LivePanel:
    """Last 5 lines + progress bar. Updates in place on a TTY; otherwise overwrites the bar with \\r on each eval."""

    def __init__(self, n: int = 5):
        self.n = n
        self.buf: list[str] = []
        self.drawn = False
        self.tty = _enable_vt()
        self.eval_n = 0
        self.frac = 0.0
        self.gen = 0
        self.total = 1
        self.extra = "start"
        self.circle_r = None
        self.circle_rmax = None
        self.circle_r_full = None
        self.stall = ""
        self.bar = ""
        self._render_bar()

    def start(self):
        if self.tty:
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()
        self._emit()

    def _render_bar(self):
        width = 28
        frac = min(max(float(self.frac), 0.0), 1.0)
        filled = int(round(frac * width))
        bar = "#" * filled + "-" * (width - filled)
        extra = self.extra.strip()
        tail = f"  {extra}" if extra else ""
        if self.circle_r is not None:
            rmax = self.circle_rmax if self.circle_rmax else 0.0
            rfull = float(self.circle_r_full) if self.circle_r_full else 0.0
            if rfull > 1e-6:
                map_pct = 100.0 * min(float(self.circle_r) / rfull, 1.0)
                phase = f"map~{map_pct:.0f}%  r={self.circle_r:g}/{rmax:g}(limit)"
            else:
                phase = f"trend r={self.circle_r:g}/{rmax:g}"
            if self.stall:
                phase += f"  live {self.stall}"
        elif self.gen <= 0:
            phase = "circle"
        else:
            phase = f"gen {self.gen}/{max(int(self.total), 1)}"
        self.bar = f"[{bar}] {100.0 * frac:5.1f}%  {phase}  eval {self.eval_n}{tail}"

    def bump_eval(self):
        self.eval_n += 1
        self._render_bar()
        self._emit()

    def push(self, text: str):
        line = " ".join(text.split())
        self.buf.append(line)
        self.buf = self.buf[-self.n :]
        if self.tty:
            self._emit()
        else:
            print(f"  {line}", flush=True)
            self._emit_bar_only()

    def set_bar(
        self,
        frac: float,
        gen: int,
        total: int,
        extra: str = "",
        circle_r=None,
        circle_rmax=None,
        circle_r_full=None,
        stall=None,
    ):
        self.frac = min(max(float(frac), 0.0), 1.0)
        self.gen = int(gen)
        self.total = max(int(total), 1)
        if extra.strip():
            self.extra = extra.strip()
        if circle_r is not None:
            self.circle_r = float(circle_r)
            if circle_rmax is not None:
                self.circle_rmax = float(circle_rmax)
            if circle_r_full is not None:
                self.circle_r_full = float(circle_r_full)
            if stall is not None:
                self.stall = str(stall)
        elif gen >= 1:
            self.circle_r = None
            self.circle_r_full = None
            self.stall = ""
        self._render_bar()
        self._emit()

    def persist(self, msg: str):
        """Keep status text above; redraw the last 5 lines + bar below it."""
        line = " ".join(str(msg).split())
        if self.tty and self.drawn:
            sys.stdout.write(f"\033[{_LIVE_ROWS}A")
            sys.stdout.write("\033[2K\r" + line[: self._cols()] + "\n")
            self.drawn = False
            self._draw()
            return
        if not self.tty:
            sys.stdout.write("\n")
        print(line, flush=True)
        self._emit()

    def _cols(self) -> int:
        try:
            cols = os.get_terminal_size().columns
        except OSError:
            cols = 100
        return max(40, min(cols, 160) - 1)

    def _emit(self):
        if self.tty:
            self._draw()
        else:
            self._emit_bar_only()

    def _emit_bar_only(self):
        sys.stdout.write("\r\033[2K" + self.bar[: self._cols()])
        sys.stdout.flush()

    def _draw(self):
        cols = self._cols()
        lines = [""] * (self.n - len(self.buf)) + list(self.buf)
        lines.append(self.bar)
        if self.drawn:
            sys.stdout.write(f"\033[{_LIVE_ROWS}A")
        for ln in lines:
            sys.stdout.write("\033[2K\r" + ln[:cols] + "\n")
        sys.stdout.flush()
        self.drawn = True

    def close(self):
        if not self.tty:
            sys.stdout.write("\n")
        if self.tty:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
        self.drawn = False


def _mute_fd2():
    class _CM:
        def __enter__(self):
            self.saved = self.dn = None
            try:
                self.dn = os.open(os.devnull, os.O_WRONLY)
                self.saved = os.dup(2)
                os.dup2(self.dn, 2)
            except OSError:
                self.saved = self.dn = None
            return self

        def __exit__(self, *exc):
            if self.saved is not None:
                os.dup2(self.saved, 2)
                os.close(self.saved)
            if self.dn is not None:
                os.close(self.dn)
            return False

    return _CM()



def _reveal(cfg, model, tag: str, note: str = "", center=None):
    if model is None:
        return
    if center is not None:
        model.chart.set_center(center)
    det = save_reveal(cfg.out_dir, tag, model, note=note)
    if _LIVE is not None:
        _LIVE.persist(f"    reveal {tag}  FDTD samples n={int(det)} (not all 2601 indices)")


def _repair_cfg(ind: Individual, feed, cfg) -> Individual:
    return repair_upper(ind, feed, connect=bool(getattr(cfg, "connect_repair", False)))


def crossover(a: Individual, b: Individual, rng: np.random.Generator, feed, cfg=None) -> Individual:
    """Per-cell 50% mixing breaks the shape, so mix in blocks."""
    child = np.asarray(a.metal, dtype=np.uint8).copy()
    other = np.asarray(b.metal, dtype=np.uint8)
    block = 8
    for i in range(0, PIX, block):
        for j in range(0, PIX, block):
            if rng.random() < 0.5:
                child[i : i + block, j : j + block] = other[i : i + block, j : j + block]
    mode = a.mode if rng.random() < 0.5 else b.mode
    return _repair_cfg(Individual(child, mode), feed, cfg)


def _feed_sites_on_patch(cfg) -> list[tuple[int, int]]:
    lx = float(getattr(cfg, "patch_lx_mm", 9.0))
    ly = float(getattr(cfg, "patch_ly_mm", 11.7))
    return patch_port_sites(fixed_patch_metal(lx, ly))


def crossover_feed(
    a: Individual, b: Individual, rng: np.random.Generator, cfg
) -> Individual:
    pa = effective_port(a, cfg.feed)
    pb = effective_port(b, cfg.feed)
    port = pa if rng.random() < 0.5 else pb
    child = seed_fixed_patch_port(
        port[0], port[1], float(cfg.patch_lx_mm), float(cfg.patch_ly_mm)
    )
    child.mode = a.mode if rng.random() < 0.5 else b.mode
    return child


def mutate_feed(
    ind: Individual,
    rng: np.random.Generator,
    cfg,
) -> Individual:
    sites = _feed_sites_on_patch(cfg)
    site_set = set(sites)
    if not sites:
        return clone_ind(ind)
    cur = effective_port(ind, cfg.feed)
    if rng.random() < 0.35:
        port = sites[int(rng.integers(0, len(sites)))]
    else:
        nbr = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (int(cur[0]) + di, int(cur[1]) + dj)
            if p in site_set:
                nbr.append(p)
        port = nbr[int(rng.integers(0, len(nbr)))] if nbr else cur
    out = seed_fixed_patch_port(
        port[0], port[1], float(cfg.patch_lx_mm), float(cfg.patch_ly_mm)
    )
    out.mode = ind.mode
    out.src = ind.src
    return out


def pick_feed_pair(
    parents: list[Individual],
    rng: np.random.Generator,
    min_dist: int = 3,
) -> tuple[Individual, Individual] | None:
    scored = [p for p in parents if p.score is not None]
    if len(scored) < 2:
        return None
    rng.shuffle(scored)
    for i, a in enumerate(scored):
        pa = effective_port(a, (0, 0))
        for b in scored[i + 1 :]:
            pb = effective_port(b, (0, 0))
            if port_manhattan(pa, pb) >= min_dist:
                return a, b
    return scored[0], scored[1]


def build_feed_population(
    solver,
    cfg,
    rng,
    freqs,
    model: OffspringModel | None = None,
    progress=None,
) -> list[Individual]:
    """FDTD for every port on the fixed patch, sorted by score."""
    sites = _feed_sites_on_patch(cfg)
    pop: list[Individual] = []
    n = len(sites)
    for k, (i, j) in enumerate(sites):
        ind = seed_fixed_patch_port(
            i, j, float(cfg.patch_lx_mm), float(cfg.patch_ly_mm)
        )
        fill_eval(solver, ind, freqs, cfg, model=model, elite=None, note="feed", log_chart=False)
        pop.append(ind)
        if progress is not None:
            progress((k + 1) / max(n, 1), f"feed {k+1}/{n}  port={mm_from_ij(i,j)}")
    pop.sort(key=lambda x: x.score if x.score is not None else -1e18, reverse=True)
    return pop


def mutate(
    ind: Individual,
    rng: np.random.Generator,
    feed,
    mut_pixels: int,
    diffuse_steps: int = 2,
    diffuse_rate: float = 0.4,
    kernel: np.ndarray | None = None,
    cfg=None,
) -> Individual:
    g = ind.metal.copy()
    flip_p = float(getattr(cfg, "mut_flip_frac", 0.5) if cfg is not None else 0.5)
    ys, xs = np.nonzero(g)
    if len(xs) == 0:
        g[feed] = 1
        return Individual(g, ind.mode)
    for _ in range(mut_pixels):
        if rng.random() < flip_p:
            i = int(rng.integers(0, PIX))
            j = int(rng.integers(0, PIX))
            g[i, j] = 1 - int(g[i, j] >= 0.5)
        elif len(xs):
            k = int(rng.integers(0, len(xs)))
            i, j = int(ys[k]), int(xs[k])
            if rng.random() < 0.5:
                g[i, j] = 0
            else:
                di, dj = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
                ni, nj = np.clip(i + di, 0, PIX - 1), np.clip(j + dj, 0, PIX - 1)
                g[ni, nj] = 1
        else:
            i = int(rng.integers(0, PIX))
            j = int(rng.integers(0, PIX))
            g[i, j] = 1
        ys, xs = np.nonzero(g)
    g = diffuse_grow(g, rng, diffuse_rate, diffuse_steps, kernel=kernel)
    g[feed] = 1
    return _repair_cfg(Individual(g, ind.mode), feed, cfg)


def air_metrics(air: np.ndarray | None, r_mm: float = 15.0):
    """
    |E|_rms on the air plane. Larger the more the field concentrates at one spot, wherever it is.
    D_dB : peak/mean of |E|^2
    beam : energy fraction within r_mm of the peak (not the center)
    peak : peak position (mm)
    """
    if air is None:
        return 0.0, 0.0, None, 0.0
    e = np.asarray(air, dtype=np.float64)
    e2 = e * e
    p_all = float(e2.sum()) + 1e-30
    d_lin = float(e2.max() / (e2.mean() + 1e-30))
    d_db = 10.0 * np.log10(d_lin + 1e-15)
    nx, ny = e.shape
    x = np.arange(nx) - (nx - 1) / 2.0
    y = np.arange(ny) - (ny - 1) / 2.0
    ip, jp = np.unravel_index(int(np.argmax(e2)), e2.shape)
    px, py = float(x[ip]), float(y[jp])
    r2 = (x[:, None] - px) ** 2 + (y[None, :] - py) ** 2
    beam = float(e2[r2 <= r_mm**2].sum() / p_all)
    w = e2 / p_all
    cx = float((w * x[:, None]).sum())
    cy = float((w * y[None, :]).sum())
    r_rms = float(np.sqrt((w * ((x[:, None] - cx) ** 2 + (y[None, :] - cy) ** 2)).sum()))
    return float(d_db), beam, (px, py), r_rms


def cfg_f0_hz(cfg) -> float:
    """Design/display center Hz: cfg.f0_hz, or the band/sweep midpoint."""
    if cfg is None:
        return 6.0e9
    f0 = getattr(cfg, "f0_hz", None)
    if f0 is not None and float(f0) > 0.0:
        return float(f0)
    band = getattr(cfg, "band_hz", None)
    if band is not None:
        return 0.5 * (float(band[0]) + float(band[1]))
    return 0.5 * (float(getattr(cfg, "f_start", 3e9)) + float(getattr(cfg, "f_stop", 9e9)))


def cfg_excitation_fc_hz(cfg) -> float:
    """FDTD Gaussian FC = band half-width (f_max−f_min)/2."""
    if cfg is None:
        return 3.0e9
    fc = getattr(cfg, "excitation_fc_hz", None)
    if fc is not None and float(fc) > 0.0:
        return float(fc)
    band = getattr(cfg, "band_hz", None)
    if band is not None:
        return 0.5 * (float(band[1]) - float(band[0]))
    f0 = float(getattr(cfg, "f_start", 3e9))
    f1 = float(getattr(cfg, "f_stop", 9e9))
    return 0.5 * (f1 - f0)


def _s11_band_mask(freqs: np.ndarray, cfg) -> np.ndarray:
    band = (freqs >= cfg.band_hz[0]) & (freqs <= cfg.band_hz[1])
    if not np.any(band):
        return np.ones_like(freqs, dtype=bool)
    return band


def _band_match_eta_per_hz(s11: np.ndarray, freqs: np.ndarray, cfg) -> np.ndarray:
    """1-|S11|² (0~1) at each frequency in the band."""
    band = _s11_band_mask(freqs, cfg)
    mag2 = np.clip(10.0 ** (np.asarray(s11)[band] / 10.0), 0.0, 0.999)
    return 1.0 - mag2


def match_efficiency(s11: np.ndarray, freqs: np.ndarray, cfg) -> float:
    """Band-average match efficiency 1-|S11|^2 (plot η_avg; can be high from one deep dip)."""
    per = _band_match_eta_per_hz(s11, freqs, cfg)
    return float(np.mean(per)) if per.size else 0.0


def match_efficiency_min(s11: np.ndarray, freqs: np.ndarray, cfg) -> float:
    """η at the worst (most reflective) frequency in the band; tracks uniformly low S11."""
    per = _band_match_eta_per_hz(s11, freqs, cfg)
    return float(np.min(per)) if per.size else 0.0


def match_efficiency_for_score(s11: np.ndarray, freqs: np.ndarray, cfg) -> tuple[float, float, float]:
    """η for scoring: blend of band average and band worst."""
    per = _band_match_eta_per_hz(s11, freqs, cfg)
    if per.size == 0:
        return 0.0, 0.0, 0.0
    eta_avg = float(np.mean(per))
    eta_min = float(np.min(per))
    mix = float(np.clip(getattr(cfg, "eta_band_min_mix", 0.55), 0.0, 1.0))
    eta_sc = eta_avg * (1.0 - mix) + eta_min * mix
    return eta_sc, eta_avg, eta_min


CONE_HALF_DEG = 30.0  # cone 60° full angle (camera FOV)
FACE_AXES = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)
FACE_NAME = ("+x", "-x", "+y", "-y", "+z", "-z")
# v4 grid: NZ_BELOW=28, NZ_SUB=4, NZ_ABOVE=40 → NZ=72, K_GND=28, K_PATCH=32 (same as native)
GRID_NZ = 72
GRID_K_GND = 28
GRID_K_PATCH = 32
Z_PATCH_MM = GRID_K_PATCH * 0.375
Z_FACE_MM = (GRID_NZ - 3 - GRID_K_PATCH) * 0.375  # +z plane for plots (above the patch)
Z_MID_MM = (GRID_K_GND + 2) * 0.375
Z_AIR_MM = (GRID_K_PATCH + 4) * 0.375
Z_TOP_MM = (GRID_NZ - 3) * 0.375


def _cone_frac(faces) -> float:
    a = np.asarray(faces, dtype=np.float64).reshape(-1) if faces is not None else np.zeros(0)
    if a.size < 12:
        return float("nan")
    return float(np.clip(a[6 + _best_face(faces)], 0.0, 1.0))


def _best_face(faces) -> int:
    if faces is None:
        return 4
    a = np.asarray(faces, dtype=np.float64).reshape(-1)
    if a.size >= 13 and a[12] == 1.0:
        return int(np.argmax(a[6:12]))
    if a.size < 6 or not np.any(np.isfinite(a[:6])) or float(np.max(a[:6])) <= 0.0:
        return 4
    return int(np.argmax(a[:6]))


def _draw_cone_xy(ax, faces, z_mm: float = Z_FACE_MM):
    """60° full-angle cone on the top |E|. Defaults to the scoring +z plane height (for visibility)."""
    fi = _best_face(faces)
    nx, ny, nz = FACE_AXES[fi]
    half = np.deg2rad(CONE_HALF_DEG)
    ca, sa = float(np.cos(half)), float(np.sin(half))
    if abs(nz) > 0.9:
        if nz * z_mm <= 0:
            return FACE_NAME[fi]
        rad = abs(z_mm) * sa / max(ca, 1e-9)
        ax.add_patch(
            plt.Circle((0.0, 0.0), rad, fill=True, facecolor="#00e5ff", alpha=0.16, zorder=5)
        )
        ax.add_patch(
            plt.Circle((0.0, 0.0), rad, fill=False, edgecolor="#00e5ff", lw=2.2, zorder=6)
        )
        ax.plot(0.0, 0.0, marker="+", color="#00e5ff", ms=9, mew=1.6, zorder=7)
        return FACE_NAME[fi]
    t = np.linspace(-40.0, 40.0, 240)
    cot = ca / max(sa, 1e-9)
    if abs(nx) > 0.9:
        xx = np.sign(nx) * cot * np.sqrt(t**2 + z_mm**2)
        ax.plot(xx, t, color="#00e5ff", lw=2.2, zorder=6)
    else:
        yy = np.sign(ny) * cot * np.sqrt(t**2 + z_mm**2)
        ax.plot(t, yy, color="#00e5ff", lw=2.2, zorder=6)
    return FACE_NAME[fi]


def _draw_cone_xz(ax, faces, z_patch: float = Z_PATCH_MM):
    """Cone's two edges in the front xz view. If the axis is ±y it doesn't intersect this cut."""
    fi = _best_face(faces)
    nx, ny, nz = FACE_AXES[fi]
    half = np.deg2rad(CONE_HALF_DEG)
    z1 = z_patch + 20.0
    xl, xr = ax.get_xlim()
    zb, zt = ax.get_ylim()
    if abs(nz) > 0.9:
        z_end = zt if nz > 0 else zb
        z = np.linspace(z_patch, z_end, 32)
        w = np.abs(z - z_patch) * np.tan(half)
        ax.fill_betweenx(z, -w, w, color="#00e5ff", alpha=0.16, zorder=4)
        ax.plot(w, z, color="#00e5ff", lw=2.2, zorder=6)
        ax.plot(-w, z, color="#00e5ff", lw=2.2, zorder=6)
        return FACE_NAME[fi]
    if abs(nx) > 0.9:
        x_end = xr if nx > 0 else xl
        x = np.linspace(0.0, x_end, 32)
        rise = np.abs(x) * np.tan(half)
        ax.fill_between(x, z_patch - rise, z_patch + rise, color="#00e5ff", alpha=0.16, zorder=4)
        ax.plot(x, z_patch + rise, color="#00e5ff", lw=2.2, zorder=6)
        ax.plot(x, z_patch - rise, color="#00e5ff", lw=2.2, zorder=6)
        return FACE_NAME[fi]
    return FACE_NAME[fi]


def face_directivity(faces) -> tuple[float, float, float]:
    """
    faces[0:6] mean |E|^2 on the 6 faces, faces[6:12] per-face 60° full-angle cone power fraction.
    bias = strongest face / 6-face mean (dB).
    cone = power in that axis cone / total, in dB relative to isotropic.
    D_dir = 0.5 bias + 0.5 cone.
    """
    if faces is None:
        return 0.0, 0.0, 0.0
    a = np.asarray(faces, dtype=np.float64).reshape(-1)
    if a.size < 6 or not np.any(np.isfinite(a[:6])) or float(np.max(a[:6])) <= 0.0:
        return 0.0, 0.0, 0.0
    half = np.deg2rad(CONE_HALF_DEG)
    iso = 0.5 * (1.0 - float(np.cos(half)))
    if a.size >= 13 and a[12] == 1.0:
        # v4 far field (NTFF): [0:6] axial directivity (linear, isotropic=1), [6:12] radiated power fraction in the axis cone (60° full angle).
        # pick the axis with the largest cone; bias = that axis directivity in dBi.
        cone6 = np.clip(a[6:12], 0.0, 1.0)
        best = int(np.argmax(cone6))
        cone_db = 10.0 * np.log10(float(cone6[best]) / max(iso, 1e-9) + 1e-15)
        d_bias = 10.0 * np.log10(max(float(a[best]), 1e-15))
        return float(0.5 * d_bias + 0.5 * cone_db), float(d_bias), float(cone_db)
    inten = np.clip(a[:6], 0.0, None)
    mean = float(np.mean(inten) + 1e-30)
    best = int(np.argmax(inten))
    d_bias = 10.0 * np.log10(inten[best] / mean + 1e-15)
    cone_frac = float(a[6 + best]) if a.size >= 12 else 0.0
    cone_frac = float(np.clip(cone_frac, 0.0, 1.0))
    half = np.deg2rad(CONE_HALF_DEG)
    iso = 0.5 * (1.0 - float(np.cos(half)))
    cone_db = 10.0 * np.log10(cone_frac / max(iso, 1e-9) + 1e-15)
    d_dir = 0.5 * d_bias + 0.5 * cone_db
    return float(d_dir), float(d_bias), float(cone_db)


def gain_parts(s11, freqs, cfg, air=None, metal=None, faces=None, leave=None):
    """
    Trend line: cone 60°, η, out/in
    Score: w_cone*cone + w_eta*η_dB + w_leave*out/in
    """
    d_db, beam, _peak, r_rms = air_metrics(air)
    if s11 is not None:
        _eta_sc, eta, eta_min = match_efficiency_for_score(s11, freqs, cfg)
        eta_db = 10.0 * np.log10(_eta_sc + 1e-15)
    else:
        eta, eta_min = 0.0, 0.0
        eta_db = -150.0
    f0 = cfg_f0_hz(cfg)
    lam_mm = 3.0e8 / max(f0, 1.0) * 1e3
    copper = int(np.asarray(metal).sum()) if metal is not None else 0
    area = float(max(copper, 1))
    d_ap = 10.0 * np.log10(np.clip(4.0 * np.pi * area / (lam_mm**2), 1e-8, 1e6))
    _d_use, face6_db, cone_db = face_directivity(faces)
    leave_r = float(leave) if leave is not None and np.isfinite(float(leave)) else 0.0
    leave_db = 10.0 * np.log10(max(leave_r, 1e-30))
    return cone_db, eta, eta_db, beam, r_rms, copper, d_ap, cone_db, face6_db, leave_db, leave_r, eta_min


def _gain_metric_effective(cone_db: float, eta_db: float, leave_db: float, cfg) -> tuple[float, float, float, float]:
    """Cone and leave outside min/max become 0. η is taken as-is from S11."""
    cmin = float(getattr(cfg, "cone_db_min", 0.0))
    lmin = float(getattr(cfg, "leave_db_min", 0.0))
    lmax = float(getattr(cfg, "leave_db_max", 18.0))
    cone_u = float(cone_db) if float(cone_db) >= cmin else 0.0
    raw_leave = float(leave_db)
    if lmax <= 0.0:
        # v4: leave = total efficiency (P_rad/P_inc) dB ≤ 0. Clip out-of-range to the bounds (lower stays worse)
        leave_u = float(np.clip(raw_leave, lmin, lmax))
    elif raw_leave < lmin or raw_leave > lmax:
        leave_u = 0.0
    else:
        leave_u = raw_leave
    eta_u = float(eta_db)
    return cone_u, eta_u, leave_u, leave_u


def _score_balance_bonus(
    cone_u: float, eta_u: float, leave_u: float, cfg, *, face6_u: float = 0.0
) -> float:
    """Small bonus the more balanced the metrics are."""
    w = float(getattr(cfg, "score_balance_w", 0.0))
    if w <= 0.0:
        return 0.0
    spread_lim = max(float(getattr(cfg, "score_balance_spread", 0.38)), 1e-6)
    l_shift = 6.0 if float(getattr(cfg, "leave_db_max", 18.0)) <= 0.0 else 2.0  # v4 total efficiency dB gets +6 like η
    parts = [max(cone_u + 2.0, 0.08), max(eta_u + 6.0, 0.08), max(leave_u + l_shift, 0.08)]
    wf = float(getattr(cfg, "w_face6", 0.0))
    if wf > 0.0:
        parts.append(max(face6_u + 2.0, 0.08))
    arr = np.array(parts, dtype=np.float64)
    m = float(arr.mean())
    rel = arr / m
    spread = float(np.std(rel))
    return w * max(0.0, 1.0 - spread / spread_lim)


def gain_score(
    cone_db: float, eta_db: float, leave_db: float, face6_db: float, cfg
) -> tuple[float, float]:
    """
    FITNESS=gain. cone 60° + 6-face bias + η + leave (6-face total out/in).
    """
    cone_u, eta_u, leave_u, ld = _gain_metric_effective(cone_db, eta_db, leave_db, cfg)
    f6min = float(getattr(cfg, "face6_db_min", 3.0))
    face6_u = float(face6_db) if float(face6_db) >= f6min else 0.0
    wc = float(getattr(cfg, "w_cone", getattr(cfg, "w_directivity", 1.0)))
    wf = float(getattr(cfg, "w_face6", 0.0))
    we = float(getattr(cfg, "w_efficiency", 1.0))
    wl = float(getattr(cfg, "w_leave", 1.0))
    wsum = max(wc + wf + we + wl, 1e-9)
    lin = wc * cone_u + wf * face6_u + we * eta_u + wl * leave_u
    mix = float(np.clip(getattr(cfg, "score_geom_mix", 0.0), 0.0, 1.0))
    bal = _score_balance_bonus(cone_u, eta_u, leave_u, cfg, face6_u=face6_u)
    if mix <= 0.0:
        return float(lin + bal), ld
    pc = max(cone_u, 0.0) + 0.25
    pf = max(face6_u, 0.0) + 0.25
    pe = max(eta_u + 6.0, 0.05)
    if float(getattr(cfg, "leave_db_max", 18.0)) <= 0.0:
        pl = max(leave_u + 6.0, 0.05)  # v4 total efficiency dB
    else:
        pl = max(leave_u, 0.0) + 0.25
    logsum = wc * np.log(pc) + we * np.log(pe) + wl * np.log(pl)
    if wf > 0.0:
        logsum += wf * np.log(pf)
    geom = float(np.exp(logsum / wsum))
    score = (1.0 - mix) * lin + mix * geom + bal
    return float(score), ld


def score_s11(
    s11: np.ndarray, freqs: np.ndarray, cfg, air=None, metal=None, faces=None, leave=None
) -> tuple[float, float]:
    """Always higher is better."""
    band = (freqs >= cfg.band_hz[0]) & (freqs <= cfg.band_hz[1])
    if not np.any(band):
        band = np.ones_like(s11, dtype=bool)
    worst = float(np.max(s11[band]))
    f0 = cfg_f0_hz(cfg)
    at_f0 = float(s11[np.argmin(np.abs(freqs - f0))])
    fitness = getattr(cfg, "fitness", "gain")
    if fitness == "f0":
        return -at_f0, worst
    if fitness == "bandwidth":
        below = s11 < cfg.s11_target_db
        return float(np.mean(below[band])) * 20.0 - 0.2 * worst, worst
    if fitness == "band_worst":
        return -0.7 * worst - 0.3 * at_f0, worst
    tgt = float(getattr(cfg, "s11_target_db", -10.0))
    s11_band = np.asarray(s11, dtype=np.float64)[band]
    match_frac = (
        float(np.mean(s11_band < tgt + 1e-9)) if s11_band.size else 0.0
    )
    _cone, eta, eta_db, _beam, _r_rms, copper, _d_ap, cone_db, face6_db, leave_db, _lv, _eta_min = gain_parts(
        s11, freqs, cfg, air=air, metal=metal, faces=faces, leave=leave
    )
    score, _ld = gain_score(cone_db, eta_db, leave_db, face6_db, cfg)
    ww = float(getattr(cfg, "w_band_worst", 1.5))
    wm = float(getattr(cfg, "w_band_match_frac", 15.0))
    score += ww * (-worst / 10.0) + wm * match_frac
    if worst > tgt:
        pk = float(getattr(cfg, "s11_score_penalty_k", 5.0))
        score -= pk * float(worst - tgt)
    return float(score), worst


def _field_limits(e: np.ndarray) -> tuple[float, float, float, float]:
    finite = e[np.isfinite(e)]
    emax = float(np.max(finite)) if finite.size else 1.0
    pos = finite[finite > 0]
    emin = float(np.min(pos)) if pos.size else 0.0
    vlo = float(np.percentile(finite, 2)) if finite.size else 0.0
    vhi = emax if np.isfinite(emax) and emax > vlo else vlo + 1e-30
    return emin, emax, vlo, vhi


def _xy_extent(nx: int, ny: int) -> list[float]:
    return [-nx / 2.0, nx / 2.0, -ny / 2.0, ny / 2.0]


# native/voxel_fdtd.cpp PAD — position of the 51×51 copper plate in the FDTD grid
FDTD_PAD = 12
METAL_XY_EXT = (-25.5, 25.5, -25.5, 25.5)
# Copper map panels only: no copper=green, copper=gold, port/ground marks=red/blue (|E| stays inferno)
_COLOR_NO_COPPER = (0.18, 0.58, 0.28)
_COLOR_COPPER = (0.85, 0.68, 0.12)
_COLOR_GND = (0.22, 0.52, 0.98)
_COLOR_FEED = (0.95, 0.12, 0.12)
_K_GND = GRID_K_GND
_K_PATCH = GRID_K_PATCH
_CELL_MM = 1.0
_DZ_MM = 0.375


def _metal_xy_rgb(
    metal,
    feed,
    *,
    show_gnd: bool = True,
    show_feed: bool = True,
) -> np.ndarray:
    m = np.asarray(metal, dtype=np.uint8)
    rgb = np.empty((PIX, PIX, 3), dtype=np.float32)
    rgb[:, :] = _COLOR_NO_COPPER
    show = m.T
    rgb[show >= 1] = _COLOR_COPPER
    if show_gnd:
        gix, giy = ij_from_mm(*GND_MARK_MM)
        if 0 <= gix < PIX and 0 <= giy < PIX:
            rgb[giy, gix] = _COLOR_GND
    if show_feed:
        ix, iy = int(feed[0]), int(feed[1])
        if 0 <= ix < PIX and 0 <= iy < PIX:
            rgb[iy, ix] = _COLOR_FEED
    return rgb


def _plot_metal_xy_panel(ax, metal, feed, title: str, *, fontsize: int = 9) -> None:
    rgb = _metal_xy_rgb(metal, feed)
    ax.imshow(
        rgb,
        origin="lower",
        extent=list(METAL_XY_EXT),
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")


def _plot_single_layer_panel(
    ax,
    metal,
    feed,
    subtitle: str,
    *,
    fontsize: int = 9,
    show_gnd: bool = False,
    show_feed: bool = True,
) -> None:
    rgb = _metal_xy_rgb(
        metal, feed, show_gnd=show_gnd, show_feed=show_feed
    )
    ax.imshow(
        rgb,
        origin="lower",
        extent=list(METAL_XY_EXT),
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(subtitle, fontsize=fontsize)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")


def _plot_two_layer_metal_on_fig(
    fig,
    gridspec_cell,
    upper_metal,
    feed,
    title: str,
    *,
    fontsize: int = 9,
    cfg=None,
) -> None:
    """Full xy for back (lower) and front (upper) layers separately; y is not split on one axis."""
    lx = float(getattr(cfg, "patch_lx_mm", 9.0)) if cfg is not None else 9.0
    ly = float(getattr(cfg, "patch_ly_mm", 11.7)) if cfg is not None else 11.7
    lo = lower_layer_metal(lx, ly)
    sub = gridspec_cell.subgridspec(2, 1, height_ratios=(1, 1), hspace=0.32)
    ax_lo = fig.add_subplot(sub[0, 0])
    ax_hi = fig.add_subplot(sub[1, 0])
    _plot_single_layer_panel(
        ax_lo,
        lo,
        feed,
        "Lower layer (back), fixed GND rectangle",
        fontsize=max(7, fontsize - 1),
        show_gnd=True,
        show_feed=False,
    )
    _plot_single_layer_panel(
        ax_hi,
        np.asarray(upper_metal, dtype=np.uint8),
        feed,
        f"Upper layer (front), chart map\n{title}",
        fontsize=max(7, fontsize - 1),
        show_gnd=False,
        show_feed=True,
    )


def _plot_two_layer_metal(
    ax,
    upper_metal,
    feed,
    title: str,
    *,
    fontsize: int = 9,
    cfg=None,
) -> None:
    """Single ax slot (byproduct etc.): delegate to fig + cell."""
    fig = ax.figure
    pos = ax.get_position()
    ax.remove()
    gs = fig.add_gridspec(1, 1, left=pos.x0, right=pos.x1, bottom=pos.y0, top=pos.y1)
    cell = gs[0, 0]
    _plot_two_layer_metal_on_fig(
        fig, cell, upper_metal, feed, title, fontsize=fontsize, cfg=cfg
    )


def _crop_field_to_metal_plane(e: np.ndarray) -> np.ndarray:
    """(NX,NY)=[x,y] |E| → 51×51 [y,x]. [row=y, col=x], same as imshow(origin=lower) and contour(metal.T).

    There used to be no .T here; it cancelled out a bug where the mask reached the solver transposed, so plots looked right.
    With the transpose bug fixed, .T is needed here so the copper outline and field line up.
    """
    e = np.asarray(e, dtype=np.float64)
    p = int(FDTD_PAD)
    if e.shape[0] <= p + PIX or e.shape[1] <= p + PIX:
        return np.asarray(e, dtype=np.float64).T
    return e[p : p + PIX, p : p + PIX].T


def _field_sig(ind: Individual) -> bytes | None:
    if ind.air is None:
        return None
    fe = getattr(ind, "field_extra", None)
    if not isinstance(fe, dict) or fe.get("mid") is None:
        return None
    h = hashlib.sha256()
    h.update(np.asarray(ind.air, dtype=np.float32).tobytes())
    h.update(np.asarray(fe["mid"], dtype=np.float32).tobytes())
    return h.digest()[:16]


def _plane_overlay_corr(metal: np.ndarray, plane: np.ndarray) -> tuple[float, float]:
    m = np.asarray(metal, dtype=np.float64)
    p = np.asarray(plane, dtype=np.float64)
    if m.size != p.size:
        return float("nan"), float("nan")

    def _c(a: np.ndarray, b: np.ndarray) -> float:
        a = a.ravel()
        b = b.ravel()
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return _c(m.T, p), _c(m.T, p.T)


def _eval_fields_synced(ind: Individual) -> bool:
    """Block cases where eval_ck matches but |E| and S11 are from old copper (rectangular field in Evolution JPG)."""
    ck = _metal_cache_key(ind)
    if ind.eval_ck != ck or ind.s11 is None or ind.air is None:
        return False
    sig = _field_sig(ind)
    if sig is None or _METAL_FIELD_SIG.get(ck) != sig:
        return False
    cu = int(np.asarray(ind.metal).sum())
    sc = float(ind.score) if ind.score is not None else 0.0
    if cu < 30 and sc > 9.0:
        return False
    if cu < 80:
        return True
    fe = getattr(ind, "field_extra", None)
    if not isinstance(fe, dict) or fe.get("mid") is None:
        return False
    plane = _crop_field_to_metal_plane(fe["mid"])
    aligned, swapped = _plane_overlay_corr(ind.metal, plane)
    if not np.isfinite(aligned):
        return False
    if np.isfinite(swapped) and aligned + 1e-6 < swapped:
        return False
    return aligned >= 0.18


def _plot_xy_e(
    ax,
    e: np.ndarray,
    metal,
    title: str,
    *,
    scale: str = "auto",
    peak_frac: float = 0.3,
) -> None:
    plane = _crop_field_to_metal_plane(e)
    ext = list(METAL_XY_EXT)
    _emin, emax, vlo, vhi = _field_limits(plane)
    if scale == "peak_frac" and emax > 0:
        vhi = max(vlo + 1e-30, peak_frac * emax)
    elif scale == "log":
        plane = np.log10(np.maximum(plane, 0.0) + 1e-3)
        _emin, emax, vlo, vhi = _field_limits(plane)
    ax.imshow(
        plane,
        origin="lower",
        cmap="inferno",
        extent=ext,
        aspect="equal",
        vmin=vlo,
        vmax=vhi,
    )
    ax.contour(
        metal.T,
        levels=[0.5],
        extent=ext,
        colors="w",
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")


def _plot_xz_e(
    ax,
    c: np.ndarray,
    metal,
    feed,
    title: str,
    faces=None,
    *,
    scale: str = "auto",
    peak_frac: float = 0.3,
    draw_cone: bool = True,
    feed_mark: bool = True,
) -> None:
    c = np.asarray(c, dtype=np.float64)
    nx, nz = c.shape
    dx, dz = 1.0, 0.375
    k_gnd, k_patch = _K_GND, _K_PATCH
    x0, x1 = -nx * dx / 2.0, nx * dx / 2.0
    z0, z1 = 0.0, nz * dz
    _emin, emax, vlo, vhi = _field_limits(c)
    disp = c
    if scale == "peak_frac" and emax > 0:
        vhi = max(vlo + 1e-30, peak_frac * emax)
    elif scale == "log":
        disp = np.log10(np.maximum(c, 0.0) + 1e-3)
        _emin, emax, vlo, vhi = _field_limits(disp)
    ax.imshow(
        disp.T,
        origin="lower",
        cmap="inferno",
        extent=[x0, x1, z0, z1],
        aspect="auto",
        vmin=vlo,
        vmax=vhi,
    )
    ax.axhline(k_gnd * dz, color="#8ec8ff", lw=0.7)
    ax.axhline(k_patch * dz, color="w", lw=0.7)
    if feed_mark:
        fy = int(feed[1])
        if 0 <= fy < metal.shape[1]:
            xs = (np.arange(metal.shape[0]) - CENTER) * dx
            on = np.asarray(metal[:, fy]) >= 0.5
            ax.plot(xs[on], np.full(int(np.sum(on)), k_patch * dz), "w.", ms=2)
    if draw_cone and faces is not None:
        _draw_cone_xz(ax, faces, z_patch=k_patch * dz)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")


def save_plot(
    out: Path,
    tag: str,
    metal,
    s11,
    freqs,
    score,
    worst,
    feed,
    air=None,
    cfg=None,
    cut=None,
    faces=None,
    field_extra=None,
):
    out.mkdir(parents=True, exist_ok=True)
    copper = int(np.asarray(metal).sum())
    sm = str(getattr(cfg, "search_mode", "metal")).lower() if cfg is not None else "metal"
    stack_layers = sm in ("metal", "feed")
    fe = field_extra if isinstance(field_extra, dict) else None
    rich = fe is not None and fe.get("patch") is not None and air is not None
    d_db, beam, peak, r_rms = 0.0, 0.0, None, 0.0
    eta = 0.0
    eta_min = 0.0
    if cfg is not None and s11 is not None:
        eta = match_efficiency(s11, freqs, cfg)
        eta_min = match_efficiency_min(s11, freqs, cfg)
    f0 = cfg_f0_hz(cfg) / 1e9 if cfg is not None else 6.0

    if rich:
        fig = plt.figure(figsize=(14.4, 12.0))
        gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.32)
        ax_air = fig.add_subplot(gs[0, 1])
        ax_patch = fig.add_subplot(gs[0, 2])
        ax_mid = fig.add_subplot(gs[1, 0])
        ax_ymax = fig.add_subplot(gs[1, 1])
        ax_top = fig.add_subplot(gs[1, 2])
        ax_feed = fig.add_subplot(gs[2, 0])
        ax_s11 = fig.add_subplot(gs[2, 1:])
        metal_title = f"{tag}  copper={copper}px"
        if stack_layers:
            _plot_two_layer_metal_on_fig(
                fig, gs[0, 0], metal, feed, metal_title, fontsize=9, cfg=cfg
            )
        else:
            ax_m = fig.add_subplot(gs[0, 0])
            _plot_metal_xy_panel(ax_m, metal, feed, metal_title)
        e_air = np.asarray(air, dtype=np.float64)
        d_db, beam, peak, r_rms = air_metrics(e_air)
        _plot_xy_e(ax_air, e_air, metal, f"Air xy  z={Z_AIR_MM:g} mm")
        axis = _draw_cone_xy(ax_air, faces, z_mm=Z_FACE_MM)
        frac = _cone_frac(faces)
        frac_s = f" in {100 * frac:.0f}%" if np.isfinite(frac) else ""
        ax_air.set_title(f"Air xy z={Z_AIR_MM:g}  cone{axis}60°{frac_s}", fontsize=9)
        _plot_xy_e(ax_patch, fe["patch"], metal, f"Patch plane xy  z={Z_PATCH_MM:g} mm")
        _plot_xy_e(ax_mid, fe["mid"], metal, f"Substrate mid xy  z={Z_MID_MM:g} mm")
        _plot_xz_e(
            ax_ymax,
            fe["cut_max"],
            metal,
            feed,
            "xz  max_y |E|  log10",
            faces=faces,
            scale="log",
            draw_cone=True,
            feed_mark=False,
        )
        _plot_xy_e(ax_top, fe["top"], metal, f"+z box face xy  z={Z_TOP_MM:g} mm")
        if cut is not None:
            _plot_xz_e(
                ax_feed,
                cut,
                metal,
                feed,
                "xz  feed y  vmax=0.3·peak",
                faces=faces,
                scale="peak_frac",
                peak_frac=0.3,
                draw_cone=True,
                feed_mark=True,
            )
        else:
            ax_feed.set_visible(False)
        ax2 = ax_s11
    else:
        has_air = air is not None
        has_cut = cut is not None
        if has_air and has_cut:
            fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.4))
            ax1, ax_e = axes[0]
            ax_cut, ax2 = axes[1]
        elif has_air:
            fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
            ax1, ax_e, ax2 = axes
            ax_cut = None
        else:
            fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))
            ax1, ax2 = axes
            ax_e = ax_cut = None
        metal_title = f"{tag}  copper={copper}px"
        if stack_layers:
            fig.subplots_adjust(left=0.06, right=0.48, top=0.95, bottom=0.08)
            gs_m = fig.add_gridspec(1, 1, left=0.05, right=0.48, top=0.92, bottom=0.12)
            _plot_two_layer_metal_on_fig(
                fig, gs_m[0, 0], metal, feed, metal_title, fontsize=10, cfg=cfg
            )
        else:
            _plot_metal_xy_panel(ax1, metal, feed, metal_title, fontsize=10)
        if ax_e is not None:
            e = np.asarray(air, dtype=np.float64)
            plane = _crop_field_to_metal_plane(e)
            ext = list(METAL_XY_EXT)
            emin, emax, vlo, vhi = _field_limits(plane)
            im = ax_e.imshow(
                plane,
                origin="lower",
                cmap="inferno",
                extent=ext,
                aspect="equal",
                vmin=vlo,
                vmax=vhi,
            )
            ax_e.contour(metal.T, levels=[0.5], extent=ext, colors="w", linewidths=0.6)
            if emax > 0:
                xs = np.linspace(ext[0], ext[1], plane.shape[1])
                ys = np.linspace(ext[2], ext[3], plane.shape[0])
                ax_e.contour(
                    xs, ys, plane, levels=[0.5 * emax], colors="#7CFF6B", linewidths=0.9
                )
            d_db, beam, peak, r_rms = air_metrics(e)
            axis = _draw_cone_xy(ax_e, faces, z_mm=Z_FACE_MM)
            frac = _cone_frac(faces)
            frac_s = f" in {100 * frac:.0f}%" if np.isfinite(frac) else ""
            ax_e.set_title(f"Top cut |E|  cone{axis} full 60°{frac_s}  (xy)")
            ax_e.set_xlabel("x (mm)")
            ax_e.set_ylabel("y (mm)")
            fig.colorbar(im, ax=ax_e, fraction=0.046, pad=0.04).set_label("|E|")
        elif air is not None:
            d_db, beam, peak, r_rms = air_metrics(air)
        if ax_cut is not None:
            c = np.asarray(cut, dtype=np.float64)
            nx, nz = c.shape
            dx, dz = 1.0, 0.375
            k_gnd, k_patch = _K_GND, _K_PATCH
            x0, x1 = -nx * dx / 2.0, nx * dx / 2.0
            z0, z1 = 0.0, nz * dz
            _emin, emax, vlo, vhi = _field_limits(c)
            imc = ax_cut.imshow(
                c.T,
                origin="lower",
                cmap="inferno",
                extent=[x0, x1, z0, z1],
                aspect="auto",
                vmin=vlo,
                vmax=vhi,
            )
            ax_cut.axhline(k_gnd * dz, color="#8ec8ff", lw=0.8)
            ax_cut.axhline(k_patch * dz, color="w", lw=0.8)
            fy = int(feed[1])
            if 0 <= fy < metal.shape[1]:
                xs = (np.arange(metal.shape[0]) - CENTER) * dx
                on = np.asarray(metal[:, fy]) >= 0.5
                ax_cut.plot(xs[on], np.full(int(np.sum(on)), k_patch * dz), "w.", ms=3)
            if emax > 0:
                xs = np.linspace(x0, x1, nx)
                zs = np.linspace(z0, z1, nz)
                ax_cut.contour(xs, zs, c.T, levels=[0.5 * emax], colors="#7CFF6B", linewidths=0.9)
            axis = _draw_cone_xz(ax_cut, faces, z_patch=k_patch * dz)
            frac = _cone_frac(faces)
            frac_s = f" in {100 * frac:.0f}%" if np.isfinite(frac) else ""
            ax_cut.set_title(f"Front |E|  cone{axis} full 60°{frac_s}  (xz)")
            ax_cut.set_xlabel("x (mm)")
            ax_cut.set_ylabel("z (mm)")
            fig.colorbar(imc, ax=ax_cut, fraction=0.046, pad=0.04).set_label("|E|")
    ax2.plot(np.asarray(freqs) / 1e9, s11, "b-")
    tgt_db = float(getattr(cfg, "s11_target_db", -10.0)) if cfg is not None else -10.0
    ax2.axhline(tgt_db, color="r", ls="--", label=f"target {tgt_db:g} dB")
    ax2.axvline(f0, color="gray", ls=":")
    if cfg is not None:
        b0, b1 = getattr(cfg, "band_hz", (7e9, 7.5e9))
        ax2.axvspan(b0 / 1e9, b1 / 1e9, color="#446688", alpha=0.12, label="score band")
    ax2.set_ylim([-40, 0])
    ax2.grid(True, ls=":")
    band_note = ""
    match_pct = ""
    if cfg is not None:
        b0, b1 = getattr(cfg, "band_hz", (3e9, 9e9))
        band_note = f"  band={b0/1e9:g}–{b1/1e9:g} GHz"
        tgt_db2 = float(getattr(cfg, "s11_target_db", -10.0))
        bm = (freqs >= b0) & (freqs <= b1)
        if np.any(bm):
            mf = float(np.mean(np.asarray(s11)[bm] < tgt_db2 + 1e-9))
            match_pct = f"  match={100*mf:.0f}%"
    ax2.set_title(
        f"S11  worst={worst:.1f}dB  η_avg={100 * eta:.0f}%  η_min={100 * eta_min:.0f}%"
        f"{match_pct}{band_note}"
    )
    ax2.set_xlabel("GHz")
    ax2.set_ylabel("S11 (dB)")
    if cfg is not None:
        band = getattr(cfg, "band_hz", (7e9, 7.5e9))
        wc = float(getattr(cfg, "w_cone", getattr(cfg, "w_directivity", 1.0)))
        we = float(getattr(cfg, "w_efficiency", 1.0))
        wl = float(getattr(cfg, "w_leave", 1.0))
        cap = float(getattr(cfg, "leave_db_max", 18.0))
        mix = float(getattr(cfg, "score_geom_mix", 0.0))
        fitness = getattr(cfg, "fitness", "gain")
        peak_s = f"({peak[0]:.1f},{peak[1]:.1f})mm" if peak is not None else "-"
        frac = _cone_frac(faces)
        frac_s = f"{100 * frac:.0f}%" if np.isfinite(frac) else "-"
        tns = float(getattr(cfg, "fdtd_t_sec", 3.5e-9)) * 1e9
        mix_s = f"  mix={mix:g}" if mix > 0 else ""
        foot = (
            f"score={score:.2f} = {wc:g}·cone60° + {we:g}·η_dB + {wl:g}·out/in(≤{cap:g}dB){mix_s}  "
            f"FITNESS={fitness}  t={tns:g}ns  "
            f"cone=full 60°({FACE_NAME[_best_face(faces)]}) in {frac_s}  "
            f"|E|peak={peak_s}  "
            f"η_avg={100 * eta:.0f}%  η_min={100 * eta_min:.0f}%  S11_worst={worst:.1f}dB  copper={copper}  "
            f"band={band[0]/1e9:.2f}-{band[1]/1e9:.2f}GHz  f0={f0:.2f}GHz"
        )
        fig.text(0.5, 0.012, foot, ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(out / f"{tag}.jpg", dpi=130)
    plt.close(fig)


def save_cuda_field_npz(
    cfg,
    tag: str,
    ind: Individual,
    freqs,
    *,
    suffix: str = "",
    meta: dict | None = None,
) -> Path:
    """Save |E| and S11 read from CUDA FDTD to npz (for verification separate from the JPG)."""
    root = Path(getattr(cfg, "byproduct_dir", Path("run_byproduct"))) / "cuda_fields"
    root.mkdir(parents=True, exist_ok=True)
    name = f"{tag}{suffix}.npz"
    payload: dict[str, np.ndarray] = {
        "metal": np.asarray(ind.metal, dtype=np.uint8),
    }
    if ind.score is not None:
        payload["score"] = np.asarray(float(ind.score), dtype=np.float64)
    if ind.s11 is not None:
        payload["s11"] = np.asarray(ind.s11, dtype=np.float64)
    if ind.air is not None:
        payload["air"] = np.asarray(ind.air, dtype=np.float32)
        payload["air_cropped"] = np.asarray(
            _crop_field_to_metal_plane(ind.air), dtype=np.float32
        )
    if ind.cut is not None:
        payload["cut"] = np.asarray(ind.cut, dtype=np.float32)
    fe = ind.field_extra if isinstance(getattr(ind, "field_extra", None), dict) else {}
    for key in ("patch", "mid", "top", "cut_max"):
        if key not in fe or fe[key] is None:
            continue
        payload[key] = np.asarray(fe[key], dtype=np.float32)
        if key in ("patch", "mid", "top"):
            payload[f"{key}_cropped"] = np.asarray(
                _crop_field_to_metal_plane(fe[key]), dtype=np.float32
            )
    if freqs is not None:
        payload["freqs_hz"] = np.asarray(freqs, dtype=np.float64)
    ck = _metal_cache_key(ind)
    payload["eval_ck_hash"] = np.frombuffer(ck[0][:8], dtype=np.uint8)
    if meta:
        for mk, mv in meta.items():
            payload[f"meta_{mk}"] = np.asarray(str(mv))
    path = root / name
    np.savez_compressed(path, **payload)
    return path


def save_best(cfg, ind: Individual, freqs, r: float, tag: str | None = None):
    """Exploration stage. Saved as Best_r012.jpg, Best_p20.jpg, etc."""
    if ind is None or ind.s11 is None:
        return
    if not is_feasible(ind, cfg):
        return
    ck = _metal_cache_key(ind)
    synced = _eval_fields_synced(ind)
    if ind.air is None or ind.eval_ck != ck:
        if _LIVE is not None:
            _LIVE.persist(
                f"  skip {tag or 'Best'}: no eval or metal fingerprint mismatch"
            )
        return
    if not synced and _LIVE is not None:
        _LIVE.persist(
            f"  Best {tag or ''}: field consistency check failed, S11 and |E| still saved"
        )
    if not tag:
        tag = f"Best_r{int(round(float(r))):03d}"
    by = Path(getattr(cfg, "byproduct_dir", Path("run_byproduct"))) / "metal"
    by.mkdir(parents=True, exist_ok=True)
    np.save(by / f"{tag}.npy", np.asarray(ind.metal, dtype=np.uint8))
    save_plot(
        cfg.out_dir,
        tag,
        ind.metal,
        ind.s11,
        freqs,
        ind.score,
        ind.worst,
        effective_port(ind, cfg.feed),
        air=ind.air,
        cut=ind.cut,
        cfg=cfg,
        faces=getattr(ind, "faces", None),
        field_extra=getattr(ind, "field_extra", None),
    )


def ensure_evaluated(
    solver,
    ind: Individual,
    freqs,
    cfg,
    model: OffspringModel | None = None,
    note: str = "sync",
) -> Individual:
    """Check metal matches FDTD |E| and S11 before saving the JPG."""
    ck = _metal_cache_key(ind)
    if (
        ind.eval_ck == ck
        and ind.s11 is not None
        and ind.air is not None
        and getattr(ind, "field_extra", None) is not None
        and _eval_fields_synced(ind)
    ):
        return ind
    return fill_eval(
        solver, ind, freqs, cfg, model=model, elite=None, note=note, log_chart=False
    )


def save_evolution(
    cfg,
    ind: Individual,
    freqs,
    gen: int | None = None,
    solver=None,
    model: OffspringModel | None = None,
):
    """Evolution stage. Evolution_gen001.jpg, Evolution_final.jpg."""
    if ind is None:
        return
    if solver is not None:
        ind = ensure_evaluated(
            solver,
            ind,
            freqs,
            cfg,
            model=model,
            note=f"Evolution gen{gen or 'final'}",
        )
    if ind.s11 is None:
        return
    ck = _metal_cache_key(ind)
    if ind.eval_ck != ck or ind.air is None or not _eval_fields_synced(ind):
        if _LIVE is not None:
            _LIVE.persist(
                f"  skip Evolution gen{gen}: copper/field mismatch (copper={int(ind.metal.sum())})"
            )
        return
    tag = "Evolution_final" if gen is None else f"Evolution_gen{int(gen):03d}"
    save_plot(
        cfg.out_dir,
        tag,
        ind.metal,
        ind.s11,
        freqs,
        ind.score,
        ind.worst,
        effective_port(ind, cfg.feed),
        air=ind.air,
        cut=ind.cut,
        cfg=cfg,
        faces=getattr(ind, "faces", None),
        field_extra=getattr(ind, "field_extra", None),
    )


def dump_byproduct(cfg, ind: Individual, freqs, gen: int, batch: int, idx: int, kept: int, kind: str):
    if not getattr(cfg, "save_byproduct", False):
        return
    root = Path(getattr(cfg, "byproduct_dir", Path("run_byproduct")))
    metal_dir = root / "metal"
    metal_dir.mkdir(parents=True, exist_ok=True)
    d_db, beam, peak, r_rms = air_metrics(ind.air)
    eta = match_efficiency(ind.s11, freqs, cfg) if ind.s11 is not None else 0.0
    px = py = ""
    if peak is not None:
        px, py = f"{peak[0]:.2f}", f"{peak[1]:.2f}"
    name = f"g{gen:03d}_b{batch:03d}_{kind}{idx:03d}"
    npy_p = metal_dir / f"{name}.npy"
    png_p = metal_dir / f"{name}.png"
    np.save(npy_p, np.asarray(ind.metal, dtype=np.uint8))
    port = effective_port(ind, cfg.feed)
    sm = str(getattr(cfg, "search_mode", "metal")).lower()
    title = f"{name}  sc={ind.score:.2f}  Cu={int(ind.metal.sum())}  kept={kept}"
    fig = plt.figure(figsize=(3.2, 4.6))
    gs = fig.add_gridspec(1, 1)
    if sm in ("metal", "feed"):
        _plot_two_layer_metal_on_fig(fig, gs[0, 0], ind.metal, port, title, fontsize=7, cfg=cfg)
    else:
        ax = fig.add_subplot(gs[0, 0])
        _plot_metal_xy_panel(ax, ind.metal, port, title, fontsize=7)
    fig.tight_layout(pad=0.15)
    fig.savefig(png_p, dpi=90)
    plt.close(fig)
    csv_p = root / "log.csv"
    new = not csv_p.exists()
    with csv_p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                [
                    "gen",
                    "batch",
                    "idx",
                    "kind",
                    "kept",
                    "score",
                    "d_db",
                    "beam",
                    "eta",
                    "s11_worst",
                    "copper",
                    "peak_x",
                    "peak_y",
                    "npy",
                    "png",
                ]
            )
        w.writerow(
            [
                gen,
                batch,
                idx,
                kind,
                kept,
                f"{ind.score:.6f}" if ind.score is not None else "",
                f"{d_db:.4f}",
                f"{beam:.4f}",
                f"{eta:.4f}",
                f"{ind.worst:.4f}" if ind.worst is not None else "",
                int(ind.metal.sum()),
                px,
                py,
                str(npy_p.as_posix()),
                str(png_p.as_posix()),
            ]
        )


_METAL_EVAL_CACHE: dict[tuple[bytes, int], Individual] = {}
_METAL_FIELD_SIG: dict[tuple[bytes, int], bytes] = {}


def reset_eval_cache() -> None:
    """Clear the copper↔|E| memory cache (at process start or rerun)."""
    _METAL_EVAL_CACHE.clear()
    _METAL_FIELD_SIG.clear()


def _metal_cache_key(ind: Individual) -> tuple:
    p = getattr(ind, "port_ij", None)
    port = (-1, -1) if p is None else (int(p[0]), int(p[1]))
    return (np.asarray(ind.metal, dtype=np.uint8).tobytes(), int(ind.mode), port)


def _copy_eval_fields(src: Individual, dst: Individual) -> None:
    dst.score = src.score
    dst.worst = src.worst
    dst.s11 = None if src.s11 is None else np.array(src.s11, copy=True)
    dst.air = None if src.air is None else np.array(src.air, copy=True)
    dst.cut = None if src.cut is None else np.array(src.cut, copy=True)
    dst.d_use = src.d_use
    dst.eta = src.eta
    dst.eta_min = getattr(src, "eta_min", None)
    dst.d_ap = getattr(src, "d_ap", None)
    dst.d_peak = getattr(src, "d_peak", None)
    dst.beam_db = getattr(src, "beam_db", None)
    dst.leave = getattr(src, "leave", None)
    dst.leave_db = getattr(src, "leave_db", None)
    dst.faces = (
        None
        if getattr(src, "faces", None) is None
        else np.array(src.faces, copy=True)
    )
    src_fe = getattr(src, "field_extra", None)
    dst.field_extra = (
        None
        if src_fe is None
        else {k: np.array(v, copy=True) for k, v in src_fe.items()}
    )
    dst.eval_ck = getattr(src, "eval_ck", None)


def evaluate(solver, metal, freqs, cfg):
    if hasattr(solver, "set_sim"):
        solver.set_sim(
            float(getattr(cfg, "fdtd_t_sec", 3.5e-9)),
            float(getattr(cfg, "fdtd_cfl", 0.99)),
            cfg_f0_hz(cfg),
            cfg_excitation_fc_hz(cfg),
        )
    if _LIVE is not None:
        with _mute_fd2():
            s11, mindb, minhz, nused = solver.run(metal, freqs)
    else:
        s11, mindb, minhz, nused = solver.run(metal, freqs)
    air = solver.air_e() if hasattr(solver, "air_e") else None
    cut = solver.cut_e() if hasattr(solver, "cut_e") else None
    faces = solver.faces() if hasattr(solver, "faces") else None
    leave = solver.leave() if hasattr(solver, "leave") else None
    field_extra = solver.extra_fields() if hasattr(solver, "extra_fields") else None
    sc, worst = score_s11(s11, freqs, cfg, air=air, metal=metal, faces=faces, leave=leave)
    return sc, worst, s11, mindb, minhz, nused, air, cut, faces, leave, field_extra


def fill_eval(
    solver,
    ind: Individual,
    freqs,
    cfg,
    model: OffspringModel | None = None,
    elite: Individual | None = None,
    note: str = "",
    log_chart: bool = True,
) -> Individual:
    if bool(getattr(cfg, "connect_repair", False)) and getattr(cfg, "search_mode", "metal") == "metal":
        # Before every FDTD eval: remove upper-layer copper not 4-connected to the feed (incl. init, RL and diffusion paths)
        from circle.diagonal import strip_diagonal_contacts

        port0 = effective_port(ind, cfg.feed)
        g = np.asarray(ind.metal, dtype=np.uint8).copy()
        g[int(port0[0]), int(port0[1])] = 1
        g, _ = strip_diagonal_contacts(g, (int(port0[0]), int(port0[1])))
        g = keep_connected(g, (int(port0[0]), int(port0[1]))).astype(np.uint8)
        if not np.array_equal(g, ind.metal):
            ind.metal = g
    ck = _metal_cache_key(ind)
    if ind.eval_ck != ck or not _eval_fields_synced(ind):
        ind.score = None
        ind.s11 = None
        ind.worst = None
        ind.air = None
        ind.cut = None
        ind.field_extra = None
        ind.eval_ck = None
    if ind.score is not None and ind.s11 is not None and ind.eval_ck == ck:
        return ind
    cached = _METAL_EVAL_CACHE.get(ck)
    if cached is not None and cached.s11 is not None:
        _copy_eval_fields(cached, ind)
        if ind.eval_ck != ck:
            ind.eval_ck = ck
        ind.feasible = is_feasible(ind, cfg)
        if model is not None and log_chart and ind.feasible:
            model.observe_eval(
                ind.metal,
                float(ind.score or 0.0),
                d_use=ind.d_use,
                eta=ind.eta,
                d_ap=getattr(ind, "d_ap", None),
                d_peak=getattr(ind, "d_peak", None),
                beam_db=getattr(ind, "beam_db", None),
                leave=getattr(ind, "leave_db", None),
                elite=elite,
                log_chart=True,
                chart_uv=getattr(ind, "chart_uv", None),
                kind=getattr(ind, "chart_kind", "seed"),
                root=int(getattr(ind, "chart_root", 0) or 0),
            )
        return ind
    port = effective_port(ind, cfg.feed)
    if hasattr(solver, "set_feed"):
        solver.set_feed(int(port[0]), int(port[1]))
    sc, worst, s11, mindb, minhz, nused, air, cut, faces, leave, field_extra = evaluate(
        solver, ind.metal, freqs, cfg
    )
    ind.score = sc
    ind.worst = worst
    ind.s11 = s11
    ind.air = air
    ind.cut = cut
    ind.field_extra = field_extra
    cone_db, eta, _eta_db, beam, r_rms, _cu, d_ap, beam_db, face6_db, leave_db, leave_r, eta_min = gain_parts(
        s11, freqs, cfg, air=air, metal=ind.metal, faces=faces, leave=leave
    )
    ind.d_use = cone_db
    ind.d_bias = face6_db
    ind.eta = eta
    ind.eta_min = eta_min
    ind.d_ap = d_ap
    ind.d_peak = leave_db
    ind.beam_db = beam_db
    ind.leave = leave_r
    ind.leave_db = leave_db
    ind.faces = None if faces is None else np.asarray(faces, dtype=np.float64)
    ind.feasible = is_feasible(ind, cfg)
    ind.eval_ck = ck
    if model is not None and log_chart and ind.feasible:
        model.observe_eval(
            ind.metal,
            sc,
            d_use=cone_db,
            eta=eta,
            d_ap=d_ap,
            d_peak=leave_db,
            beam_db=beam_db,
            leave=leave_db,
            elite=elite,
            log_chart=True,
            chart_uv=getattr(ind, "chart_uv", None),
            kind=getattr(ind, "chart_kind", "seed"),
            root=int(getattr(ind, "chart_root", 0) or 0),
        )
        if elite is not None and elite.score is not None:
            model.observe_move(elite.metal, ind.metal, sc - float(elite.score))
    cap = float(getattr(cfg, "leave_db_max", 18.0))
    leave_s = f"{leave_db:.2f}dB"
    if leave_db > cap + 0.05:
        leave_s = f"{leave_db:.2f}dB(score≤{cap:g})"
    tag = f"{note}  " if note else ""
    line = (
        f"{tag}nused={nused}  score={sc:.2f}  "
        f"cone60={beam_db:.2f}  6face={face6_db:.2f}  η={100 * eta:.0f}%  total eff={leave_s}  "
        f"S11_worst={worst:.1f}dB  "
        f"min={mindb:.1f}dB@{minhz / 1e9:.2f}GHz  copper={int(ind.metal.sum())}"
    )
    if _LIVE is not None:
        _LIVE.push(line)
        _LIVE.bump_eval()
    else:
        print(line)
    src_fe = ind.field_extra
    snap = Individual(
        metal=ind.metal.copy(),
        mode=ind.mode,
        port_ij=getattr(ind, "port_ij", None),
        score=ind.score,
        s11=None if ind.s11 is None else np.array(ind.s11, copy=True),
        air=None if ind.air is None else np.array(ind.air, copy=True),
        cut=None if ind.cut is None else np.array(ind.cut, copy=True),
        worst=ind.worst,
        d_use=ind.d_use,
        eta=ind.eta,
        d_ap=ind.d_ap,
        d_peak=ind.d_peak,
        beam_db=ind.beam_db,
        leave=ind.leave,
        leave_db=ind.leave_db,
        faces=(
            None
            if getattr(ind, "faces", None) is None
            else np.array(ind.faces, copy=True)
        ),
        field_extra=(
            None
            if src_fe is None
            else {k: np.array(v, copy=True) for k, v in src_fe.items()}
        ),
        eval_ck=ck,
    )
    _METAL_EVAL_CACHE[ck] = snap
    fs = _field_sig(ind)
    if fs is not None:
        _METAL_FIELD_SIG[ck] = fs
    return ind


def _boundary_sites(metal: np.ndarray, feed) -> np.ndarray:
    m = metal.astype(np.uint8)
    pad = np.pad(m, 1, mode="constant")
    nbr = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    add = (m == 0) & (nbr > 0)
    rem = (m == 1) & (nbr < 4)
    rem[feed] = False
    ys, xs = np.nonzero(add | rem)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([ys, xs], axis=1)


def local_climb(
    solver,
    ind: Individual,
    freqs,
    cfg,
    rng,
    model: OffspringModel | None = None,
    elite: Individual | None = None,
) -> Individual:
    """Flip one edge cell; keep if the score rises. Runs to the basin's local max."""
    fill_eval(solver, ind, freqs, cfg, model=model, elite=elite, note="raw", log_chart=True)
    feed = cfg.feed
    max_steps = int(getattr(cfg, "local_climb_steps", 8))
    n_try = int(getattr(cfg, "local_try_per_step", 4))
    start = ind.score or 0.0
    used = 0
    for _ in range(max_steps):
        sites = _boundary_sites(ind.metal, feed)
        if len(sites) == 0:
            break
        rng.shuffle(sites)
        moved = False
        for i, j in sites[: max(1, n_try)]:
            g = ind.metal.copy()
            g[int(i), int(j)] = 1 - g[int(i), int(j)]
            cand = _repair_cfg(Individual(g, ind.mode, src=ind.src), feed, cfg)
            if np.array_equal(cand.metal, ind.metal):
                continue
            fill_eval(
                solver, cand, freqs, cfg, model=model, elite=elite, note="try", log_chart=False
            )
            used += 1
            if (cand.score or -1e18) > (ind.score or -1e18):
                ds = float(cand.score - ind.score)
                if model is not None:
                    model.observe_step(ind.metal, cand.metal, ds)
                cand.chart_kind = getattr(ind, "chart_kind", "seed")
                cand.chart_root = int(getattr(ind, "chart_root", 0) or 0)
                cand.chart_uv = getattr(ind, "chart_uv", None)
                ind = cand
                moved = True
                break
        if not moved:
            break
    if _LIVE is not None:
        _LIVE.push(
            f"climb {start:.2f}→{ind.score:.2f}  steps={used}  copper={int(ind.metal.sum())}"
        )
    else:
        print(
            f"      climb {start:.2f}→{ind.score:.2f}  steps={used}  "
            f"copper={int(ind.metal.sum())}"
        )
    # chart_uv = FDTD sample grid cell. Not converted mask→index (project) after climb.
    return ind


def eval_to_peak(
    solver,
    ind: Individual,
    freqs,
    cfg,
    rng,
    model: OffspringModel | None = None,
    elite: Individual | None = None,
) -> Individual:
    # No pixel climb. Chart (u,v)±1 8-neighbors are handled in warmup.verify_peak_uv8.
    return fill_eval(solver, ind, freqs, cfg, model=model, elite=elite, note="eval")


def spawn_children(
    parents: list[Individual],
    rng,
    cfg,
    feed,
    n: int,
    model: OffspringModel | None = None,
    elite: Individual | None = None,
    gen: int = 1,
    gens: int = 10,
) -> list[Individual]:
    scored = [p for p in parents if p.score is not None] or parents
    p_rand = float(getattr(cfg, "random_start", 1.0))
    p_rand *= 1.0 - (max(int(gen), 1) - 1) / max(int(gens), 1)
    p_rand = float(np.clip(p_rand, 0.0, 1.0))

    min_h = max(int(getattr(cfg, "circle_xover_dr", 16.0)), 40)

    use_deficit = bool(getattr(cfg, "ga_breed_deficit", True))

    def pick_pair() -> tuple[Individual, Individual]:
        if use_deficit and elite is not None:
            pair = pick_breed_pair(scored, rng, elite=elite, min_hamming=min_h)
        else:
            pair = pick_metric_pair(scored, rng, min_hamming=min_h)
        if pair is not None:
            return pair
        a = max(scored, key=lambda x: x.score or -1e18)
        others = [p for p in scored if not np.array_equal(p.metal, a.metal)]
        if not others:
            return a, a
        return a, max(others, key=lambda p: hamming(p.metal, a.metal))

    feed_only = str(getattr(cfg, "search_mode", "metal")).lower() == "feed"
    kids = []
    while len(kids) < n:
        if feed_only:
            pair = pick_feed_pair(scored, rng, min_dist=2)
            if pair is None or rng.random() < p_rand * 0.5:
                sites = _feed_sites_on_patch(cfg)
                i, j = sites[int(rng.integers(0, len(sites)))]
                child = seed_fixed_patch_port(
                    i, j, float(cfg.patch_lx_mm), float(cfg.patch_ly_mm)
                )
            else:
                pa, pb = pair
                child = crossover_feed(pa, pb, rng, cfg)
                child = mutate_feed(child, rng, cfg)
        else:
            mode = int(rng.choice(getattr(cfg, "symmetry_modes", [-1])))
            if rng.random() < p_rand:
                child = seed_random(rng, feed, mode)
            else:
                pa, pb = pick_pair()
                child = crossover(pa, pb, rng, feed, cfg)
                child = mutate(
                    child,
                    rng,
                    feed,
                    cfg.mut_pixels,
                    getattr(cfg, "diffuse_steps", 2),
                    getattr(cfg, "diffuse_rate", 0.4),
                    kernel=None if model is None else model.annealed_kernel(),
                    cfg=cfg,
                )
        child.score = None
        child.s11 = None
        child.air = None
        child.cut = None
        child.field_extra = None
        child.eval_ck = None
        child.src = "ga"
        kids.append(child)
    n_rl = int(getattr(cfg, "extra_rl", 0))
    n_df = int(getattr(cfg, "extra_diffuse", 0))
    if not feed_only and model is not None and n_rl > 0:
        mode = elite.mode if elite is not None else -1
        kids.extend(model.sample_rl(n_rl, rng, feed, mode, parent=elite))
    if not feed_only and model is not None and elite is not None and n_df > 0:
        kids.extend(model.sample_diffuse_b(n_df, elite, rng, feed))
    return kids


def _progress_bar(
    frac: float,
    gen: int,
    total: int,
    extra: str = "",
    circle_r=None,
    circle_rmax=None,
    circle_r_full=None,
    stall=None,
) -> None:
    if _LIVE is not None:
        _LIVE.set_bar(
            frac,
            gen,
            total,
            extra,
            circle_r=circle_r,
            circle_rmax=circle_rmax,
            circle_r_full=circle_r_full,
            stall=stall,
        )
        return
    width = 28
    total = max(int(total), 1)
    frac = min(max(float(frac), 0.0), 1.0)
    filled = int(round(frac * width))
    bar = "#" * filled + "-" * (width - filled)
    extra = extra.strip()
    tail = f"  {extra}" if extra else ""
    if circle_r is not None:
        rmax = circle_rmax if circle_rmax is not None else 0.0
        rfull = float(circle_r_full) if circle_r_full is not None else 0.0
        if rfull > 1e-6:
            map_pct = 100.0 * min(float(circle_r) / rfull, 1.0)
            phase = f"map~{map_pct:.0f}%  r={float(circle_r):g}/{float(rmax):g}(limit)"
        else:
            phase = f"trend r={float(circle_r):g}/{float(rmax):g}"
        if stall:
            phase += f"  live {stall}"
    elif gen <= 0:
        phase = "circle"
    else:
        phase = f"gen {gen}/{total}"
    print(f"  [{bar}] {100.0 * frac:5.1f}%  {phase}{tail}", flush=True)


def run_ga(solver, cfg, rl_refine=None) -> Individual:
    rng = np.random.default_rng(cfg.seed)
    feed = cfg.feed
    freqs = np.linspace(cfg.f_start, cfg.f_stop, cfg.n_freq)
    solver.set_feed(*feed)

    pop = []
    retries = int(getattr(cfg, "offspring_retries", 40))
    extra_rl = int(getattr(cfg, "extra_rl", 0))
    extra_df = int(getattr(cfg, "extra_diffuse", 0))
    model = OffspringModel(pca_rank=int(getattr(cfg, "diffuse_rank", 8)))
    gens = int(cfg.generations)
    light_only = bool(getattr(cfg, "circle_light_only", False))
    if light_only:
        print(
            f"light-only  disk≤r={getattr(cfg, 'circle_r_disk_end', None) or 'frac'}  "
            f"step={getattr(cfg, 'circle_r_step', 4):g}  "
            f"r_max={getattr(cfg, 'circle_r_max', 24):g}  "
            f"backend={getattr(solver, 'backend', '?')}  fitness={cfg.fitness}  "
            f"no evolution"
        )
    else:
        sm = str(getattr(cfg, "search_mode", "metal")).lower()
        print(
            f"GA start  pop={cfg.population} gens={gens}  "
            f"backend={getattr(solver, 'backend', '?')}  fitness={cfg.fitness}  "
            f"search={sm}  feed={feed}  max_offspring_batches={retries}  "
            f"extra_rl={extra_rl} extra_diffuseB={extra_df}  "
            f"climb={getattr(cfg, 'local_climb_steps', 8)}x{getattr(cfg, 'local_try_per_step', 4)}  "
            f"circle_warmup={getattr(cfg, 'circle_warmup', True)}  "
            f"T fixed  align log only"
        )
    global _LIVE
    _LIVE = LivePanel()
    _LIVE.start()
    best_hist = []
    elite: Individual | None = None

    def climb(ind: Individual) -> Individual:
        return eval_to_peak(solver, ind, freqs, cfg, rng, model, elite=None)

    try:
        if getattr(cfg, "circle_warmup", True):
            from circle.warmup import run_circle_warmup

            _LIVE.persist(
                "  seed circle exploration only, no evolution"
                if light_only
                else "  seed circle exploration → evolve when trend lines end"
            )
            _progress_bar(
                0.0,
                0,
                1,
                "seed",
                circle_r=0.0,
                circle_rmax=float(getattr(cfg, "circle_r_max", 48.0)),
                stall=f"0/{int(getattr(cfg, 'circle_stagnate', 2))}",
            )

            def on_warm(frac, msg, r=None, r_max=None, r_map_full=None, stall=None):
                _progress_bar(
                    frac,
                    0,
                    1,
                    msg,
                    circle_r=r,
                    circle_rmax=r_max,
                    circle_r_full=r_map_full,
                    stall=stall,
                )

            pop = run_circle_warmup(
                cfg,
                rng,
                climb=climb,
                model=model,
                log=lambda m: _LIVE.push(m),
                progress=on_warm,
            )
            if light_only:
                if not pop:
                    _LIVE.persist("  exploration failed, no points")
                    return None
                pop.sort(
                    key=lambda x: x.score if x.score is not None else -1e18,
                    reverse=True,
                )
                elite = clone_ind(pop[0])
                best_hist.append(elite.score)
                r_now = float(getattr(model.chart, "r_probed", 0.0) or 0.0)
                save_best(cfg, elite, freqs, r_now)
                cfg.out_dir.mkdir(parents=True, exist_ok=True)
                np.savetxt(
                    cfg.out_dir / "score_history.csv",
                    np.array(best_hist, dtype=float),
                    delimiter=",",
                )
                _LIVE.persist(
                    f"  exploration done  evals={len(pop)}  best={elite.score:.2f}  "
                    f"copper={int(elite.metal.sum())}  (no evolution)"
                )
                _progress_bar(1.0, 0, 1, "light done")
                return elite
        elif str(getattr(cfg, "search_mode", "metal")).lower() == "feed":
            _LIVE.persist("  fixed patch, exhaustive FDTD over ports on patch")
            sites = _feed_sites_on_patch(cfg)
            _LIVE.persist(f"    port candidates {len(sites)} cells  patch {cfg.patch_lx_mm}×{cfg.patch_ly_mm} mm")

            def on_feed(frac, msg):
                _progress_bar(frac * 0.95, 0, 1, msg)

            pop = build_feed_population(
                solver, cfg, rng, freqs, model=model, progress=on_feed
            )
            if pop:
                best = pop[0]
                p = effective_port(best, feed)
                _LIVE.persist(
                    f"    exhaustive best score={best.score:.2f}  port={mm_from_ij(*p)}  "
                    f"S11_worst={best.worst:.1f} dB"
                )
            pop = pop[: max(cfg.population, 1)]
        else:
            pop = []
            _LIVE.persist("  gen1 initial population (random)")
            while len(pop) < cfg.population:
                mode = int(rng.choice(cfg.symmetry_modes))
                pop.append(seed_random(rng, feed, mode))
        _progress_bar(0.0, 1, gens)
        n_pop = len(pop)
        for n, ind in enumerate(pop):
            if ind.score is None or ind.s11 is None:
                pop[n] = eval_to_peak(solver, ind, freqs, cfg, rng, model)
            _progress_bar((n + 1) / n_pop / gens, 1, gens, f"ind {n+1}/{n_pop}")
        pop.sort(key=lambda x: x.score if x.score is not None else -1e18, reverse=True)
        elite = clone_ind(pop[0])
        for ind in pop:
            resid = (ind.metal.astype(np.float64) - elite.metal.astype(np.float64)).ravel()
            model.b.observe(resid, 1.0)
        model.b.refit()
        for n, ind in enumerate(pop):
            dump_byproduct(cfg, ind, freqs, gen=1, batch=0, idx=n + 1, kept=int(n == 0), kind="init")
        best_hist.append(elite.score)
        save_evolution(cfg, elite, freqs, gen=1, solver=solver, model=model)
        _LIVE.persist(
            f"  ★ best score={elite.score:.2f}  "
            f"S11_worst={elite.worst:.1f} dB  copper={int(elite.metal.sum())}"
        )
        for ln in model.basin_lines(elite):
            _LIVE.persist(f"    {ln}")
        _progress_bar(1.0 / gens, 1, gens, "gen done")

        for gen in range(2, gens + 1):
            if rl_refine is not None and cfg.rl_steps_per_gen > 0:
                refined = rl_refine(elite, solver, cfg, rng, freqs, n_steps=cfg.rl_steps_per_gen)
                refined = eval_to_peak(solver, refined, freqs, cfg, rng, model, elite=elite)
                if (refined.score or -1e18) > (elite.score or -1e18):
                    elite = clone_ind(refined)

            improved = None
            for attempt in range(retries):
                kids = spawn_children(
                    pop,
                    rng,
                    cfg,
                    feed,
                    cfg.population - 1,
                    model=model,
                    elite=elite,
                    gen=gen,
                    gens=gens,
                )
                ga_diag = sum(int(getattr(k, "ga_diag_cut", 0) or 0) for k in kids)
                if ga_diag:
                    _LIVE.persist(f"    GA diagonal cut  gen={gen}  cells={ga_diag}")
                n_kids = max(len(kids), 1)
                align, _pair = model.alignment(elite)
                _progress_bar(
                    (gen - 1) / gens,
                    gen,
                    gens,
                    f"batch {attempt+1}/{retries} kids={len(kids)} align={align:.2f}",
                )
                for i, kid in enumerate(kids):
                    kids[i] = eval_to_peak(solver, kid, freqs, cfg, rng, model, elite=elite)
                    _progress_bar(
                        (gen - 1 + (i + 1) / n_kids) / gens,
                        gen,
                        gens,
                        f"child {i+1}/{n_kids}",
                    )
                better = [k for k in kids if (k.score or -1e18) > (elite.score or -1e18)]
                extras: list[Individual] = []
                r_probe = None
                if not better and getattr(cfg, "circle_warmup", True):
                    from circle.warmup import probe_after_fail

                    def climb_now(ind: Individual) -> Individual:
                        return eval_to_peak(solver, ind, freqs, cfg, rng, model, elite=elite)

                    r_probe, extras = probe_after_fail(
                        elite, cfg, rng, climb_now, model=model, attempt=attempt
                    )
                    better = [
                        k for k in (kids + extras) if (k.score or -1e18) > (elite.score or -1e18)
                    ]
                    if extras:
                        _LIVE.persist(
                            f"    circle extra r={r_probe:g}  n={len(extras)}  "
                            f"max={max(extras, key=lambda x: x.score or -1e18).score:.2f}"
                        )

                if better:
                    improved = max(better, key=lambda k: k.score or -1e18)
                    shown = kids + extras
                    for i, kid in enumerate(shown):
                        dump_byproduct(
                            cfg,
                            kid,
                            freqs,
                            gen=gen,
                            batch=attempt + 1,
                            idx=i + 1,
                            kept=int(kid is improved),
                            kind="kid" if i < len(kids) else "ring",
                        )
                    pop = keep_metric_parents(
                        pop + shown, cfg.population, elite=improved, cfg=cfg
                    )
                    _LIVE.persist(
                        f"    updated {elite.score:.2f} → {improved.score:.2f}  "
                        f"copper={int(improved.metal.sum())}  src={improved.src}"
                    )
                    elite = clone_ind(improved)
                    break
                shown = kids + extras
                for i, kid in enumerate(shown):
                    dump_byproduct(
                        cfg,
                        kid,
                        freqs,
                        gen=gen,
                        batch=attempt + 1,
                        idx=i + 1,
                        kept=0,
                        kind="kid" if i < len(kids) else "ring",
                    )
                pool = [k for k in shown if k.score is not None]
                if pool:
                    pop = keep_metric_parents(
                        pop + shown, cfg.population, elite=elite, cfg=cfg
                    )
                    _LIVE.persist(
                        "    elite not beaten → keep score/metric peaks in parent pool"
                    )
                else:
                    _LIVE.persist("    no better offspring → discard and regenerate")

            if improved is None:
                align, _pair = model.alignment(elite)
                msg = f"  stop: {retries} regenerations failed to beat gen{gen-1} score."
                if align >= 0.45:
                    msg += f"  3-dir alignment={align:.2f} → likely local convergence in same basin."
                _LIVE.persist(msg)
                for ln in model.basin_lines(elite):
                    _LIVE.persist(f"    {ln}")
                _progress_bar((gen - 1) / gens, gen - 1, gens, "stop")
                break

            best_hist.append(elite.score)
            save_evolution(cfg, elite, freqs, gen=gen, solver=solver, model=model)
            _LIVE.persist(
                f"  ★ best score={elite.score:.2f}  "
                f"S11_worst={elite.worst:.1f} dB  copper={int(elite.metal.sum())}"
            )
            for ln in model.basin_lines(elite):
                _LIVE.persist(f"    {ln}")
            _progress_bar(gen / gens, gen, gens, "gen done")

        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(cfg.out_dir / "score_history.csv", np.array(best_hist, dtype=float), delimiter=",")
        if elite is not None:
            save_evolution(cfg, elite, freqs, solver=solver, model=model)
            for ln in model.basin_lines(elite):
                _LIVE.persist(f"  final {ln}")
        return elite
    finally:
        _LIVE.close()
        _LIVE = None

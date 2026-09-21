# -*- coding: utf-8 -*-
"""유전 알고리즘. 진화 기준(적합도, 대칭, 돌연변이)은 main 에서 넘긴 Config 를 따른다."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import (
    PIX,
    CENTER,
    Individual,
    clone_ind,
    apply_symmetry,
    keep_connected,
    repair,
    seed_rectangle,
    seed_random,
    DIFFUSE_K,
    diffuse_grow,
)
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
    """최근 5줄 + 진행바. TTY면 제자리 갱신, 아니면 평가마다 바를 \\r 로 덮음."""

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
        phase = "warmup" if self.gen <= 0 else f"gen {self.gen}/{max(int(self.total), 1)}"
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

    def set_bar(self, frac: float, gen: int, total: int, extra: str = ""):
        self.frac = min(max(float(frac), 0.0), 1.0)
        self.gen = int(gen)
        self.total = max(int(total), 1)
        if extra.strip():
            self.extra = extra.strip()
        self._render_bar()
        self._emit()

    def persist(self, msg: str):
        if self.tty and self.drawn:
            self.drawn = False
        elif not self.tty:
            sys.stdout.write("\n")
        print(msg, flush=True)
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
        _LIVE.persist(f"    reveal {tag}  n={int(det)}")


def crossover(a: Individual, b: Individual, rng: np.random.Generator, feed) -> Individual:
    mask = rng.random((PIX, PIX)) < 0.5
    child = np.where(mask, a.metal, b.metal).astype(np.uint8)
    mode = a.mode if rng.random() < 0.5 else b.mode
    return repair(Individual(child, mode), feed)


def mutate(ind: Individual, rng: np.random.Generator, feed, mut_pixels: int,
           diffuse_steps: int = 2, diffuse_rate: float = 0.4,
           kernel: np.ndarray | None = None) -> Individual:
    g = ind.metal.copy()
    ys, xs = np.nonzero(g)
    if len(xs) == 0:
        g[feed] = 1
        return Individual(g, ind.mode)
    for _ in range(mut_pixels):
        if rng.random() < 0.7 and len(xs):
            k = int(rng.integers(0, len(xs)))
            i, j = int(ys[k]), int(xs[k])
            # 70% 는 가장자리에서 구리를 늘림 (지우기보다 성장)
            if rng.random() < 0.25:
                g[i, j] = 0
            else:
                di, dj = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
                ni, nj = np.clip(i + di, 0, PIX - 1), np.clip(j + dj, 0, PIX - 1)
                g[ni, nj] = 1
        else:
            i = int(rng.integers(0, PIX))
            j = int(rng.integers(0, PIX))
            g[i, j] = 1
    g = diffuse_grow(g, rng, diffuse_rate, diffuse_steps, kernel=kernel)
    g[feed] = 1
    return repair(Individual(g, ind.mode), feed)


def air_metrics(air: np.ndarray | None, r_mm: float = 15.0):
    """
    공기면 |E|_rms. 어디를 쏘든 한 점에 모일수록 큼.
    D_dB : |E|^2 의 peak/평균
    beam : 피크 주변 r_mm 원 안 에너지 비율 (중심이 아님)
    peak : 피크 위치 (mm)
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


def match_efficiency(s11: np.ndarray, freqs: np.ndarray, cfg) -> float:
    """대역 평균 정합 효율 1-|S11|^2. 반사가 적을수록 1."""
    band = (freqs >= cfg.band_hz[0]) & (freqs <= cfg.band_hz[1])
    if not np.any(band):
        band = np.ones_like(s11, dtype=bool)
    mag2 = np.clip(10.0 ** (np.asarray(s11)[band] / 10.0), 0.0, 0.999)
    return float(np.mean(1.0 - mag2))


def gain_parts(s11, freqs, cfg, air=None, metal=None):
    """지향성 대용 D_use 와 정합효율. score_s11 과 지표별 맵이 같이 씀."""
    _d_db, beam, _peak, r_rms = air_metrics(air)
    eta = match_efficiency(s11, freqs, cfg) if s11 is not None else 0.0
    eta_db = 10.0 * np.log10(eta + 1e-15)
    f0 = float(getattr(cfg, "f0_hz", 7.25e9))
    lam_mm = 3.0e8 / max(f0, 1.0) * 1e3
    area = np.pi * max(r_rms, 1.0) ** 2
    d_ap = 10.0 * np.log10(np.clip(4.0 * np.pi * area / (lam_mm**2), 1e-8, 1e6))
    beam_db = 10.0 * np.log10(beam / 0.13 + 1e-15)
    d_use = 0.5 * d_ap + 0.5 * beam_db
    copper = int(np.asarray(metal).sum()) if metal is not None else 0
    return d_use, eta, eta_db, beam, r_rms, copper


def score_s11(s11: np.ndarray, freqs: np.ndarray, cfg, air=None, metal=None) -> tuple[float, float]:
    """항상 최대화가 좋게. gain 이면 실현이득 근사 D×η."""
    band = (freqs >= cfg.band_hz[0]) & (freqs <= cfg.band_hz[1])
    if not np.any(band):
        band = np.ones_like(s11, dtype=bool)
    worst = float(np.max(s11[band]))
    at_f0 = float(s11[np.argmin(np.abs(freqs - cfg.f0_hz))])
    fitness = getattr(cfg, "fitness", "gain")
    if fitness == "f0":
        return -at_f0, worst
    if fitness == "bandwidth":
        below = s11 < cfg.s11_target_db
        return float(np.mean(below[band])) * 20.0 - 0.2 * worst, worst
    if fitness == "band_worst":
        return -0.7 * worst - 0.3 * at_f0, worst
    d_use, eta, eta_db, _beam, _r_rms, copper = gain_parts(s11, freqs, cfg, air=air, metal=metal)
    wd = float(getattr(cfg, "w_directivity", 1.0))
    we = float(getattr(cfg, "w_efficiency", 1.0))
    min_cu = int(getattr(cfg, "min_copper", 80))
    pen = 0.0
    if copper < min_cu:
        pen = -30.0 * (1.0 - copper / max(min_cu, 1))
    score = wd * d_use + we * eta_db + pen
    return float(score), worst


def save_plot(out: Path, tag: str, metal, s11, freqs, score, worst, feed, air=None, cfg=None):
    out.mkdir(parents=True, exist_ok=True)
    nax = 3 if air is not None else 2
    fig, axes = plt.subplots(1, nax, figsize=(6 * nax, 5.6))
    if nax == 2:
        ax1, ax2 = axes
        ax_e = None
    else:
        ax1, ax_e, ax2 = axes
    copper = int(np.asarray(metal).sum())
    ax1.imshow(metal.T, origin="lower", cmap="Greys", extent=[-25.5, 25.5, -25.5, 25.5])
    ax1.plot(feed[0] - 25, feed[1] - 25, "rx", ms=8, mew=2)
    ax1.set_title(f"{tag}  copper={copper}px  feed={feed}")
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    d_db, beam, peak, r_rms = 0.0, 0.0, None, 0.0
    eta = 0.0
    if ax_e is not None:
        e = np.asarray(air, dtype=np.float64)
        nx, ny = e.shape
        ext = [-nx / 2.0, nx / 2.0, -ny / 2.0, ny / 2.0]
        finite = e[np.isfinite(e)]
        emax = float(np.max(finite)) if finite.size else 1.0
        pos = finite[finite > 0]
        emin = float(np.min(pos)) if pos.size else 0.0
        vlo = float(np.percentile(finite, 2)) if finite.size else 0.0
        vhi = emax
        if not np.isfinite(vhi) or vhi <= vlo:
            vhi = vlo + 1e-30
        im = ax_e.imshow(
            e.T,
            origin="lower",
            cmap="inferno",
            extent=ext,
            aspect="equal",
            vmin=vlo,
            vmax=vhi,
        )
        ax_e.contour(metal.T, levels=[0.5], extent=[-25.5, 25.5, -25.5, 25.5], colors="w", linewidths=0.6)
        if emax > 0:
            xs = np.linspace(ext[0], ext[1], nx)
            ys = np.linspace(ext[2], ext[3], ny)
            ax_e.contour(
                xs,
                ys,
                e.T,
                levels=[0.5 * emax],
                colors="#7CFF6B",
                linewidths=0.9,
            )
        d_db, beam, peak, r_rms = air_metrics(e)
        if peak is not None:
            circ = plt.Circle(peak, 15.0, fill=False, color="cyan", lw=1.0, ls="--")
            ax_e.add_patch(circ)
            ax_e.plot(peak[0], peak[1], "c+", ms=10, mew=2)
        ratio = emax / max(emin, 1e-30)
        ax_e.set_title(
            f"air |E|  min={emin:.2e}  max={emax:.2e}  max/min={ratio:.1f}  "
            f"beam={beam:.2f}  r_rms={r_rms:.1f}mm"
        )
        ax_e.set_xlabel("x (mm)")
        cb = fig.colorbar(im, ax=ax_e, fraction=0.046, pad=0.04)
        cb.set_label("|E|")
    elif air is not None:
        d_db, beam, peak, r_rms = air_metrics(air)
    if cfg is not None and s11 is not None:
        eta = match_efficiency(s11, freqs, cfg)
    f0 = float(getattr(cfg, "f0_hz", 7.25e9)) / 1e9 if cfg is not None else 7.25
    ax2.plot(np.asarray(freqs) / 1e9, s11, "b-")
    ax2.axhline(-10, color="r", ls="--")
    ax2.axvline(f0, color="gray", ls=":")
    ax2.set_ylim([-40, 0])
    ax2.grid(True, ls=":")
    ax2.set_title(f"S11  worst={worst:.1f}dB  η={100 * eta:.0f}%")
    ax2.set_xlabel("GHz")
    ax2.set_ylabel("S11 (dB)")
    if cfg is not None:
        band = getattr(cfg, "band_hz", (7e9, 7.5e9))
        wd = float(getattr(cfg, "w_directivity", 1.0))
        we = float(getattr(cfg, "w_efficiency", 1.0))
        fitness = getattr(cfg, "fitness", "gain")
        min_cu = int(getattr(cfg, "min_copper", 80))
        peak_s = f"({peak[0]:.1f},{peak[1]:.1f})mm" if peak is not None else "-"
        foot = (
            f"score={score:.2f} = {wd:g}·(0.5 D_ap + 0.5 beam_dB) + {we:g}·η_dB  "
            f"minCu={min_cu}  FITNESS={fitness}  beam@peak={beam:.2f}  r_rms={r_rms:.1f}mm  "
            f"peak={peak_s}  η={100 * eta:.0f}%  S11_worst={worst:.1f}dB  copper={copper}  "
            f"band={band[0]/1e9:.2f}-{band[1]/1e9:.2f}GHz  f0={f0:.2f}GHz"
        )
        fig.text(0.5, 0.012, foot, ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(out / f"{tag}.jpg", dpi=130)
    plt.close(fig)


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
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.imshow(ind.metal.T, origin="lower", cmap="Greys")
    ax.set_axis_off()
    ax.set_title(
        f"{name}  sc={ind.score:.2f}  Cu={int(ind.metal.sum())}  kept={kept}",
        fontsize=7,
    )
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


def evaluate(solver, metal, freqs, cfg):
    if _LIVE is not None:
        with _mute_fd2():
            s11, mindb, minhz, nused = solver.run(metal, freqs)
    else:
        s11, mindb, minhz, nused = solver.run(metal, freqs)
    air = solver.air_e() if hasattr(solver, "air_e") else None
    sc, worst = score_s11(s11, freqs, cfg, air=air, metal=metal)
    return sc, worst, s11, mindb, minhz, nused, air


def fill_eval(
    solver,
    ind: Individual,
    freqs,
    cfg,
    model: OffspringModel | None = None,
    elite: Individual | None = None,
    note: str = "",
) -> Individual:
    if ind.score is not None and ind.s11 is not None:
        return ind
    sc, worst, s11, mindb, minhz, nused, air = evaluate(solver, ind.metal, freqs, cfg)
    ind.score = sc
    ind.worst = worst
    ind.s11 = s11
    ind.air = air
    d_use, eta, _eta_db, beam, r_rms, _cu = gain_parts(s11, freqs, cfg, air=air, metal=ind.metal)
    ind.d_use = d_use
    ind.eta = eta
    if model is not None:
        model.observe_eval(ind.metal, sc, d_use=d_use, eta=eta, elite=elite)
        if elite is not None and elite.score is not None:
            model.observe_move(elite.metal, ind.metal, sc - float(elite.score))
    tag = f"{note}  " if note else ""
    line = (
        f"{tag}nused={nused}  score={sc:.2f}  D={d_use:.2f}  beam={beam:.2f}  "
        f"r_rms={r_rms:.1f}mm  η={100 * eta:.0f}%  S11_worst={worst:.1f}dB  "
        f"min={mindb:.1f}dB@{minhz / 1e9:.2f}GHz  copper={int(ind.metal.sum())}"
    )
    if _LIVE is not None:
        _LIVE.push(line)
        _LIVE.bump_eval()
    else:
        print(line)
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
    """가장자리 1칸 뒤집기. 오르면 채택. 분지 국소최대까지."""
    fill_eval(solver, ind, freqs, cfg, model=model, elite=elite, note="raw")
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
            cand = repair(Individual(g, ind.mode, src=ind.src), feed)
            if np.array_equal(cand.metal, ind.metal):
                continue
            fill_eval(solver, cand, freqs, cfg, model=model, elite=elite, note="try")
            used += 1
            if (cand.score or -1e18) > (ind.score or -1e18):
                ds = float(cand.score - ind.score)
                if model is not None:
                    model.observe_step(ind.metal, cand.metal, ds)
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
    return local_climb(solver, ind, freqs, cfg, rng, model, elite=elite)


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

    def pick():
        k = min(cfg.tournament, len(scored))
        ix = rng.choice(len(scored), size=k, replace=False)
        return scored[int(ix[np.argmax([scored[i].score or -1e18 for i in ix])])]

    kids = []
    while len(kids) < n:
        mode = int(rng.choice(getattr(cfg, "symmetry_modes", [-1])))
        if rng.random() < p_rand:
            child = seed_random(rng, feed, mode)
        else:
            child = crossover(pick(), pick(), rng, feed)
            child = mutate(
                child,
                rng,
                feed,
                cfg.mut_pixels,
                getattr(cfg, "diffuse_steps", 2),
                getattr(cfg, "diffuse_rate", 0.4),
                kernel=None if model is None else model.annealed_kernel(),
            )
        child.score = None
        child.s11 = None
        child.air = None
        child.src = "ga"
        kids.append(child)
    n_rl = int(getattr(cfg, "extra_rl", 0))
    n_df = int(getattr(cfg, "extra_diffuse", 0))
    if model is not None and n_rl > 0:
        mode = elite.mode if elite is not None else -1
        kids.extend(model.sample_rl(n_rl, rng, feed, mode, parent=elite))
    if model is not None and elite is not None and n_df > 0:
        kids.extend(model.sample_diffuse_b(n_df, elite, rng, feed))
    return kids


def _progress_bar(frac: float, gen: int, total: int, extra: str = "") -> None:
    if _LIVE is not None:
        _LIVE.set_bar(frac, gen, total, extra)
        return
    width = 28
    total = max(int(total), 1)
    frac = min(max(float(frac), 0.0), 1.0)
    filled = int(round(frac * width))
    bar = "#" * filled + "-" * (width - filled)
    extra = extra.strip()
    tail = f"  {extra}" if extra else ""
    print(f"  [{bar}] {100.0 * frac:5.1f}%  gen {gen}/{total}{tail}", flush=True)


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
    print(
        f"GA start  pop={cfg.population} gens={gens}  "
        f"backend={getattr(solver, 'backend', '?')}  fitness={cfg.fitness}  "
        f"feed={feed}  max_offspring_batches={retries}  "
        f"extra_rl={extra_rl} extra_diffuseB={extra_df}  "
        f"climb={getattr(cfg, 'local_climb_steps', 8)}x{getattr(cfg, 'local_try_per_step', 4)}  "
        f"circle_warmup={getattr(cfg, 'circle_warmup', True)}  "
        f"T고정  align로그만"
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

            _LIVE.persist("  circle 예열 (원 안 꼭대기 → 진화 시작)")
            _progress_bar(0.0, 0, gens, "seed 0")

            def on_warm(frac: float, msg: str):
                _progress_bar(frac, 0, gens, msg)

            pop = run_circle_warmup(
                cfg,
                rng,
                climb=climb,
                model=model,
                log=lambda m: _LIVE.persist(m),
                progress=on_warm,
            )
        else:
            _LIVE.persist("  gen1 초기 집단 (랜덤)")
            while len(pop) < cfg.population:
                mode = int(rng.choice(cfg.symmetry_modes))
                pop.append(seed_random(rng, feed, mode))
        _progress_bar(0.0, 1, gens)
        n_pop = len(pop)
        for n, ind in enumerate(pop):
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
        save_plot(
            cfg.out_dir,
            "Generation_001_Best",
            elite.metal,
            elite.s11,
            freqs,
            elite.score,
            elite.worst,
            feed,
            air=elite.air,
            cfg=cfg,
        )
        _LIVE.persist(
            f"  ★ gen1 elite score={elite.score:.2f}  "
            f"S11_worst={elite.worst:.1f} dB  copper={int(elite.metal.sum())}"
        )
        for ln in model.basin_lines(elite):
            _LIVE.persist(f"    {ln}")
        _reveal(cfg, model, "Reveal_gen001", note="gen1", center=elite.metal)
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
                    from circle.warmup import inject_peak, probe_after_fail

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
                            f"    원 추가 r={r_probe:g}  n={len(extras)}  "
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
                    rest = [k for k in shown if k is not improved]
                    pop = [clone_ind(improved)] + rest
                    pop = pop[: cfg.population]
                    _LIVE.persist(
                        f"    갱신 {elite.score:.2f} → {improved.score:.2f}  "
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
                    from circle.warmup import inject_peak

                    batch_max = max(pool, key=lambda k: k.score or -1e18)
                    pop = inject_peak(pop, elite, batch_max, cfg.population)
                    _LIVE.persist(
                        f"    엘리트 못 넘김 → 추가탐색 최고 {batch_max.score:.2f} 를 집단에 유지"
                    )
                else:
                    _LIVE.persist("    더 나은 자손 없음 → 버리고 다시 생성")

            if improved is None:
                align, _pair = model.alignment(elite)
                msg = f"  중단: {retries}번 다시 만들어도 gen{gen-1} 점수를 못 넘김."
                if align >= 0.45:
                    msg += f"  세 방향 정렬={align:.2f} → 같은 분지 국소수렴 가능."
                _LIVE.persist(msg)
                for ln in model.basin_lines(elite):
                    _LIVE.persist(f"    {ln}")
                _reveal(cfg, model, f"Reveal_gen{gen-1:03d}_stop", note="stop", center=elite.metal)
                _progress_bar((gen - 1) / gens, gen - 1, gens, "stop")
                break

            best_hist.append(elite.score)
            save_plot(
                cfg.out_dir,
                f"Generation_{gen:03d}_Best",
                elite.metal,
                elite.s11,
                freqs,
                elite.score,
                elite.worst,
                feed,
                air=elite.air,
                cfg=cfg,
            )
            _LIVE.persist(
                f"  ★ gen{gen}/{gens} elite score={elite.score:.2f}  "
                f"S11_worst={elite.worst:.1f} dB  copper={int(elite.metal.sum())}"
            )
            for ln in model.basin_lines(elite):
                _LIVE.persist(f"    {ln}")
            _reveal(cfg, model, f"Reveal_gen{gen:03d}", note=f"gen{gen}", center=elite.metal)
            _progress_bar(gen / gens, gen, gens, "gen done")

        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(cfg.out_dir / "score_history.csv", np.array(best_hist, dtype=float), delimiter=",")
        if elite is not None and elite.s11 is not None:
            save_plot(
                cfg.out_dir,
                "GA_final_best",
                elite.metal,
                elite.s11,
                freqs,
                elite.score,
                elite.worst,
                feed,
                air=elite.air,
                cfg=cfg,
            )
        if elite is not None:
            for ln in model.basin_lines(elite):
                _LIVE.persist(f"  final {ln}")
            _reveal(cfg, model, "Reveal_final", note="final", center=elite.metal)
        return elite
    finally:
        _LIVE.close()
        _LIVE = None

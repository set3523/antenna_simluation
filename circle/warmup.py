# -*- coding: utf-8 -*-
"""Direct FDTD on the seed circle, then grow r and infer only needed angles. No evolution here."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from circle.acquire import (
    assign_metric_key,
    pick_cone_probe,
    pick_fork_theta,
    pick_heading_split,
    pick_inertial_probe,
    pick_ring_probes,
)
from circle.index_map import (
    attach_spiral_sample,
    attach_sample_uv,
    chart_dist,
    chart_neighbor_rs8,
    seed_chart_metal,
    uv_global,
    uv_local,
    verify_mask_matches_rs,
)
from circle.spiral_chart import SpiralChart
from circle.pattern import breaks_score_trend, next_trend_probes, reconnect_by_pattern
from circle.roots import (
    HEADING_SEP,
    KEYS,
    Root,
    apply_steer,
    cone_half,
    heading_of,
    keep_metric_parents,
    rematch_live_roots_to_chains,
    label_disk_points,
    metric_val,
    same_key_headings,
    start_from_chains,
    start_from_ring,
)
from circle.schedule import RadiusSchedule
from core import PIX, Individual, clone_ind, hamming, is_feasible, seed_random, seed_rectangle
from learn.infer import (
    disk_holes,
    disk_rings,
    inferred_combined_max,
    length_scale,
    n_ang_to_cover,
    orange_good_sites,
)
from learn.viz import save_reveal


def snap_best(cfg, ind: Individual, r: float, tag: str | None = None):
    if ind is None or ind.s11 is None or not is_feasible(ind, cfg):
        return
    b = np.asarray(ind.metal, dtype=np.uint8).tobytes()
    t = tag if tag else f"Best_r{int(round(float(r))):03d}"
    key = (b, t)
    if getattr(snap_best, "_last_key", None) == key:
        return
    snap_best._last_key = key
    from opt_ga import save_best

    freqs = np.linspace(cfg.f_start, cfg.f_stop, cfg.n_freq)
    save_best(cfg, ind, freqs, r, tag=tag)


def snap_reveal(cfg, model, tag: str, log=None, note: str = "", r=None) -> float | None:
    if model is None:
        return None
    if r is not None:
        model.chart.set_r_probed(float(r))
    import time

    t0 = time.perf_counter()
    det = save_reveal(cfg.out_dir, tag, model, note=note)
    dt = time.perf_counter() - t0
    if log is not None:
        log(f"    reveal {tag}  FDTD samples n={int(det)} (not all 2601 indices)  {dt:.1f}s")
    return det


def _diverse_pick(
    peaks: list[Individual], n: int, cfg, min_hamming: int = 40
) -> list[Individual]:
    return keep_metric_parents(peaks, n, min_hamming=min_hamming, cfg=cfg)


def _champ(peaks: list[Individual], cfg) -> Individual | None:
    feas = [p for p in peaks if p.score is not None and is_feasible(p, cfg)]
    if not feas:
        return None
    return max(feas, key=lambda x: x.score or -1e18)


def _peaks_for_best(peaks: list[Individual], cfg) -> list[Individual]:
    """Best selection: drop candidates with bad stub/band S11 when possible (gain-score trap)."""
    pool = [
        p
        for p in peaks
        if p.score is not None and is_feasible(p, cfg)
    ]
    if not pool:
        return peaks

    min_cu = int(getattr(cfg, "best_min_upper_copper", 0) or 0)
    if min_cu > 0:
        rich = [p for p in pool if int(np.asarray(p.metal).sum()) >= min_cu]
        if rich:
            pool = rich

    if getattr(cfg, "best_prefer_s11_in_band", True):
        tgt = float(getattr(cfg, "s11_target_db", -10.0))
        matched = [
            p
            for p in pool
            if getattr(p, "worst", None) is not None and float(p.worst) <= tgt + 1e-9
        ]
        if matched:
            pool = matched
    return pool if pool else peaks


def _champ_for_best(
    peaks: list[Individual],
    cfg,
    anchor: tuple[float, float],
    r_cap: float | None,
) -> Individual | None:
    pool = _peaks_for_best(peaks, cfg)
    ch = _champ_in_disk(pool, cfg, anchor, r_cap)
    return ch or _champ(pool, cfg)


def _champ_in_disk(
    peaks: list[Individual],
    cfg,
    anchor: tuple[float, float],
    r_cap: float | None,
) -> Individual | None:
    """Best_r***: top-scoring candidate among those FDTD'd up to chart_r ≤ r_cap."""
    _ = cfg, anchor
    feas = [p for p in peaks if p.score is not None and is_feasible(p, cfg)]
    if not feas:
        return None
    if r_cap is not None and float(r_cap) > 1e-9:
        rc = float(r_cap) + 1e-6
        in_disk = []
        for p in feas:
            if getattr(p, "chart_kind", "") == "baseline":
                in_disk.append(p)
                continue
            cr = getattr(p, "chart_r", None)
            if cr is None:
                continue
            if float(cr) <= rc:
                in_disk.append(p)
        feas = in_disk
    if not feas:
        return None
    return max(feas, key=lambda x: x.score or -1e18)


def schedule_from_cfg(cfg) -> RadiusSchedule:
    """Extra exploration radius after GA stalls (linear step)."""
    from circle.schedule import LinearStep

    return LinearStep(
        r_max=float(getattr(cfg, "circle_r_max", 130.0)),
        step=float(getattr(cfg, "circle_r_step", 4.0)),
    )


def r_disk_end_from_cfg(cfg, r_max: float, r_step: float) -> float:
    """Last chart r of the disk-ring FDTD (trend lines start after it)."""
    end = getattr(cfg, "circle_r_disk_end", None)
    if end is not None and float(end) > 0.0:
        r_disk = float(end)
    else:
        fr0 = getattr(cfg, "circle_r_fracs", (0.20,))
        frac = float(fr0[0]) if fr0 else 0.20
        r_disk = float(r_max) * frac
    step = max(float(r_step), 1e-6)
    r_disk = step * round(r_disk / step)
    return float(np.clip(r_disk, step, float(r_max)))


def probe_after_fail(
    center: Individual,
    cfg,
    rng: np.random.Generator,
    climb: Callable[[Individual], Individual],
    model=None,
    attempt: int = 0,
) -> tuple[float, list[Individual]]:
    """After failing to beat the elite, explore the circle edge a bit more with the updated map/B."""
    radii = schedule_from_cfg(cfg).radii()
    r = float(radii[min(max(int(attempt), 0), len(radii) - 1)])
    n_ang = max(1, int(getattr(cfg, "circle_fail_angles", 2)))
    offset = float(rng.random() * 2.0 * np.pi)
    b = None if model is None else getattr(model, "b", None)
    peaks = []
    for i in range(n_ang):
        th = offset + 2.0 * np.pi * i / n_ang
        anchor = (0, 0)
        spiral_h = SpiralChart.from_seed(center.metal, cfg.feed, int(getattr(cfg, "circle_r_max", 130)))
        kid = spiral_h.individual_from_polar(r, th, mode=center.mode, src="circleH")
        if kid is None:
            continue
        kid = climb(kid)
        if is_feasible(kid, cfg):
            peaks.append(kid)
    if model is not None:
        model.chart.set_center(center.metal)
        bind_chart_origin(model, center, cfg)
        model.chart.set_r_probed(r)
    return r, peaks


def inject_peak(
    pop: list[Individual], elite: Individual, peak: Individual, n: int
) -> list[Individual]:
    """Keep the elite and add the best extra-exploration result to the population."""
    out = [clone_ind(elite)]
    rest = [p for p in pop if not np.array_equal(p.metal, elite.metal)]
    if peak.score is None or np.array_equal(peak.metal, elite.metal):
        return (out + rest)[:n]
    rest = [p for p in rest if not np.array_equal(p.metal, peak.metal)]
    rest.append(clone_ind(peak))
    rest.sort(key=lambda x: x.score if x.score is not None else -1e18, reverse=True)
    return (out + rest)[:n]


def _anchor_kind(cfg) -> str:
    return str(getattr(cfg, "circle_anchor", "origin") or "origin").lower().strip()


def make_warmup_seeds(
    cfg,
    rng: np.random.Generator,
    feed,
    n: int,
    modes: list,
) -> list[Individual]:
    """Anchor mode applies to the first seed only. openEMS gate rectangle is not used here (origin default)."""
    n = max(int(n), 1)
    modes = list(modes if modes else [-1])
    kind = _anchor_kind(cfg)
    out: list[Individual] = []
    for i in range(n):
        mode = int(rng.choice(modes))
        if i == 0 and kind == "rectangle":
            out.append(seed_rectangle(feed))
        elif i == 0 and kind == "origin":
            g = seed_chart_metal(feed, symmetry_mode=4)
            out.append(Individual(g, mode=4, src="cseed_origin"))
        else:
            out.append(seed_random(rng, feed, mode))
    return out


def warmup_anchor(seed: Individual, cfg) -> tuple[int, int]:
    _ = seed, cfg
    return (0, 0)


def bind_chart_origin(model, seed: Individual, cfg) -> tuple[int, int]:
    _ = seed, cfg
    if model is not None:
        model.chart.set_origin(0, 0)
    return (0, 0)


def _expand_radii(r_seed: float, r_step: float, r_max: float) -> list[float]:
    step = max(float(r_step), 1e-6)
    r_max = max(float(r_max), float(r_seed) + step)
    out = []
    r = float(r_seed) + step
    while r <= r_max + 1e-9:
        out.append(float(r))
        r += step
    return out


def run_circle_warmup(
    cfg,
    rng: np.random.Generator,
    climb: Callable[[Individual], Individual],
    model=None,
    log=None,
    progress=None,
) -> list[Individual]:
    """
    Initial disk (r≤r_disk) gets direct FDTD in Δr rings. r_disk=`circle_r_disk_end` or R_FRACS[0]×r_max.
    Beyond r_disk, only per-metric trend lines (branches, waves). No live trend lines → evolution.
    Hand off to evolution when no trend lines are alive.
    climb(ind) is an Individual that has only been evaluated. Chart-extremum 8-neighbor checks are done here separately.
    """
    feed = cfg.feed
    modes = list(getattr(cfg, "symmetry_modes", [-1]))
    n_seed = int(getattr(cfg, "circle_seeds", 1))
    n_ang = int(getattr(cfg, "circle_seed_angles", getattr(cfg, "circle_angles", 6)))
    n_cand = int(getattr(cfg, "circle_expand_cand", 16))
    r_step = float(getattr(cfg, "circle_r_step", 4.0))
    r_max = float(getattr(cfg, "circle_r_max", 48.0))
    r_full = float(r_max)
    r_disk = r_disk_end_from_cfg(cfg, r_max, r_step)
    fracs = tuple(float(x) for x in getattr(cfg, "circle_r_fracs", (0.20,)))
    stall_need = max(1, int(getattr(cfg, "circle_stagnate", 2)))
    root_max = max(1, int(getattr(cfg, "circle_root_max", 6)))
    n_pop = int(cfg.population)
    early_rings = disk_rings(r_disk, r_step)
    if early_rings and abs(float(early_rings[-1]) - r_disk) > 0.25 * r_step:
        early_rings = list(early_rings) + [float(r_disk)]
    elif not early_rings:
        early_rings = [float(r_disk)]
    ring_angs = [n_ang_to_cover(rr, r_step=r_step) for rr in early_rings]
    n_hole = max(0, int(getattr(cfg, "circle_disk_holes", 12)))
    n_orange = int(getattr(cfg, "circle_orange_best", 0))
    orange_slack = float(getattr(cfg, "circle_orange_slack", 0.35))
    orange_job_est = (n_orange + 9) if n_orange > 0 else 48
    expand_rs = _expand_radii(r_disk, r_step, r_max)
    n_jobs = max(
        n_seed
        * (
            1
            + sum(ring_angs)
            + len(early_rings) * n_hole
            + (len(early_rings) + len(expand_rs)) * orange_job_est
        ),
        1,
    )
    done = 0
    cur_r = 0.0
    n_alive = 0

    def say(msg: str):
        if log is not None:
            log(msg)
        else:
            print(msg, flush=True)

    def tick(msg: str, inc: bool = False, r=None):
        nonlocal done, cur_r
        if r is not None:
            cur_r = float(r)
        if inc:
            done += 1
        if progress is not None:
            frac = min(cur_r / max(r_max, 1e-6), 1.0)
            progress(
                frac,
                msg,
                r=cur_r,
                r_max=r_max,
                r_map_full=r_full,
                stall=f"{n_alive}/{root_max}",
            )

    say(
        f"  seed circle exploration  seeds={n_seed}  anchor={_anchor_kind(cfg)}  "
        f"disk≤r_disk={r_disk:g} ({100.0*r_disk/r_max:.0f}% of r_max)  "
        f"then trend lines→r_max={r_max:g}  "
        f"rings={','.join(f'{x:g}×{a}' for x, a in zip(early_rings, ring_angs))}  "
        f"step={r_step:g}  "
        f"marks={','.join(f'{int(round(f*100))}%' for f in fracs)}  "
        f"stall {stall_need}  trend lines≤{root_max}"
    )
    peaks: list[Individual] = []
    seeds = make_warmup_seeds(cfg, rng, feed, n_seed, modes)

    for si, seed in enumerate(seeds):
        snap_best._last_key = None
        marked_frac: set[float] = set()
        last_best_sc = -1e18

        def mark_fracs(r_now: float):
            nonlocal last_best_sc
            if not peaks:
                return
            champ = _champ_for_best(peaks, cfg, anchor, r_now)
            if champ is None:
                champ = _champ(peaks, cfg)
            if champ is None:
                return
            sc = float(champ.score or -1e18)
            if sc > last_best_sc + 1e-9:
                if last_best_sc > -1e17:
                    say(f"    Best updated  {last_best_sc:.2f} → {sc:.2f}  r={r_now:g}")
                last_best_sc = sc
            for f in fracs:
                if f in marked_frac:
                    continue
                if r_now + 1e-9 < float(f) * r_full:
                    continue
                marked_frac.add(f)
                pct = int(round(float(f) * 100.0))
                snap_best(cfg, champ, r_now, tag=f"Best_p{pct:02d}")
                snap_reveal(
                    cfg,
                    model,
                    f"Reveal_p{pct:02d}",
                    log=log,
                    note=f"{pct}% r={float(f)*r_full:g} best={sc:.2f}",
                    r=r_now,
                )
                say(f"    {pct}%  Best_p{pct:02d}  score={sc:.2f}  r={r_now:g}")
        seed.src = f"cseed{si+1}"
        seed.chart_kind = "seed"
        if model is not None:
            model.chart.set_r_seed(r_disk)
        anchor = bind_chart_origin(model, seed, cfg)
        center = seed
        spiral = SpiralChart.from_seed(seed.metal, feed, int(r_max))
        chart_diag_skip = 0

        b = None if model is None else getattr(model, "b", None)

        def _chart_ell():
            su, sv, ss, _sr = model.chart.points(b)
            ell = length_scale(
                su,
                sv,
                r_step=r_step,
                r_seed=max(r_disk, 4.0),
                n_seed_ang=max(ring_angs + [n_ang]),
            )
            return su, sv, ss, ell

        def eval_at_polar(r_pt: float, theta: float, src: str, msg: str, r_tick: float):
            nonlocal chart_diag_skip
            tick(msg, r=r_tick)
            kid = spiral.individual_from_polar(
                r_pt, theta, mode=seed.mode, src=src, reject_diagonal=True
            )
            if kid is None:
                chart_diag_skip += 1
                say(f"    skip diagonal/range  r={r_pt:g} θ={theta:.2f}  ({src})")
                return None
            kid.chart_kind = "seed"
            kid = climb(kid)
            rs = int(getattr(kid, "chart_s", 0) or 0)
            rr = int(round(float(getattr(kid, "chart_r", r_pt) or r_pt)))
            if not verify_mask_matches_rs(kid.metal, spiral, rr, rs):
                say(f"    WARN chart/FDTD mask mismatch  {src}  r={rr} s={rs}")
            attach_spiral_sample(kid, rr, rs, theta, spiral)
            kid.chart_r = float(r_tick)
            tick(msg, inc=True, r=r_tick)
            if is_feasible(kid, cfg):
                peaks.append(kid)
            else:
                wst = kid.worst if kid.worst is not None else float("nan")
                say(
                    f"    infeasible  score={kid.score or 0:.2f}  "
                    f"cu={int(kid.metal.sum())}  S11_w={wst:.1f}dB"
                )
            return kid

        def eval_at_plane(lu: float, lv: float, src: str, msg: str, r_tick: float):
            r_pt = float(np.hypot(lu, lv))
            th = float(np.arctan2(lv, lu)) if r_pt > 1e-9 else 0.0
            return eval_at_polar(r_pt, th, src, msg, r_tick)

        def _path_local(uv: tuple[float, float]) -> tuple[float, float]:
            return uv_local(uv[0], uv[1], anchor)

        def _polar_local(r: float, theta: float) -> tuple[float, float]:
            return float(r * np.cos(theta)), float(r * np.sin(theta))

        def _probe_val(ind: Individual, metric_key: str | None) -> float:
            if metric_key in KEYS:
                return float(metric_val(ind, metric_key))
            return float(ind.score or -1e18)

        def verify_peak_rs8(
            r_c: int,
            s_c: int,
            center_sc: float,
            src: str,
            msg_prefix: str,
            r_tick: float,
            metric_key: str | None = None,
        ) -> float:
            """FDTD on the 8 cells at (r,s) ±1."""
            best = float(center_sc)
            for ni, (nr, ns, nth) in enumerate(chart_neighbor_rs8(r_c, s_c, spiral)):
                nk = eval_at_polar(
                    float(nr),
                    nth,
                    f"{src}v{ni}",
                    f"{msg_prefix} chart±1 {ni + 1}/8",
                    r_tick,
                )
                if nk is not None:
                    sc = _probe_val(nk, metric_key)
                    if sc > best + 1e-9:
                        best = sc
            if best > float(center_sc) + 1e-9:
                lab = metric_key or "score"
                say(
                    f"    {msg_prefix} not a peak ({lab}): "
                    f"center {center_sc:.2f} < best neighbor {best:.2f}"
                )
            return best

        def compute_orange_bests(r_lim: float) -> None:
            if model is None:
                return
            su, sv, ss, ell = _chart_ell()
            if su.size == 0:
                return
            ch = _champ(peaks, cfg)
            champ = float(ch.score or -1e18) if ch is not None else -1e18
            d_peak = model.chart.series_at_points("leave")
            beam = model.chart.series_at_points("beam")
            eta = model.chart.series_at_points("eta")
            sites = orange_good_sites(
                su,
                sv,
                ell,
                champ,
                r_lim,
                scores=ss,
                d_peak=d_peak,
                beam=beam,
                eta=eta,
                leave=d_peak,
                n_max=n_orange,
                slack=orange_slack,
                r_step=r_step,
            )
            if sites:
                say(f"    inferred (orange) score candidates {len(sites)}  r_lim={r_lim:g}  champ={champ:.2f}")
            for hi, (lu, lv, yhat) in enumerate(sites):
                kid = eval_at_plane(
                    lu,
                    lv,
                    f"orange{si+1}",
                    f"seed {si+1}/{n_seed} orange {hi+1}/{len(sites)}",
                    r_lim,
                )
                if kid is None:
                    continue
                sc = kid.score or -1e18
                rh = float(np.hypot(lu, lv))
                if sc > champ + 1e-9:
                    say(f"    inferred (orange) est {yhat:.2f} → direct {sc:.2f}  Best  r={rh:g}")
                    champ = sc
                else:
                    say(f"    inferred (orange) est {yhat:.2f} → direct {sc:.2f}  r={rh:g}")
                # Orange (inferred) skips the ±1 8-cell FDTD like trend does (avoids tens of minutes and SMB waits per wave)
            su, sv, ss, ell = _chart_ell()
            mx = inferred_combined_max(
                su,
                sv,
                ell,
                r_lim,
                ss,
                d_peak=model.chart.series_at_points("leave"),
                beam=model.chart.series_at_points("beam"),
                eta=model.chart.series_at_points("eta"),
                leave=model.chart.series_at_points("leave"),
                r_step=r_step,
            )
            if mx is None:
                return
            lu, lv, yh = mx
            dmin = float(np.min((su - lu) ** 2 + (sv - lv) ** 2)) if su.size else 1e18
            if dmin >= (0.45 * r_step) ** 2:
                kid = eval_at_plane(
                    lu,
                    lv,
                    f"imax{si+1}",
                    f"seed {si+1}/{n_seed} imax",
                    r_lim,
                )
                if kid is not None:
                    sc_im = float(kid.score or -1e18)
                    say(
                        f"    inferred peak est {yh:.2f} → direct {sc_im:.2f}  "
                        f"local({lu:.1f},{lv:.1f})"
                    )
                    verify_peak_rs8(
                        int(round(float(kid.chart_r or 0))),
                        int(kid.chart_s or 0),
                        sc_im,
                        f"imax{si+1}",
                        f"inferred ŷ peak",
                        r_lim,
                    )

        if getattr(cfg, "circle_baseline_rect", True):
            bl = seed_rectangle(feed)
            bl.src = f"baseline{si+1}"
            bl.chart_kind = "baseline"
            attach_sample_uv(bl, anchor[0], anchor[1])
            bl = climb(bl)
            attach_sample_uv(bl, anchor[0], anchor[1])
            if is_feasible(bl, cfg):
                peaks.append(bl)
                say(
                    f"    baseline top-layer rect  cu={int(bl.metal.sum())}  "
                    f"score={bl.score:.2f}  S11_w={bl.worst:.1f}dB"
                )

        disk_rot = float(rng.uniform(0.0, 2.0 * np.pi))
        eval_at_polar(0.0, 0.0, f"seed0{si+1}", f"seed {si+1}/{n_seed} r=0", 0.0)
        for r_ring, n_ring in zip(early_rings, ring_angs):
            thetas = (
                disk_rot + np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
            ) % (2.0 * np.pi)
            for ai, theta in enumerate(thetas):
                eval_at_polar(
                    float(r_ring),
                    float(theta),
                    f"seedR{si+1}",
                    f"seed {si+1}/{n_seed} r={r_ring:g} ang={ai+1}/{n_ring}",
                    r_ring,
                )
            if model is not None and n_hole > 0:
                su, sv, ss, _sr = model.chart.points(b)
                ell = length_scale(
                    su,
                    sv,
                    r_step=r_step,
                    r_seed=max(r_disk, 4.0),
                    n_seed_ang=max(ring_angs + [n_ang]),
                )
                holes = disk_holes(
                    su,
                    sv,
                    ss,
                    r_ring,
                    ell,
                    n_max=n_hole,
                    r_step=r_step,
                )
                for hi, (lu, lv) in enumerate(holes):
                    eval_at_plane(
                        lu,
                        lv,
                        f"seedH{si+1}",
                        f"seed {si+1}/{n_seed} r={r_ring:g} hole={hi+1}/{len(holes)}",
                        r_ring,
                    )
                if holes:
                    say(f"    disk gaps direct {len(holes)}  r={r_ring:g}")
            compute_orange_bests(r_ring)
            rr = int(round(r_ring))
            if n_seed == 1 and abs(r_ring - early_rings[0]) < 1e-9:
                tag = "Reveal_start"
            elif n_seed == 1:
                tag = f"Reveal_start_r{rr:03d}"
            else:
                tag = f"Reveal_start_s{si+1:02d}_r{rr:03d}"
            snap_reveal(
                cfg,
                model,
                tag,
                log=log,
                note=f"start r={r_ring:g}",
                r=r_ring,
            )
            say(f"    seed circle r={r_ring:g}  direct {n_ring} angles  n={len(peaks)}")
            ch_snap = _champ_for_best(peaks, cfg, anchor, r_ring)
            if ch_snap is not None:
                snap_best(cfg, ch_snap, r_ring)
            mark_fracs(r_ring)
            if model is not None:
                model.chart.set_r_probed(float(r_ring))

        ch0 = _champ(peaks, cfg)
        best = float(ch0.score or -1e18) if ch0 is not None else -1e18
        if model is not None:
            model.chart.set_r_probed(float(r_disk))
            label_disk_points(peaks, model.chart)
            chains = reconnect_by_pattern(model.chart, HEADING_SEP)
            roots = start_from_chains(chains, peaks, model.chart, root_max)
        else:
            roots = []
        if not roots:
            roots = start_from_ring(peaks, r_disk, root_max, anchor=anchor)
        if model is not None:
            snap_reveal(
                cfg,
                model,
                "Reveal_start" if n_seed == 1 else f"Reveal_start_s{si+1:02d}",
                log=log,
                note=f"disk end trend line r={r_disk:g}",
                r=r_disk,
            )
        n_alive = len(roots)
        next_rid = 1 + max((rt.rid for rt in roots), default=0)
        say(
            f"    trend lines start {len(roots)}/{root_max}  "
            + " ".join(f"{rt.key}#{rt.rid}@{rt.theta:.2f}" for rt in roots)
        )
        if not roots:
            say("    no trend lines → evolve from seeds only")

        def stamp(ind: Individual, key: str, rid: int) -> Individual:
            kid = clone_ind(ind)
            kid.chart_kind = key
            kid.chart_root = int(rid)
            if (
                model is not None
                and kid.score is not None
                and is_feasible(kid, cfg)
            ):
                model.chart.observe(
                    kid.metal,
                    float(kid.score),
                    uv=kid.chart_uv,
                    kind=key,
                    d_use=kid.d_use,
                    eta=kid.eta,
                    root=int(rid),
                    d_ap=getattr(kid, "d_ap", None),
                    d_peak=getattr(kid, "leave_db", getattr(kid, "d_peak", None)),
                    leave=getattr(kid, "leave_db", None),
                    beam_db=getattr(kid, "beam_db", None),
                )
            return kid

        for rt in roots:
            stamp(rt.tip, rt.key, rt.rid)
        n_wave = int(round((r_max - r_disk) / max(r_step, 1e-6))) + root_max
        n_wave = max(n_wave, 1)

        def plant(
            r: float,
            theta: float,
            key: str,
            rid: int,
            label: str,
            *,
            verify_neighbors: bool = True,
        ) -> Individual:
            kid = spiral.individual_from_polar(
                r, theta, mode=seed.mode, src=f"root:{key}:{rid}"
            )
            if kid is None:
                return None
            kid.chart_kind = key
            kid.chart_root = int(rid)
            tick(label, r=r)
            kid = climb(kid)
            attach_spiral_sample(
                kid,
                int(round(float(kid.chart_r or r))),
                int(kid.chart_s or 0),
                float(theta),
                spiral,
            )
            kid.chart_kind = key
            kid.chart_root = int(rid)
            tick(label, inc=True, r=r)
            if verify_neighbors:
                ref = _probe_val(kid, key if key in KEYS else None)
                verify_peak_rs8(
                    int(round(float(kid.chart_r or r))),
                    int(kid.chart_s or 0),
                    ref,
                    f"root{key}{rid}",
                    f"trend line {key}#{rid} r={r:g}",
                    r,
                    metric_key=key if key in KEYS else None,
                )
            return kid

        def chart_xy():
            su_ = sv_ = ss_ = np.zeros(0, dtype=np.float64)
            if model is not None:
                return model.chart.points(b)
            return su_, sv_, ss_, su_

        for wave in range(1, n_wave + 1):
            live = [rt for rt in roots if rt.alive]
            if not live:
                say("  no live trend lines → exploration wave done")
                break
            r_tip = max(float(rt.r) for rt in live)
            if r_tip >= float(r_max) - 1e-6:
                say(f"  r_max={r_max:g} reached → exploration done")
                for rt in live:
                    rt.alive = False
                break
            if model is not None and (wave == 1 or wave % 3 == 0):
                chains_pre = reconnect_by_pattern(model.chart, HEADING_SEP)
                rematch_live_roots_to_chains(roots, chains_pre, log=say)
            b = None if model is None else getattr(model, "b", None)
            su, sv, ss, _sr = chart_xy()
            ell = length_scale(
                su, sv, r_step=r_step, r_seed=r_disk, n_seed_ang=max(ring_angs + [n_ang])
            )
            series = {
                k: (model.chart.series_at_points(k) if model is not None else ss) for k in KEYS
            }
            r_front = min(r_max, max(rt.r for rt in live) + r_step)
            n_alive = sum(1 for rt in roots if rt.alive)
            tick(
                f"seed {si+1}/{n_seed} live {n_alive}/{root_max} wave {wave}",
                r=r_front,
            )
            say(
                f"    wave {wave}  live {len(live)}/{len(roots)}  r→{r_front:g}"
            )
            stepped: list = []
            split_done = False
            for rt in live:
                nxt = rt.r + r_step * float(rt.speed)
                if nxt > r_max + 1e-9:
                    rt.alive = False
                    say(f"    root{rt.rid} {rt.key}  r cap")
                    continue
                su, sv, ss, _sr = chart_xy()
                series = {
                    k: (model.chart.series_at_points(k) if model is not None else ss) for k in KEYS
                }
                twin = next(
                    (
                        x
                        for x in stepped
                        if abs(x.r - nxt) < 0.51
                        and abs(np.arctan2(np.sin(x.theta - rt.theta), np.cos(x.theta - rt.theta)))
                        < 0.40
                    ),
                    None,
                )
                if twin is not None and twin.tip is not None:
                    kid = stamp(twin.tip, rt.key, rt.rid)
                    peaks.append(kid)
                    val = metric_val(kid, rt.key)
                    rt.theta = twin.theta
                    rt.r = twin.r
                    rt.omega = twin.omega
                    rt.speed = twin.speed
                    rt.tip = clone_ind(kid)
                    if kid.chart_uv is not None:
                        rt.path.append(_path_local(kid.chart_uv))
                    if val > rt.best + 1e-9:
                        rt.best = val
                        rt.stall = 0
                    else:
                        rt.stall += 1
                        if rt.stall >= stall_need:
                            rt.alive = False
                            say(f"    root{rt.rid} {rt.key}  done")
                    say(f"    root{rt.rid} {rt.key}  together r={rt.r:g}")
                    stepped.append(rt)
                    continue
                avoid = [
                    x.theta
                    for x in roots
                    if x.alive
                    and x.rid != rt.rid
                    and abs(np.arctan2(np.sin(x.theta - rt.theta), np.cos(x.theta - rt.theta)))
                    >= 0.40
                ]
                th, acq = pick_inertial_probe(
                    nxt,
                    heading_of(rt),
                    rt.omega,
                    su,
                    sv,
                    series.get(rt.key, ss),
                    ell=ell,
                    n_cand=n_cand,
                    half=cone_half(rt),
                    rng=rng,
                    avoid=avoid,
                )
                if acq < 0.0:
                    rt.stall += 1
                    say(
                        f"    root{rt.rid} {rt.key}  no neighbors  "
                        f"stall {rt.stall}/{stall_need}"
                    )
                    if rt.stall >= stall_need:
                        rt.alive = False
                        say(f"    root{rt.rid} {rt.key}  done")
                    continue
                kid = plant(
                    nxt,
                    th,
                    rt.key,
                    rt.rid,
                    f"seed {si+1}/{n_seed} r={nxt:g} {rt.key}#{rt.rid}",
                )
                if not is_feasible(kid, cfg):
                    rt.stall += 1
                    say(
                        f"    root{rt.rid} {rt.key}  infeasible  "
                        f"stall {rt.stall}/{stall_need}"
                    )
                    if rt.stall >= stall_need:
                        rt.alive = False
                    stepped.append(rt)
                    continue
                peaks.append(kid)
                val = metric_val(kid, rt.key)
                uv = kid.chart_uv
                if (
                    model is not None
                    and uv is not None
                    and breaks_score_trend(model.chart, rt.key, rt.rid, uv, val)
                ):
                    model.chart.mark_kind(kid.metal, "seed")
                    kid.chart_kind = "seed"
                    rt.alive = False
                    say(f"    root{rt.rid} {rt.key}  score trend broken → direct FDTD  r={nxt:g}")
                    best = max(best, kid.score or -1e18)
                    stepped.append(rt)
                    continue
                improved = val > rt.best + 1e-9
                apply_steer(rt, th, improved)
                rt.r = nxt
                rt.tip = clone_ind(kid)
                if kid.chart_uv is not None:
                    rt.path.append(_path_local(kid.chart_uv))
                if improved:
                    say(
                        f"    root{rt.rid} {rt.key}  updated {rt.best:.2f}→{val:.2f}  "
                        f"r={nxt:g}  v={rt.speed:.2f}"
                    )
                    rt.best = val
                    rt.stall = 0
                else:
                    rt.stall += 1
                    say(
                        f"    root{rt.rid} {rt.key}  stall {rt.stall}/{stall_need}  "
                        f"val={val:.2f}  best={rt.best:.2f}  v={rt.speed:.2f}"
                    )
                    if rt.stall >= stall_need:
                        rt.alive = False
                        say(f"    root{rt.rid} {rt.key}  done")
                best = max(best, kid.score or -1e18)
                stepped.append(rt)
                if (
                    not split_done
                    and rt.alive
                    and len(roots) < root_max
                ):
                    su, sv, ss, _sr = chart_xy()
                    series = {
                        k: (model.chart.series_at_points(k) if model is not None else ss) for k in KEYS
                    }
                    owned = same_key_headings(roots, rt.key)
                    split = pick_heading_split(
                        rt.r,
                        heading_of(rt),
                        su,
                        sv,
                        series.get(rt.key, ss),
                        ell=ell,
                        owned=owned,
                        sep=HEADING_SEP,
                    )
                    if split is not None:
                        th_f, acq_f, side = split
                        stamp(rt.tip, rt.key, next_rid)
                        child = Root(
                            rid=next_rid,
                            key=rt.key,
                            theta=th_f,
                            r=rt.r,
                            tip=clone_ind(rt.tip),
                            best=rt.best,
                            omega=0.12 * float(side),
                            speed=max(0.70, 0.85 * rt.speed),
                            path=[
                                _path_local(rt.tip.chart_uv)
                                if rt.tip.chart_uv is not None
                                else (
                                    float(rt.r * np.cos(rt.theta)),
                                    float(rt.r * np.sin(rt.theta)),
                                )
                            ],
                        )
                        roots.append(child)
                        say(
                            f"    split root{next_rid} {rt.key} "
                            f"{th_f:.2f}  A={acq_f:.2f}  n={len(roots)}/{root_max}"
                        )
                        next_rid += 1
                        split_done = True

            su, sv, ss, _sr = chart_xy()
            series = {
                k: (model.chart.series_at_points(k) if model is not None else ss) for k in KEYS
            }
            r_fork = min(r_max, max((rt.r for rt in roots), default=r_disk))
            if r_fork < r_disk + 0.5 * r_step:
                r_fork = min(r_max, r_disk + r_step)
            used = [rt.theta for rt in roots if rt.alive]
            if (not split_done) and len(roots) < root_max:
                found = pick_fork_theta(
                    r_fork,
                    su,
                    sv,
                    ss,
                    ell=ell,
                    n_cand=n_cand,
                    rng=rng,
                    used=used,
                    min_sep=0.70,
                    min_acq=0.50,
                )
                if found is not None:
                    th, acq = found
                    lu, lv = _polar_local(r_fork, th)
                    key = assign_metric_key(lu, lv, su, sv, series, ell)
                    kid = plant(
                        r_fork,
                        th,
                        key,
                        next_rid,
                        f"seed {si+1}/{n_seed} fork {key}#{next_rid}",
                    )
                    peaks.append(kid)
                    best = max(best, kid.score or -1e18)
                    uv = kid.chart_uv or spiral.plane_xy(r_fork, th)
                    roots.append(
                        Root(
                            rid=next_rid,
                            key=key,
                            theta=th,
                            r=r_fork,
                            tip=clone_ind(kid),
                            best=metric_val(kid, key),
                            path=[(0.0, 0.0), _path_local(uv)],
                        )
                    )
                    say(
                        f"    branch root{next_rid} {key} @{th:.2f}  "
                        f"A={acq:.2f}  n={len(roots)}/{root_max}"
                    )
                    next_rid += 1

            r_now = max((rt.r for rt in roots), default=r_disk)
            if model is not None:
                chains = reconnect_by_pattern(model.chart, HEADING_SEP)
                su, sv, ss, _sr = chart_xy()
                uu, vv, _, rr = model.chart.points()
                n_trend = 0
                for key_t, r_t, th_t, rid_t in next_trend_probes(
                    chains, uu, vv, rr, r_max, r_step=r_step
                ):
                    if su.size:
                        lu, lv = _polar_local(float(r_t), float(th_t))
                        dmin = float(np.min(np.hypot(lu - su, lv - sv)))
                        if dmin < 0.45 * ell:
                            continue
                    kid = plant(
                        r_t,
                        th_t,
                        key_t,
                        rid_t,
                        f"seed {si+1}/{n_seed} trend {key_t} r={r_t:g}",
                        verify_neighbors=False,
                    )
                    if is_feasible(kid, cfg):
                        peaks.append(kid)
                        best = max(best, kid.score or -1e18)
                    uv = kid.chart_uv or spiral.plane_xy(float(r_t), float(th_t))
                    val = metric_val(kid, key_t)
                    if breaks_score_trend(model.chart, key_t, rid_t, uv, val):
                        model.chart.mark_kind(kid.metal, "seed")
                        kid.chart_kind = "seed"
                        say(
                            f"    trend X eval  score mismatch → direct FDTD  "
                            f"{key_t} r={r_t:g}"
                        )
                    else:
                        say(
                            f"    trend {key_t}  r={r_t:g}  θ={th_t:.2f}  "
                            f"{key_t}={val:.2f}"
                        )
                    next_rid += 1
                    r_now = max(r_now, r_t)
                    if n_trend == 0 or n_trend % 2 == 0:
                        reconnect_by_pattern(model.chart, HEADING_SEP)
                    n_trend += 1
                    if n_trend >= 2:
                        break
            n_chart = int(model.chart.n()) if model is not None else 0
            say(
                f"    wave {wave} wrap-up  r={r_now:g}  chart={n_chart}  "
                f"inferred (orange), reveal…"
            )
            compute_orange_bests(max(r_now, r_disk))
            snap_reveal(
                cfg,
                model,
                f"Reveal_w_s{si+1:02d}_r{int(round(r_now)):03d}",
                log=log,
                note=f"r={r_now:g} trend lines {sum(1 for rt in roots if rt.alive)}",
                r=r_now,
            )
            ch_w = _champ_for_best(peaks, cfg, anchor, r_now)
            if ch_w is not None:
                snap_best(cfg, ch_w, r_now)
            mark_fracs(r_now)
            if not any(rt.alive for rt in roots):
                nxt_cov = min(r_max, r_now + r_step)
                if nxt_cov > r_now + 1e-9:
                    used = [rt.theta for rt in roots]
                    found = pick_fork_theta(
                        nxt_cov,
                        su,
                        sv,
                        ss,
                        ell=ell,
                        n_cand=n_cand,
                        rng=rng,
                        used=used,
                        min_sep=0.70,
                        min_acq=0.0,
                    )
                    th = found[0] if found else float(rng.random() * 2.0 * np.pi)
                    lu, lv = _polar_local(nxt_cov, th)
                    key = assign_metric_key(lu, lv, su, sv, series, ell)
                    dead = next((rt for rt in roots if not rt.alive), None)
                    rid = dead.rid if dead is not None else next_rid
                    kid = plant(
                        nxt_cov,
                        th,
                        key,
                        rid,
                        f"seed {si+1}/{n_seed} cover {key}#{rid}",
                    )
                    peaks.append(kid)
                    best = max(best, kid.score or -1e18)
                    if dead is not None:
                        dead.alive = True
                        dead.stall = 0
                        dead.key = key
                        dead.theta = th
                        dead.r = nxt_cov
                        dead.tip = clone_ind(kid)
                        dead.best = metric_val(kid, key)
                    say(f"    cover r={nxt_cov:g} {key}#{rid}  (up to {int(round(100.0*r_max/r_full))}% of map)")
                else:
                    snap_reveal(
                        cfg,
                        model,
                        f"Reveal_stagnate_s{si+1:02d}",
                        log=log,
                        note=f"trend line end r={r_now:g}",
                        r=r_now,
                    )
                    say(f"  trend lines done → hand off to evolution  best={best:.2f}")
                    break

        if any(rt.alive for rt in roots):
            r_now = max((rt.r for rt in roots), default=r_disk)
            snap_reveal(
                cfg,
                model,
                f"Reveal_stagnate_s{si+1:02d}",
                log=log,
                note=f"limit r={r_now:g}",
                r=r_now,
            )
            say(f"  r/wave limit → hand off to evolution  best={best:.2f}")

        if chart_diag_skip:
            say(
                f"  chart diagonal/range skip  seed {si+1}/{n_seed}  "
                f"n={chart_diag_skip}"
            )

    scored = [
        p for p in peaks if p.score is not None and is_feasible(p, cfg)
    ]
    if getattr(cfg, "circle_light_only", False):
        scored.sort(key=lambda x: x.score if x.score is not None else -1e18, reverse=True)
        best = scored[0] if scored else (peaks[0] if peaks else None)
        if best is None:
            say("  exploration done  no feasible")
            return []
        say(
            f"  exploration done  evals={len(scored)}  best={best.score:.2f}  "
            f"(seed-circle map, no evolution)"
        )
        return scored
    picked = _diverse_pick(peaks, n_pop, cfg)
    if not picked:
        picked = scored[:n_pop]
    best = max(picked, key=lambda x: x.score or -1e18)
    kinds = " ".join(
        f"{getattr(p, 'chart_kind', '?')}={p.score:.2f}" for p in picked[:6]
    )
    say(
        f"  exploration done → evolution  peaks={len(peaks)}  pop={len(picked)}  "
        f"best={best.score:.2f}  parents={kinds}"
    )
    return picked

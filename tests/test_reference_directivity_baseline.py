# -*- coding: utf-8 -*-
"""Gain reference for evolution/warm-up comparison: one Vivaldi/bowtie-style "example".

Expected: a baseline tuned with FDTD from a typical directive shape (Vivaldi/bowtie/horn).
Goal: chart/GA finds a better map that **beats this example's score**.

Only the **directive template family** (and its mutate/climb) can take first place. Random blobs are logged for comparison only.

Run:
  py tests/test_reference_directivity_baseline.py

Env:
  REF_BASELINE_SKIP_GATE=1 (default)
  REF_BASELINE_MUTATE=12   mutations per top-ranked template
  REF_BASELINE_CLIMB=14
  REF_BASELINE_RANDOM=0    >0 adds random shapes for reference only (cannot rank first)
  REF_BASELINE_SEED=42
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_TESTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from core import CENTER, Individual, seed_random  # noqa: E402
from directive_shape_templates import directive_template_catalog  # noqa: E402
from main import make_config, make_solver  # noqa: E402
from opt_ga import (  # noqa: E402
    _METAL_EVAL_CACHE,
    _METAL_FIELD_SIG,
    fill_eval,
    gain_parts,
    match_efficiency,
    match_efficiency_min,
    mutate,
    reset_eval_cache,
    save_plot,
)

TAG = "Reference_gain_baseline"
OUT_SUBDIR = "_reference_directivity"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _metal_key(metal: np.ndarray) -> bytes:
    return np.asarray(metal, dtype=np.uint8).tobytes()


def _is_filled_axis_rect(metal: np.ndarray) -> bool:
    m = np.asarray(metal, dtype=np.uint8)
    ys, xs = np.nonzero(m)
    if ys.size < 24:
        return False
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    sub = m[y0 : y1 + 1, x0 : x1 + 1]
    if sub.size != int(m.sum()):
        return False
    return float(sub.mean()) >= 0.98 and sub.shape[0] >= 7 and sub.shape[1] >= 7


def _metrics_from_ind(ind: Individual, freqs, cfg) -> dict:
    cone_db, _eta, _eta_db, _beam, _rr, copper, _dap, _b2, face6_db, leave_db, leave_r, _em = gain_parts(
        ind.s11,
        freqs,
        cfg,
        air=ind.air,
        metal=ind.metal,
        faces=getattr(ind, "faces", None),
        leave=getattr(ind, "leave", None),
    )
    return {
        "score": float(ind.score or -1e18),
        "worst_s11_db": float(ind.worst or 0.0),
        "cone_db": float(cone_db),
        "face6_db": float(face6_db),
        "leave_db": float(leave_db),
        "leave_r": float(leave_r),
        "eta_avg": float(match_efficiency(ind.s11, freqs, cfg)),
        "eta_min": float(match_efficiency_min(ind.s11, freqs, cfg)),
        "copper_px": int(copper),
        "is_gate_rect": bool(_is_filled_axis_rect(ind.metal)),
    }


def _eligible_for_baseline(m: dict, cfg) -> bool:
    tgt = float(cfg.s11_target_db)
    min_cu = int(getattr(cfg, "best_min_upper_copper", 0) or 0)
    if m["worst_s11_db"] > tgt + 1e-9:
        return False
    if min_cu > 0 and m["copper_px"] < min_cu:
        return False
    if m.get("is_gate_rect"):
        return False
    return True


def _sort_key(row: dict) -> tuple:
    m = row["metrics"]
    bar = 1 if _eligible_for_baseline(m, row["cfg"]) else 0
    lmin = float(row["cfg"].leave_db_min)
    lmax = float(row["cfg"].leave_db_max)
    leave_ok = 1 if lmin <= m["leave_db"] <= lmax else 0
    return (
        bar,
        m["score"],
        leave_ok,
        m["eta_min"],
        m["cone_db"] + 0.5 * m["face6_db"],
    )


def _eval_one(
    solver,
    metal: np.ndarray,
    name: str,
    freqs,
    cfg,
    cache: dict[bytes, dict],
    *,
    directive_lineage: bool,
) -> dict | None:
    key = _metal_key(metal)
    if key in cache:
        return cache[key]
    ind = Individual(np.asarray(metal, dtype=np.uint8).copy(), mode=-1, src=name)
    fill_eval(solver, ind, freqs, cfg, note=name, log_chart=False)
    if ind.score is None or ind.s11 is None:
        return None
    met = _metrics_from_ind(ind, freqs, cfg)
    row = {
        "name": name,
        "ind": ind,
        "metrics": met,
        "cfg": cfg,
        "directive_lineage": bool(directive_lineage),
    }
    cache[key] = row
    return row


def _climb(
    solver,
    ind: Individual,
    freqs,
    cfg,
    rng: np.random.Generator,
    feed,
    steps: int,
    cache: dict[bytes, dict],
    tag: str,
) -> Individual:
    best = ind
    best_sc = float(ind.score or -1e18)
    for t in range(steps):
        cand = mutate(
            best,
            rng,
            feed,
            mut_pixels=6,
            diffuse_steps=cfg.diffuse_steps,
            diffuse_rate=cfg.diffuse_rate,
            cfg=cfg,
        )
        row = _eval_one(
            solver,
            cand.metal,
            f"{tag}_climb_{t:02d}",
            freqs,
            cfg,
            cache,
            directive_lineage=True,
        )
        if row is None:
            continue
        sc = row["metrics"]["score"]
        if sc > best_sc + 1e-9:
            best_sc = sc
            best = row["ind"]
    return best


def search_gain_baseline(
    solver,
    cfg,
    freqs,
    rng: np.random.Generator,
    *,
    n_mutate: int,
    n_climb: int,
    n_random: int,
) -> tuple[list[dict], list[dict]]:
    feed = cfg.feed
    cache: dict[bytes, dict] = {}
    directive_rows: list[dict] = []
    random_rows: list[dict] = []

    templates = directive_template_catalog(feed)
    print(f"  {len(templates)} directive templates (Vivaldi/bowtie/horn/CPW) …")
    for name, metal in templates:
        row = _eval_one(
            solver, metal, name, freqs, cfg, cache, directive_lineage=True
        )
        if row is not None:
            directive_rows.append(row)

    directive_rows.sort(key=_sort_key, reverse=True)
    elites = directive_rows[: max(4, min(6, len(directive_rows)))]

    for ei, base in enumerate(elites):
        for m in range(n_mutate):
            kid = mutate(
                base["ind"],
                rng,
                feed,
                mut_pixels=cfg.mut_pixels,
                diffuse_steps=cfg.diffuse_steps,
                diffuse_rate=cfg.diffuse_rate,
                cfg=cfg,
            )
            name = f"{base['name']}_mut{m:02d}"
            row = _eval_one(
                solver, kid.metal, name, freqs, cfg, cache, directive_lineage=True
            )
            if row is not None:
                directive_rows.append(row)

    directive_rows.sort(key=_sort_key, reverse=True)
    if directive_rows:
        champ = _climb(
            solver,
            directive_rows[0]["ind"],
            freqs,
            cfg,
            rng,
            feed,
            n_climb,
            cache,
            tag=str(directive_rows[0]["name"])[:24],
        )
        top = _eval_one(
            solver,
            champ.metal,
            "directive_climb_best",
            freqs,
            cfg,
            cache,
            directive_lineage=True,
        )
        if top is not None:
            directive_rows.append(top)

    for k in range(n_random):
        seed = seed_random(rng, feed, mode=-1)
        row = _eval_one(
            solver,
            seed.metal,
            f"random_ref_{k:02d}",
            freqs,
            cfg,
            cache,
            directive_lineage=False,
        )
        if row is not None:
            random_rows.append(row)

    directive_rows.sort(key=_sort_key, reverse=True)
    random_rows.sort(key=_sort_key, reverse=True)
    return directive_rows, random_rows


def pick_and_save_baseline(
    out_root: Path | None = None,
    *,
    backend: str | None = None,
    skip_gate: bool | None = None,
) -> dict:
    cfg = make_config()
    if backend:
        cfg.backend = backend.strip().lower()
    base = out_root or (cfg.out_dir / OUT_SUBDIR)
    base.mkdir(parents=True, exist_ok=True)
    cfg.out_dir = base
    cfg.byproduct_dir = base / "byproduct"

    if skip_gate is None:
        skip_gate = os.environ.get("REF_BASELINE_SKIP_GATE", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
    if skip_gate:
        print("  openEMS gate skipped (REF_BASELINE_SKIP_GATE=1).")
    else:
        from openems.gate import ensure_before_optimize

        ensure_before_optimize(cfg.backend)

    n_mut = _env_int("REF_BASELINE_MUTATE", 12)
    n_climb = _env_int("REF_BASELINE_CLIMB", 14)
    n_random = _env_int("REF_BASELINE_RANDOM", 0)
    seed = _env_int("REF_BASELINE_SEED", 42)
    rng = np.random.default_rng(seed)

    reset_eval_cache()
    _METAL_EVAL_CACHE.clear()
    _METAL_FIELD_SIG.clear()

    freqs = np.linspace(cfg.f_start, cfg.f_stop, cfg.n_freq)
    feed = cfg.feed

    print(
        f"  gain reference (Vivaldi-like)  mutate×{n_mut}/elite  climb={n_climb}  "
        f"random_ref={n_random}  fitness={cfg.fitness}  backend={cfg.backend}"
    )

    solver = make_solver(cfg.backend)
    try:
        directive_rows, random_rows = search_gain_baseline(
            solver,
            cfg,
            freqs,
            rng,
            n_mutate=n_mut,
            n_climb=n_climb,
            n_random=n_random,
        )
    finally:
        solver.close()

    if not directive_rows:
        raise RuntimeError("no directive template FDTD candidates")

    best_row = directive_rows[0]
    best = best_row["metrics"]
    ind = best_row["ind"]
    eligible = _eligible_for_baseline(best, cfg)

    print("  --- directive top ---")
    for line in directive_rows[:8]:
        m = line["metrics"]
        ok = "✓" if _eligible_for_baseline(m, cfg) else "·"
        print(
            f"    {ok} {line['name'][:28]:28s}  score={m['score']:.2f}  "
            f"cone={m['cone_db']:.1f}  face6={m['face6_db']:.1f}  "
            f"worst={m['worst_s11_db']:.1f}  η_min={100 * m['eta_min']:.0f}%  cu={m['copper_px']}"
        )
    if random_rows:
        r0 = random_rows[0]["metrics"]
        print(
            f"  (ref) random top score={r0['score']:.2f}  "
            f"— baseline top is Vivaldi-like only"
        )

    by = cfg.byproduct_dir / "metal"
    by.mkdir(parents=True, exist_ok=True)
    np.save(by / f"{TAG}.npy", np.asarray(ind.metal, dtype=np.uint8))

    save_plot(
        cfg.out_dir,
        TAG,
        ind.metal,
        ind.s11,
        freqs,
        ind.score,
        ind.worst,
        feed,
        air=ind.air,
        cut=ind.cut,
        cfg=cfg,
        faces=getattr(ind, "faces", None),
        field_extra=getattr(ind, "field_extra", None),
    )
    jpg = cfg.out_dir / f"{TAG}.jpg"
    assert jpg.is_file() and jpg.stat().st_size > 5000, jpg

    payload = {
        "tag": TAG,
        "family": "vivaldi_bowtie_horn_templates",
        "winner": best_row["name"],
        "eligible_bar": eligible,
        "search": {
            "mutate_per_elite": n_mut,
            "climb": n_climb,
            "random_ref_only": n_random,
            "rng_seed": seed,
        },
        "feed_ij": [int(feed[0]), int(feed[1])],
        "feed_mm": [float(feed[0] - CENTER), float(feed[1] - CENTER)],
        "backend": cfg.backend,
        "fitness": cfg.fitness,
        "band_ghz": [cfg.band_hz[0] / 1e9, cfg.band_hz[1] / 1e9],
        "baseline": {**best, "name": best_row["name"]},
        "directive_top8": [
            {**r["metrics"], "name": r["name"], "eligible": _eligible_for_baseline(r["metrics"], cfg)}
            for r in directive_rows[:8]
        ],
        "compare_hint": (
            "If Warm-up/GA Best score is at or above baseline, it beats textbook directive examples like Vivaldi, bowtie. "
            "The evolution goal is a better non-template map."
        ),
    }
    if random_rows:
        payload["random_ref_top"] = {
            **random_rows[0]["metrics"],
            "name": random_rows[0]["name"],
        }
    if not eligible:
        payload["warning"] = "Top entry misses S11/copper bar, increase mutate/climb or add template params."

    metrics_path = cfg.out_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=== gain reference (Vivaldi-like examples) ===")
    print(f"  top: {best_row['name']}  eligible={eligible}")
    print(f"  JPG: {jpg}")
    print(
        f"  to beat: score≥{best['score']:.2f}  cone≥{best['cone_db']:.2f}  face6≥{best['face6_db']:.2f}  "
        f"worst≤{best['worst_s11_db']:.1f} dB  η_min≥{100 * best['eta_min']:.0f}%  copper={best['copper_px']} px"
    )
    return payload


def test_reference_gain_baseline():
    cfg = make_config()
    out = cfg.out_dir / OUT_SUBDIR
    payload = pick_and_save_baseline(out)
    jpg = out / f"{TAG}.jpg"
    assert jpg.is_file() and jpg.stat().st_size > 5000
    assert payload["family"] == "vivaldi_bowtie_horn_templates"
    assert payload["baseline"]["score"] > -1e6


def main():
    out = os.environ.get("REF_BASELINE_OUT", "").strip()
    out_path = Path(out) if out else None
    be = os.environ.get("REF_BASELINE_BACKEND", "").strip() or None
    pick_and_save_baseline(out_path, backend=be)


if __name__ == "__main__":
    main()

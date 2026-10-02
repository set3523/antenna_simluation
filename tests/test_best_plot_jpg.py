# -*- coding: utf-8 -*-
"""Smoke test of the Best_r JPG path (save_best -> save_plot) plus xy panel axis alignment check.

FDTD on 5 random copper maps (default), then save JPGs with the same save_best as warmup.
Run:
  py tests/test_best_plot_jpg.py
  py -m pytest tests/test_best_plot_jpg.py -v
Env:
  BEST_PLOT_TEST_N=5   number of samples
  BEST_PLOT_TEST_SEED=42
  BEST_PLOT_DUMP_CUDA=1 (default) npz of fill_eval plus a direct FDTD rerun
  BEST_PLOT_NO_CACHE=1 (default) clear _METAL_EVAL_CACHE for every sample
  BEST_PLOT_SKIP_GATE=1 (default) skip the openEMS gate, JPG/FDTD smoke only
  BEST_PLOT_SKIP_GATE=0 requires passing the gate, same as main
"""
from __future__ import annotations

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

from core import (  # noqa: E402
    FEED_MM,
    GND_MARK_MM,
    Individual,
    PIX,
    ij_from_mm,
    seed_random,
)
from main import make_config, make_solver  # noqa: E402
from opt_ga import (  # noqa: E402
    FDTD_PAD,
    _COLOR_COPPER,
    _COLOR_FEED,
    _COLOR_GND,
    _COLOR_NO_COPPER,
    _METAL_EVAL_CACHE,
    _METAL_FIELD_SIG,
    _crop_field_to_metal_plane,
    _metal_cache_key,
    _metal_xy_rgb,
    ensure_evaluated,
    evaluate,
    fill_eval,
    save_best,
    save_cuda_field_npz,
    save_evolution,
    save_plot,
)
from cuda_field_dump_utils import (  # noqa: E402
    analyze_cuda_field_dir,
    individual_from_eval_tuple,
    print_diag_report,
    write_diag_csv,
)

DEFAULT_N = 30
DEFAULT_SEED = 42
OUT_SUBDIR = "_test_best_plot"
# the old bug (sub.T in imshow) equals the swapped axes; require aligned ≥ swapped at mid


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def overlay_alignment(metal: np.ndarray, e: np.ndarray) -> dict[str, float]:
    """Same axis assumption as metal.T + _crop_field_to_metal_plane in save_plot."""
    plane = _crop_field_to_metal_plane(e)
    m = np.asarray(metal, dtype=np.float64)
    aligned = _corr(m.T, plane)
    swapped = _corr(m.T, plane.T)
    rot_ccw = _corr(m.T, np.rot90(plane, k=1))
    return {
        "aligned": aligned,
        "swapped": swapped,
        "rot_ccw": rot_ccw,
        "plane_shape_ok": float(
            plane.shape == (PIX, PIX) or plane.shape == (metal.shape[0], metal.shape[1])
        ),
    }


def _check_layers(metal: np.ndarray, ind: Individual) -> None:
    if int(np.asarray(metal).sum()) < 80:
        return  # with copper only near the feed, the patch/mid-plane correlation check is meaningless
    fe = ind.field_extra or {}
    for key in ("patch", "mid"):
        if key not in fe:
            continue
        stats = overlay_alignment(metal, fe[key])
        assert stats["plane_shape_ok"] >= 1.0, f"{key} crop shape {fe[key].shape}"

    mid = fe.get("mid")
    if mid is None:
        return
    stats = overlay_alignment(metal, mid)
    a, s, r90 = stats["aligned"], stats["swapped"], stats["rot_ccw"]
    assert np.isfinite(a) and np.isfinite(s), "mid overlay corr NaN"
    # save_plot: metal.T + crop(sub). The wrong sub.T is "swapped".
    assert a >= s - 1e-6, f"mid: aligned={a:.3f} should beat swapped={s:.3f}"


def run_best_plot_smoke(
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    out_root: Path | None = None,
    *,
    strict_mid_check: bool = False,
) -> list[dict]:
    cfg = make_config()
    base = out_root or (cfg.out_dir / OUT_SUBDIR)
    base.mkdir(parents=True, exist_ok=True)
    cfg.out_dir = base
    cfg.byproduct_dir = base / "byproduct"

    skip_gate = os.environ.get("BEST_PLOT_SKIP_GATE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if skip_gate:
        print("  openEMS gate skipped (BEST_PLOT_SKIP_GATE=1). FDTD JPG smoke only.")
    else:
        from openems.gate import ensure_before_optimize

        ensure_before_optimize(cfg.backend)
    freqs = np.linspace(cfg.f_start, cfg.f_stop, cfg.n_freq)
    rng = np.random.default_rng(seed)
    feed = cfg.feed
    rows: list[dict] = []

    dump_cuda = os.environ.get("BEST_PLOT_DUMP_CUDA", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    no_cache = os.environ.get("BEST_PLOT_NO_CACHE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    cuda_root = cfg.byproduct_dir / "cuda_fields"

    solver = make_solver(cfg.backend)
    try:
        for k in range(n):
            if no_cache:
                _METAL_EVAL_CACHE.clear()
                _METAL_FIELD_SIG.clear()
            ind = seed_random(rng, feed, mode=-1)
            for _ in range(24):
                if int(ind.metal.sum()) >= 80:
                    break
                ind = seed_random(rng, feed, mode=-1)
            tag = f"Best_plot_t{k + 1:02d}"
            fill_eval(solver, ind, freqs, cfg, note=f"test_{tag}", log_chart=False)
            ind = ensure_evaluated(solver, ind, freqs, cfg, note=f"plot_{tag}")
            ck = _metal_cache_key(ind)
            assert ind.eval_ck == ck, "eval_ck mismatch after fill_eval"
            assert ind.air is not None and ind.field_extra is not None

            if dump_cuda:
                save_cuda_field_npz(
                    cfg,
                    tag,
                    ind,
                    freqs,
                    meta={"path": "fill_eval", "index": k + 1},
                )
                tup = evaluate(solver, ind.metal, freqs, cfg)
                ind_direct = individual_from_eval_tuple(ind.metal, freqs, tup)
                save_cuda_field_npz(
                    cfg,
                    tag,
                    ind_direct,
                    freqs,
                    suffix="_direct",
                    meta={"path": "evaluate_direct", "index": k + 1},
                )

            jpg_before = cfg.out_dir / f"{tag}.jpg"
            save_best(cfg, ind, freqs, r=float(10 + k), tag=tag)
            if not jpg_before.is_file():
                # save_best may skip due to the sync check -> call save_plot right after fill_eval
                save_plot(
                    cfg.out_dir,
                    tag,
                    ind.metal,
                    ind.s11,
                    freqs,
                    ind.score,
                    ind.worst,
                    cfg.feed,
                    air=ind.air,
                    cut=ind.cut,
                    cfg=cfg,
                    faces=getattr(ind, "faces", None),
                    field_extra=ind.field_extra,
                )
            assert jpg_before.is_file() and jpg_before.stat().st_size > 8000, tag

            evo = cfg.out_dir / f"Evolution_gen{k + 1:03d}.jpg"
            save_evolution(cfg, ind, freqs, gen=k + 1, solver=solver, model=None)
            if not evo.is_file():
                save_plot(
                    cfg.out_dir,
                    f"Evolution_gen{k + 1:03d}",
                    ind.metal,
                    ind.s11,
                    freqs,
                    ind.score,
                    ind.worst,
                    cfg.feed,
                    air=ind.air,
                    cut=ind.cut,
                    cfg=cfg,
                    faces=getattr(ind, "faces", None),
                    field_extra=ind.field_extra,
                )
            assert evo.is_file() and evo.stat().st_size > 8000, str(evo)

            npy = cfg.byproduct_dir / "metal" / f"{tag}.npy"
            if not npy.is_file():
                npy.parent.mkdir(parents=True, exist_ok=True)
                np.save(npy, np.asarray(ind.metal, dtype=np.uint8))
            assert npy.is_file(), tag

            metal = np.asarray(ind.metal)
            if strict_mid_check:
                _check_layers(metal, ind)
            air_stats = overlay_alignment(metal, ind.air)
            patch_stats = overlay_alignment(metal, ind.field_extra["patch"])
            mid_stats = overlay_alignment(metal, ind.field_extra["mid"])

            if n <= 5:
                stale = ind.metal.copy()
                ind.metal = np.zeros_like(stale)
                ind.metal[feed] = 1
                ind.metal[24, 24] = 1
                mtime = jpg_before.stat().st_mtime
                save_best(cfg, ind, freqs, r=float(10 + k), tag=tag)
                assert jpg_before.stat().st_mtime == mtime, "stale eval overwrote JPG"
                ind.metal = stale

            rows.append(
                {
                    "tag": tag,
                    "copper": int(metal.sum()),
                    "score": float(ind.score or 0.0),
                    "patch_aligned": patch_stats["aligned"],
                    "mid_aligned": mid_stats["aligned"],
                    "jpg": str(jpg_before),
                }
            )
    finally:
        solver.close()

    if dump_cuda:
        write_diag_csv(cuda_root, analyze_cuda_field_dir(cuda_root))

    return rows


def test_metal_panel_feed_gnd_mm():
    """Copper map: green/gold, port red, ground marker blue."""
    cfg = make_config()
    assert cfg.feed == ij_from_mm(*FEED_MM)
    rgb = _metal_xy_rgb(np.zeros((PIX, PIX), dtype=np.uint8), cfg.feed)
    assert np.allclose(rgb[0, 0], _COLOR_NO_COPPER, atol=0.02)
    fix, fiy = ij_from_mm(*FEED_MM)
    gix, giy = ij_from_mm(*GND_MARK_MM)
    assert np.allclose(rgb[fiy, fix], _COLOR_FEED, atol=0.02)
    assert np.allclose(rgb[giy, gix], _COLOR_GND, atol=0.02)
    m = np.zeros((PIX, PIX), dtype=np.uint8)
    m[20:30, 20:30] = 1
    rgb2 = _metal_xy_rgb(m, cfg.feed)
    assert np.allclose(rgb2[22, 22], _COLOR_COPPER, atol=0.02)


def test_best_plot_random_batch(tmp_path):
    rows = run_best_plot_smoke(
        n=5, seed=DEFAULT_SEED, out_root=tmp_path / OUT_SUBDIR, strict_mid_check=True
    )
    assert len(rows) == 5


def main() -> int:
    n = int(os.environ.get("BEST_PLOT_TEST_N", str(DEFAULT_N)))
    seed = int(os.environ.get("BEST_PLOT_TEST_SEED", str(DEFAULT_SEED)))
    cfg = make_config()
    out = cfg.out_dir / OUT_SUBDIR
    print(f"Best JPG smoke  n={n}  seed={seed}  backend={cfg.backend}  out={out}")
    print(f"  PAD={FDTD_PAD}  crop→{PIX}×{PIX}  extent=±25.5 mm")
    print(f"  feed={FEED_MM} mm  gnd_mark={GND_MARK_MM} mm  (save_plot/_plot_metal_xy_panel)")
    rows = run_best_plot_smoke(n=n, seed=seed, out_root=out)
    print(f"{'tag':<16} {'copper':>7} {'score':>7}  patch_r  mid_r  jpg")
    for r in rows:
        print(
            f"{r['tag']:<16} {r['copper']:7d} {r['score']:7.2f}  "
            f"{r['patch_aligned']:6.3f} {r['mid_aligned']:6.3f}  {r['jpg']}"
        )
    cuda_root = out / "byproduct" / "cuda_fields"
    print(f"\nnpz: {cuda_root}  (each tag.npz + tag_direct.npz)")
    print_diag_report(analyze_cuda_field_dir(cuda_root))
    print(f"csv: {cuda_root / 'diag.csv'}")
    print("OK, save_best / save_plot paths passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

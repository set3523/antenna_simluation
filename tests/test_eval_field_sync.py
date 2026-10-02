# -*- coding: utf-8 -*-
"""When copper changes, |E| and score must be re-run through FDTD (prevents Evolution JPG mismatch)."""
from __future__ import annotations

import numpy as np

from core import Individual, PIX, clone_ind, seed_rectangle
from main import make_config
from opt_ga import _metal_cache_key, ensure_evaluated, fill_eval, save_evolution
from solver_cuda import VoxelFDTD


def test_metal_change_forces_new_fdtd():
    cfg = make_config()
    feed = cfg.feed
    freqs = np.linspace(cfg.f_start, cfg.f_stop, cfg.n_freq)
    solver = VoxelFDTD()
    solver.set_feed(*feed)
    try:
        big = seed_rectangle(feed)
        fill_eval(solver, big, freqs, cfg, log_chart=False)
        assert big.eval_ck == _metal_cache_key(big)
        assert big.air is not None
        old_air = np.array(big.air, copy=True)
        old_score = float(big.score or 0.0)

        tiny = np.zeros((PIX, PIX), dtype=np.uint8)
        tiny[feed] = 1
        tiny[25, 24] = 1
        tiny[25, 26] = 1
        big.metal = tiny
        # old bug: kept only score/fields
        big.eval_ck = None

        fill_eval(solver, big, freqs, cfg, log_chart=False)
        assert big.eval_ck == _metal_cache_key(big)
        assert big.air is not None
        assert float(np.std(big.air - old_air)) > 1e-6 or abs(float(big.score or 0) - old_score) > 0.05
    finally:
        solver.close()


def test_save_evolution_syncs_before_plot(tmp_path):
    cfg = make_config()
    cfg.out_dir = tmp_path
    feed = cfg.feed
    freqs = np.linspace(cfg.f_start, cfg.f_stop, cfg.n_freq)
    solver = VoxelFDTD()
    solver.set_feed(*feed)
    try:
        ref = seed_rectangle(feed)
        fill_eval(solver, ref, freqs, cfg, log_chart=False)
        stale = clone_ind(ref)
        tiny = np.zeros((PIX, PIX), dtype=np.uint8)
        tiny[feed] = 1
        stale.metal = tiny
        stale.eval_ck = None
        stale.score = ref.score
        stale.s11 = ref.s11
        stale.air = ref.air
        stale.field_extra = (
            None
            if ref.field_extra is None
            else {k: np.array(v, copy=True) for k, v in ref.field_extra.items()}
        )

        save_evolution(cfg, stale, freqs, gen=None, solver=solver, model=None)
        out = tmp_path / "Evolution_final.jpg"
        assert out.is_file()
        assert stale.eval_ck == _metal_cache_key(stale)
        assert int(stale.metal.sum()) <= 10
        assert stale.air is not None and ref.air is not None
        assert float(np.std(stale.air - ref.air)) > 1e-6
        rp = ref.field_extra["patch"]
        sp = stale.field_extra["patch"]
        assert float(np.std(sp - rp)) > 1e-3
    finally:
        solver.close()


def test_eval_ck_match_but_stale_fields_forces_fdtd():
    """eval_ck matched, but |E| and score are from the old large patch (JPG rectangular field bug)."""
    cfg = make_config()
    feed = cfg.feed
    freqs = np.linspace(cfg.f_start, cfg.f_stop, cfg.n_freq)
    solver = VoxelFDTD()
    solver.set_feed(*feed)
    try:
        big = seed_rectangle(feed)
        fill_eval(solver, big, freqs, cfg, log_chart=False)
        big_score = float(big.score or 0.0)

        tiny = np.zeros((PIX, PIX), dtype=np.uint8)
        tiny[feed] = 1
        for dy in range(-3, 4):
            tiny[25 + dy, 25] = 1
        stale = Individual(tiny, mode=-1)
        stale.eval_ck = _metal_cache_key(stale)
        stale.score = big.score
        stale.s11 = np.array(big.s11, copy=True)
        stale.worst = big.worst
        stale.air = np.array(big.air, copy=True)
        stale.field_extra = {k: np.array(v, copy=True) for k, v in big.field_extra.items()}

        save_evolution(cfg, stale, freqs, gen=9, solver=solver, model=None)
        assert stale.eval_ck == _metal_cache_key(stale)
        assert int(stale.metal.sum()) < 30
        assert float(stale.score or 0) < big_score - 0.5 or float(stale.score or 0) < 8.0
        sp = stale.field_extra["patch"]
        bp = big.field_extra["patch"]
        assert float(np.std(sp - bp)) > 1e-3
    finally:
        solver.close()

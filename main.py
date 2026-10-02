# -*- coding: utf-8 -*-
"""Edit only the evolution / RL direction here."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from core import FEED_MM, PATCH_LX_MM, PATCH_LY_MM, ij_from_mm

# ---------------------------------------------------------------------------
# Algorithm
#   ga    : GA only
#   rl    : RL only
#   ga_rl : GA, then fine-tune the elites with RL
# ---------------------------------------------------------------------------
ALGORITHM = "ga"  # "ga" | "rl" | "ga_rl"

# metal: chart/index = search the upper 51×51 layer (lower 9×11.7 rectangle is fixed in FDTD)
# feed: (experimental) fixed lower layer + fixed upper rectangle, scan only the port position
SEARCH_MODE = "metal"  # "metal" | "feed"
# True: after GA mutation, remove upper copper cut off from the feed. False: same as chart, floating islands also go to FDTD
CONNECT_REPAIR = True

# What counts as a better antenna
#   gain       : directivity × match efficiency (approx. realized gain). The final goal
#   band_worst : only the worst S11 in the target band
#   f0         : only S11 at the center frequency
#   bandwidth  : fraction of the band with S11 < S11_TARGET_DB
FITNESS = "gain"  # "gain" | "band_worst" | "f0" | "bandwidth"
W_CONE = 1.75        # cone 60° (dB vs. isotropic cone). v4: based on the NTFF far-field pattern
W_FACE6 = 0.85       # v4: directivity along the cone axis, dBi (NTFF). Old: near-field 6-face bias
W_EFFICIENCY = 3.0   # η = 1-|S11|² → η_dB (score mixes η_avg and η_min)
W_LEAVE = 1.0        # v4: total efficiency = radiated/incident power (NTFF surface Poynting, incl. mismatch and FR4 loss) → dB
FACE6_DB_MIN = 3.0   # lower bound for 6-face bias (below it the face6 term is 0)
# v4: leave is total efficiency in dB (≤0). Old values (14/8) were near-field |E|²/incident energy with wrong units (120~140 dB → always 0).
LEAVE_DB_MAX = 0.0   # total efficiency 100%
LEAVE_DB_MIN = -10.0  # below 10% is clipped to −10 dB
CONE_DB_MIN = 3.0    # lower bound for cone 60° dB. Below it the cone term is 0 (suppresses 2~3 dB stubs)
SCORE_BALANCE_W = 2.0  # bonus when cone, η and leave are all good together (0=off)
SCORE_BALANCE_SPREAD = 0.38  # bonus↑ when the relative spread of the three metrics is below this
MUT_FLIP_FRAC = 0.5  # mutation: pixel flip vs. edge delete/grow
SCORE_GEOM_MIX = 0.35  # S11 ranking: damps score blow-up when only one of η or leave is high (↓)
FDTD_T_SEC = 3.5e-9  # sim time. No early stop
FDTD_CFL = 0.99      # dt = CFL / (c √(1/dx²+1/dy²+1/dz²)). Max 0.99
BAND_HZ = (3.0e9, 9.0e9)   # target: S11 and η over this whole range (score, worst, Best)
S11_TARGET_DB = -10.0
F_START = BAND_HZ[0]       # FDTD sweep = BAND_HZ
F_STOP = BAND_HZ[1]
N_FREQ = 81                # S11 frequency points per FDTD run (uniform from F_START to F_STOP)
# score η = worst frequency in the band only (avoids the single-point dip trap)
ETA_BAND_MIN_MIX = 0.0  # v3: band-average η (with true S11 the single worst point is almost always 0 dB, so the score goes flat)
# full-band S11 terms when FITNESS=gain (ranking toward -10 dB over 3~9 GHz)
W_BAND_WORST = 0.5         # −worst/10 dB scale (↑ = prioritize worst S11 in band)
W_BAND_MATCH_FRAC = 30.0   # fraction of band with S11<target (1 = all frequencies)
S11_SCORE_PENALTY_K = 0.0  # v3: off. It made everything pile up near −150  # penalty per dB when worst > -10 dB (suppresses stub and η-only traps)

# GA
POPULATION = 10
GENERATIONS = 50
GA_BREED_DEFICIT = True  # if the score leader lacks cone or η: breed metric leaders together / score × metric leader
MUT_PIXELS = 48
TOURNAMENT = 3
SYMMETRY_MODES = [-1, 1, 4]  # -1 none, 1 orthogonal 4-way, 4 x-axis mirror
DIFFUSE_STEPS = 2   # how many times to apply the 3x3 diffusion matrix
DIFFUSE_RATE = 0.45 # scale for the chance of copper spreading to neighbors
# Record next-generation JPG/score only when strictly better than before.
# If offspring don't win, discard and regenerate. Stop if no improvement after this many batches in one generation.
MAX_OFFSPRING_BATCHES = 80  # v4 run 2: later stagnation call (run 1 used 40 → stopped at gen 35)
# local edge climb per offspring. Per batch, besides GA: RL 5 (per-metric maps) + diffusion B 5.
EXTRA_RL = 5
EXTRA_DIFFUSE = 5
LOCAL_CLIMB_STEPS = 0  # no pixel edge climb. Chart (u,v)±1 8-neighbor FDTD is done in warmup
LOCAL_TRY_PER_STEP = 4
DIFFUSE_RANK = 8  # diffusion B: number of PCA axes of offspring − elite. 3×3 is local growth only.
RANDOM_START = 0.2  # mostly breed explored points. Only a few random.
# Temperature is not lowered by sample count or ranking. Ranking is only logged as a convergence metric.

# Direct seed circle → follow the end points of the metric trend lines. Fog × hope on opposite sides → branch → when all end, GA.
#
# Chart: SpiralChart (r,θ) spiral XOR, chart_uv = (r cos θ, r sin θ). Anchor = seed r=0.
#   Best_r = chart_dist_plane(uv) ≤ r. index = mask_to_index is for ID only.
#   seed rectangle ≈ 9×12 cells (a single anchor point in index space).
#   trend = angle, direction, score slope → ≥2 points on the same angle.
#   η     resonance 1mm = whole edge → r≈12, slope r≈24
#   Dpeak point source at r≈8, aperture trend r≈20
#   beam  15mm window ≈ patch → r≈16~24
#   initial disk shared by all three metrics r_disk ≈ 24  (~1% of 2601)
#   expansion: snapshot Best at 20% and 40% of the full circle radius to see if it changes.
#
# Orange is kernel interpolation of green. Paint condition is c=1-exp(-Σw)≥0.18, not the 2.2ℓ cutoff
#   (by 2.2ℓ, N=12 seems to fill r=24, but the c threshold is shorter, leaving petal gaps)
#   two neighbors: 2 exp(-(d/ℓ)²)≥-ln(0.82),  d=2 r sin(π/(2N)),  ℓ floor=0.55 Δr
#   n_ang_to_cover: number of angles per ring. With Δr=4 the disk fills all of 4, 8, 12, 16, 20, 24.
#   value accuracy: green (direct FDTD) > orange (inferred). c means "green nearby", not "true value".
CIRCLE_WARMUP = True if SEARCH_MODE == "metal" else False
CIRCLE_LIGHT_ONLY = False
# last chart r of the disk (rings). None → CIRCLE_R_FRACS[0] × CIRCLE_R_MAX (snapped to the Δr grid)
CIRCLE_R_DISK_END = 240.0
# fraction of r_max: [0]=disk end (when DISK_END is None), [1..]=Best/Reveal JPG snapshots only
CIRCLE_R_FRACS = (0.20, 0.60)
CIRCLE_SEED_ANGLES = 6      # minimum for r≤8. Outer ring angles come from n_ang_to_cover
CIRCLE_DISK_HOLES = 12      # extra direct runs for gray gaps in the disk after the rings
CIRCLE_ORANGE_BEST = 0      # FDTD for orange Best candidates. 0 = all that pass the slack/orange/distance filters, N>0 = cap at N
CIRCLE_ORANGE_SLACK = 0.35  # run if estimate ≥ leader − slack. Interpolation can't beat the leader
CIRCLE_R_STEP = 4.0         # step for both disk and trend lines. No gaps between rings
# if r_max > spiral M (independent pixel count), s_limit=0 → chart breaks. Disk 240 + trend line margin.
CIRCLE_R_MAX = 765.0  # v4 run 2: trend lines up to 60% of max (1275) (run 1: 260)
CIRCLE_STAGNATE = 2         # a trend line ends after N consecutive non-improving steps
CIRCLE_ROOT_MAX = 6         # max live trend lines. Fog × hope on opposite sides → branch
CIRCLE_XOVER_DR = 16.0      # no crossover if parent r differs by less than this
CIRCLE_EXPAND_CAND = 16     # candidate angles for the next cell
CIRCLE_SEEDS = 1            # a single reference map
# no upper rectangle in FDTD — only the lower layer is fixed in the solver, upper = chart map
CIRCLE_BASELINE_RECT = False
# Best: exclude 1~3 px upper-layer stubs (upper rectangle is not fixed)
BEST_MIN_UPPER_COPPER = 30
# Best JPG: if any candidates have band worst ≤ S11_TARGET, only they compete for first (else fall back to all)
BEST_PREFER_S11_IN_BAND = True
# penalty per 1 dB when worst > target in the gain score (suppresses stub and leave traps)
# exploration chart origin: origin=global (0,0) index0, random=seed_random uv, rectangle=gate rectangle (old)
CIRCLE_ANCHOR = "origin"
CIRCLE_ANGLES = 6
CIRCLE_FAIL_ANGLES = 2

# RL (when ALGORITHM is rl / ga_rl)
RL_EPISODES = 20
RL_STEPS_PER_GEN = 0
RL_LR = 0.08
RL_BASELINE_MOMENTUM = 0.9

FEED = ij_from_mm(*FEED_MM)
SEED = 1  # v4 run 2: different search path (run 1 results in run_out_evo_final)
BACKEND = "cuda"  # compute only. "cpu" | "cuda"
OUT_DIR = Path(__file__).resolve().parent / "run_out"
BYPRODUCT_DIR = Path(__file__).resolve().parent / "run_byproduct"
SAVE_BYPRODUCT = True  # if True, save offspring masks/CSV to run_byproduct (for debugging)


def band_center_hz(band_hz: tuple[float, float]) -> float:
    """Band center Hz = (f_min + f_max) / 2."""
    return 0.5 * (float(band_hz[0]) + float(band_hz[1]))


def band_halfspan_hz(band_hz: tuple[float, float]) -> float:
    """Feed Gaussian FC = (f_max − f_min) / 2, so the spectrum covers the band."""
    return 0.5 * (float(band_hz[1]) - float(band_hz[0]))


@dataclass
class Config:
    backend: str
    algorithm: str
    population: int
    generations: int
    mut_pixels: int
    tournament: int
    symmetry_modes: list
    fitness: str
    band_hz: tuple
    f0_hz: float
    excitation_fc_hz: float
    w_band_worst: float
    w_band_match_frac: float
    s11_target_db: float
    w_directivity: float
    w_cone: float
    w_face6: float
    w_efficiency: float
    eta_band_min_mix: float
    w_leave: float
    face6_db_min: float
    leave_db_max: float
    leave_db_min: float
    cone_db_min: float
    score_balance_w: float
    score_balance_spread: float
    score_geom_mix: float
    fdtd_t_sec: float
    fdtd_cfl: float
    f_start: float
    f_stop: float
    n_freq: int
    rl_episodes: int
    rl_steps_per_gen: int
    rl_lr: float
    rl_baseline_momentum: float
    feed: tuple
    seed: int
    diffuse_steps: int = 2
    diffuse_rate: float = 0.45
    offspring_retries: int = 40
    extra_rl: int = 5
    extra_diffuse: int = 5
    local_climb_steps: int = 8
    local_try_per_step: int = 4
    diffuse_rank: int = 8
    random_start: float = 1.0
    ga_breed_deficit: bool = True
    mut_flip_frac: float = 0.5
    circle_warmup: bool = True
    circle_light_only: bool = False
    circle_r_disk_end: float | None = None
    circle_seed_angles: int = 6
    circle_disk_holes: int = 12
    circle_orange_best: int = 0
    circle_orange_slack: float = 0.35
    circle_expand_cand: int = 16
    circle_stagnate: int = 2
    circle_root_max: int = 6
    circle_xover_dr: float = 16.0
    circle_r_max: float = 130.0
    circle_r_fracs: tuple = (0.20, 0.60)
    circle_r_step: float = 4.0
    circle_angles: int = 6
    circle_seeds: int = 1
    circle_anchor: str = "origin"
    circle_fail_angles: int = 2
    circle_baseline_rect: bool = True
    best_min_upper_copper: int = 0
    best_prefer_s11_in_band: bool = True
    s11_score_penalty_k: float = 15.0
    out_dir: Path = field(default_factory=lambda: OUT_DIR)
    byproduct_dir: Path = field(default_factory=lambda: BYPRODUCT_DIR)
    save_byproduct: bool = False
    search_mode: str = "metal"
    patch_lx_mm: float = 9.0
    patch_ly_mm: float = 11.7
    connect_repair: bool = False


def make_config() -> Config:
    return Config(
        backend=BACKEND,
        algorithm=ALGORITHM,
        population=POPULATION,
        generations=GENERATIONS,
        mut_pixels=MUT_PIXELS,
        tournament=TOURNAMENT,
        symmetry_modes=list(SYMMETRY_MODES),
        fitness=FITNESS,
        band_hz=BAND_HZ,
        f0_hz=band_center_hz(BAND_HZ),
        excitation_fc_hz=band_halfspan_hz(BAND_HZ),
        w_band_worst=W_BAND_WORST,
        w_band_match_frac=W_BAND_MATCH_FRAC,
        s11_target_db=S11_TARGET_DB,
        w_directivity=W_CONE,
        w_cone=W_CONE,
        w_face6=W_FACE6,
        w_efficiency=W_EFFICIENCY,
        eta_band_min_mix=ETA_BAND_MIN_MIX,
        w_leave=W_LEAVE,
        face6_db_min=FACE6_DB_MIN,
        leave_db_max=LEAVE_DB_MAX,
        leave_db_min=LEAVE_DB_MIN,
        cone_db_min=CONE_DB_MIN,
        score_balance_w=SCORE_BALANCE_W,
        score_balance_spread=SCORE_BALANCE_SPREAD,
        score_geom_mix=SCORE_GEOM_MIX,
        fdtd_t_sec=FDTD_T_SEC,
        fdtd_cfl=FDTD_CFL,
        f_start=F_START,
        f_stop=F_STOP,
        n_freq=N_FREQ,
        rl_episodes=RL_EPISODES,
        rl_steps_per_gen=RL_STEPS_PER_GEN,
        rl_lr=RL_LR,
        rl_baseline_momentum=RL_BASELINE_MOMENTUM,
        feed=FEED,
        seed=SEED,
        diffuse_steps=DIFFUSE_STEPS,
        diffuse_rate=DIFFUSE_RATE,
        offspring_retries=MAX_OFFSPRING_BATCHES,
        extra_rl=EXTRA_RL,
        extra_diffuse=EXTRA_DIFFUSE,
        local_climb_steps=LOCAL_CLIMB_STEPS,
        local_try_per_step=LOCAL_TRY_PER_STEP,
        diffuse_rank=DIFFUSE_RANK,
        random_start=RANDOM_START,
        ga_breed_deficit=GA_BREED_DEFICIT,
        mut_flip_frac=MUT_FLIP_FRAC,
        circle_warmup=CIRCLE_WARMUP,
        circle_light_only=CIRCLE_LIGHT_ONLY,
        circle_r_disk_end=CIRCLE_R_DISK_END,
        circle_seed_angles=CIRCLE_SEED_ANGLES,
        circle_disk_holes=CIRCLE_DISK_HOLES,
        circle_orange_best=CIRCLE_ORANGE_BEST,
        circle_orange_slack=CIRCLE_ORANGE_SLACK,
        circle_expand_cand=CIRCLE_EXPAND_CAND,
        circle_stagnate=CIRCLE_STAGNATE,
        circle_root_max=CIRCLE_ROOT_MAX,
        circle_xover_dr=CIRCLE_XOVER_DR,
        circle_r_max=CIRCLE_R_MAX,
        circle_r_fracs=CIRCLE_R_FRACS,
        circle_r_step=CIRCLE_R_STEP,
        circle_angles=CIRCLE_ANGLES,
        circle_seeds=CIRCLE_SEEDS,
        circle_anchor=CIRCLE_ANCHOR,
        circle_fail_angles=CIRCLE_FAIL_ANGLES,
        circle_baseline_rect=CIRCLE_BASELINE_RECT,
        best_min_upper_copper=BEST_MIN_UPPER_COPPER,
        best_prefer_s11_in_band=BEST_PREFER_S11_IN_BAND,
        s11_score_penalty_k=S11_SCORE_PENALTY_K,
        out_dir=OUT_DIR,
        byproduct_dir=BYPRODUCT_DIR,
        save_byproduct=SAVE_BYPRODUCT,
        search_mode=SEARCH_MODE,
        patch_lx_mm=PATCH_LX_MM,
        patch_ly_mm=PATCH_LY_MM,
        connect_repair=CONNECT_REPAIR,
    )


def make_solver(backend: str):
    from opt_ga import reset_eval_cache

    reset_eval_cache()
    backend = backend.lower().strip()
    if backend == "cuda":
        from solver_cuda import VoxelFDTD
    elif backend == "cpu":
        from solver_cpu import VoxelFDTD
    else:
        raise ValueError(f"BACKEND must be 'cpu' or 'cuda': {backend!r}")
    return VoxelFDTD()


def main():
    cfg = make_config()
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 8) - 1)))
    print(
        f"config  algorithm={cfg.algorithm}  fitness={cfg.fitness}  "
        f"search={cfg.search_mode}  patch={cfg.patch_lx_mm}×{cfg.patch_ly_mm} mm  "
        f"band={cfg.band_hz[0]/1e9:.2f}-{cfg.band_hz[1]/1e9:.2f} GHz  backend={cfg.backend}"
    )
    from openems.gate import ensure_before_optimize

    ensure_before_optimize(cfg.backend)
    solver = make_solver(cfg.backend)
    try:
        algo = cfg.algorithm.lower().strip()
        if algo == "ga":
            from opt_ga import run_ga

            run_ga(solver, cfg)
        elif algo == "rl":
            from opt_rl import run_rl

            run_rl(solver, cfg)
        elif algo == "ga_rl":
            from opt_ga import run_ga
            from opt_rl import refine_individual, run_rl

            best = run_ga(solver, cfg, rl_refine=refine_individual)
            print("GA done → RL fine-tuning")
            run_rl(solver, cfg, start=best)
        else:
            raise ValueError(f"ALGORITHM must be 'ga' | 'rl' | 'ga_rl': {algo!r}")
    finally:
        solver.close()
    print("done", cfg.out_dir)


if __name__ == "__main__":
    main()

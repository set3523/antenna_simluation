# -*- coding: utf-8 -*-
"""NTFF: tangential E and H (DFT) on the FDTD Huygens box -> far-field radiation pattern.

Used only when the solver provides voxel_ntff_info / voxel_ntff_get (v4 DLL).
Equivalent currents J = n×H, M = −n×E, radiation vectors N and L (Balanis 6.8, e^{jωt} convention).
  U(θ,φ) = k²/(32π²η) (|Lφ + η Nθ|² + |Lθ − η Nφ|²)
  D = 4π U / ∫U dΩ
Efficiency = P_rad / P_inc is computed on the C side by a surface Poynting integral (solver.leave()).
"""
from __future__ import annotations

import numpy as np

C0 = 299792458.0
ETA0 = 376.730313668
CONE_HALF_DEG = 30.0  # full angle 60°
AXES = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64
)
AXIS_NAME = ("+x", "-x", "+y", "-y", "+z", "-z")


def fibonacci_sphere(n: int = 900) -> np.ndarray:
    """Nearly uniform points on the sphere (each point covers solid angle 4π/n)."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    phi = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


_DIRS = fibonacci_sphere(640)


def _coarsen(a: np.ndarray, b0: int, b1: int) -> np.ndarray:
    """Sum a[..., n0, n1] over b0×b1 blocks (edges zero-padded)."""
    n0, n1 = a.shape[-2], a.shape[-1]
    p0 = (-n0) % b0
    p1 = (-n1) % b1
    if p0 or p1:
        pad = [(0, 0)] * (a.ndim - 2) + [(0, p0), (0, p1)]
        a = np.pad(a, pad)
    s = a.shape
    a = a.reshape(s[:-2] + (s[-2] // b0, b0, s[-1] // b1, b1))
    return a.sum(axis=(-3, -1))


def _block_centers(c0: np.ndarray, c1: np.ndarray, b0: int, b1: int):
    """Cell-center coordinate vectors c0 (n0), c1 (n1) -> block centers."""
    def cc(c, b):
        n = c.size
        p = (-n) % b
        if p:
            step = c[1] - c[0] if n > 1 else 1.0
            c = np.concatenate([c, c[-1] + step * np.arange(1, p + 1)])
        return c.reshape(-1, b).mean(axis=1)

    return cc(c0, b0), cc(c1, b1)


def build_surface(info, d3, buf, nf):
    """
    DFT buffers -> list of equivalent currents.
    Returns: pos (M,3) [m], J (nf,M,3), Mm (nf,M,3)  (multiplied by area)
    """
    ia, ib, ja, jb, ka, kb = (int(v) for v in info[:6])
    lx, ly, lz = int(info[8]), int(info[9]), int(info[10])
    k_patch, a0 = int(info[11]), int(info[12])
    dx, dy, dz = (float(v) for v in d3[:3])
    b = np.asarray(buf, dtype=np.float32).reshape(-1, nf, 4, 2)
    cplx = (b[..., 0] + 1j * b[..., 1]).astype(np.complex128)  # (cells, nf, 4)
    cx, cy, cz = ly * lz, lx * lz, lx * ly
    offs = [0, cx, 2 * cx, 2 * cx + cy, 2 * cx + 2 * cy, 2 * cx + 2 * cy + cz, 2 * cx + 2 * cy + 2 * cz]
    # grid index -> coordinates (origin = board center, patch plane)
    xc = (np.arange(ia, ib) + 0.5 - a0) * dx
    yc = (np.arange(ja, jb) + 0.5 - a0) * dy
    zc = (np.arange(ka, kb) + 0.5 - k_patch) * dz
    X_ia, X_ib = (ia - a0) * dx, (ib - a0) * dx
    Y_ja, Y_jb = (ja - a0) * dy, (jb - a0) * dy
    Z_ka, Z_kb = (ka - k_patch) * dz, (kb - k_patch) * dz
    # z has dz=0.375 mm so group 4 cells; xy groups 2 cells (block ≤ 2 mm ≪ λ/16 @ 9 GHz)
    bxy, bz = 2, 4
    pos_all, J_all, M_all = [], [], []
    for f6 in range(6):
        v = cplx[offs[f6] : offs[f6 + 1]]  # (nc, nf, 4)
        if f6 < 2:
            n0, n1, c0, c1, b0, b1 = ly, lz, yc, zc, bxy, bz
            da = dy * dz
        elif f6 < 4:
            n0, n1, c0, c1, b0, b1 = lx, lz, xc, zc, bxy, bz
            da = dx * dz
        else:
            n0, n1, c0, c1, b0, b1 = lx, ly, xc, yc, bxy, bxy
            da = dx * dy
        # local l = a + n0*b -> reshape to [b][a], then [a][b]
        v = v.reshape(n1, n0, nf, 4).transpose(2, 3, 1, 0)  # (nf,4,n0,n1)
        g0, g1 = _block_centers(c0, c1, b0, b1)
        v = _coarsen(v, b0, b1) * da  # multiply by area
        A0, A1 = np.meshgrid(g0, g1, indexing="ij")
        ea, eb, ha, hb = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        z0 = np.zeros_like(ea)
        if f6 < 2:
            X = np.full_like(A0, X_ib if f6 == 0 else X_ia)
            pos = np.stack([X, A0, A1], -1)
            E = np.stack([z0, ea, eb], -1)
            H = np.stack([z0, ha, hb], -1)
        elif f6 < 4:
            Y = np.full_like(A0, Y_jb if f6 == 2 else Y_ja)
            pos = np.stack([A0, Y, A1], -1)
            E = np.stack([ea, z0, eb], -1)
            H = np.stack([ha, z0, hb], -1)
        else:
            Z = np.full_like(A0, Z_kb if f6 == 4 else Z_ka)
            pos = np.stack([A0, A1, Z], -1)
            E = np.stack([ea, eb, z0], -1)
            H = np.stack([ha, hb, z0], -1)
        nvec = AXES[f6]
        J = np.cross(nvec, H)
        Mm = -np.cross(nvec, E)
        pos_all.append(pos.reshape(-1, 3))
        J_all.append(J.reshape(nf, -1, 3))
        M_all.append(Mm.reshape(nf, -1, 3))
    return np.concatenate(pos_all, 0), np.concatenate(J_all, 1), np.concatenate(M_all, 1)


def radiation_intensity(pos, J, Mm, freqs, dirs=None):
    """U (nf, ndir). dirs are unit vectors (ndir,3)."""
    if dirs is None:
        dirs = _DIRS
    ux, uy, uz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    th = np.arccos(np.clip(uz, -1.0, 1.0))
    ph = np.arctan2(uy, ux)
    ct, st, cp, sp = np.cos(th), np.sin(th), np.cos(ph), np.sin(ph)
    th_hat = np.stack([ct * cp, ct * sp, -st], -1)
    ph_hat = np.stack([-sp, cp, np.zeros_like(sp)], -1)
    proj = (dirs @ pos.T).astype(np.float32)  # (ndir, M)
    out = np.empty((len(freqs), dirs.shape[0]))
    for f, fr in enumerate(freqs):
        k = 2.0 * np.pi * fr / C0
        a = np.float32(k) * proj
        ph_m = np.cos(a) + 1j * np.sin(a)  # complex64 (ndir, M)
        JM = np.concatenate([J[f], Mm[f]], 1).astype(np.complex64)  # (M,6)
        NL = (ph_m @ JM).astype(np.complex128)
        Nv, Lv = NL[:, :3], NL[:, 3:]
        Nt = np.sum(Nv * th_hat, 1)
        Np = np.sum(Nv * ph_hat, 1)
        Lt = np.sum(Lv * th_hat, 1)
        Lp = np.sum(Lv * ph_hat, 1)
        out[f] = k * k / (32.0 * np.pi**2 * ETA0) * (np.abs(Lp + ETA0 * Nt) ** 2 + np.abs(Lt - ETA0 * Np) ** 2)
    return out


def pattern_metrics(U: np.ndarray, dirs=None) -> dict:
    """Per-frequency U -> axial cone fraction, axial directivity, peak directivity (averaged over frequency)."""
    if dirs is None:
        dirs = _DIRS
    n = dirs.shape[0]
    dOmega = 4.0 * np.pi / n
    P = U.sum(1) * dOmega  # (nf,)
    D = 4.0 * np.pi * U / np.maximum(P[:, None], 1e-300)  # (nf, ndir)
    cosc = np.cos(np.deg2rad(CONE_HALF_DEG))
    cone = np.zeros((U.shape[0], 6))
    d_axis = np.zeros((U.shape[0], 6))
    for a in range(6):
        c = dirs @ AXES[a]
        m = c >= cosc
        cone[:, a] = U[:, m].sum(1) * dOmega / np.maximum(P, 1e-300)
        near = np.argsort(-c)[:5]  # average of the 5 points closest to the axis
        d_axis[:, a] = D[:, near].mean(1)
    return {
        "p_u": P,
        "cone": cone.mean(0),
        "d_axis": d_axis.mean(0),
        "d_peak": float(np.mean(D.max(1))),
        "d_peak_f": D.max(1),
        "D": D,
    }


def far_field(solver) -> dict | None:
    """NTFF result from the solver (ctypes DLL wrapper). None if unavailable."""
    dll = getattr(solver, "_dll", None)
    if dll is None or not hasattr(dll, "voxel_ntff_get"):
        return None
    import ctypes

    info = np.zeros(13, dtype=np.int32)
    d3 = np.zeros(3, dtype=np.float64)
    dll.voxel_ntff_info(solver._ctx, info.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        d3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    nf, ncell = int(info[6]), int(info[7])
    buf = np.zeros(ncell * nf * 8, dtype=np.float32)
    fr = np.zeros(nf)
    pinc = np.zeros(nf)
    prad = np.zeros(nf)
    P = ctypes.POINTER(ctypes.c_double)
    dll.voxel_ntff_get(solver._ctx, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       fr.ctypes.data_as(P), pinc.ctypes.data_as(P), prad.ctypes.data_as(P))
    if not np.any(buf):
        return None
    pos, J, Mm = build_surface(info, d3, buf, nf)
    U = radiation_intensity(pos, J, Mm, fr)
    m = pattern_metrics(U)
    m.update({"freqs": fr, "p_inc": pinc, "p_rad": prad, "eff": prad / np.maximum(pinc, 1e-300)})
    return m


def faces_from_ff(ff: dict) -> np.ndarray:
    """faces array format for opt_ga: [0:6] axial directivity (linear), [6:12] cone fraction, [12]=1 (far-field marker)."""
    return np.concatenate([ff["d_axis"], ff["cone"], [1.0]]).astype(np.float64)

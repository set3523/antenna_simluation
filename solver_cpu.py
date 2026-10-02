# -*- coding: utf-8 -*-
"""CPU voxel FDTD (OpenMP DLL). Compute only. The algorithm lives in main.py."""
from __future__ import annotations

import ctypes
import os
import shutil
from pathlib import Path

import numpy as np

_DLL = None
_PIX = 51
NATIVE = Path(__file__).resolve().parent / "native"


def _load():
    global _DLL, _PIX
    if _DLL is not None:
        return _DLL
    dll_path = NATIVE / "voxel_fdtd.dll"
    if not dll_path.exists():
        raise FileNotFoundError(f"{dll_path} not found. Build it first with native/compile.ps1.")
    load_dir = Path(os.environ.get("TEMP", r"C:\Temp")) / "antenna_voxel_dll_v4"
    load_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "voxel_fdtd.dll",
        "libgomp-1.dll",
        "libgcc_s_seh-1.dll",
        "libstdc++-6.dll",
        "libwinpthread-1.dll",
        "libdl.dll",
    ):
        src = NATIVE / name
        if src.exists():
            dst = load_dir / name
            if (not dst.exists()) or src.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(src, dst)
    dll_path = load_dir / "voxel_fdtd.dll"
    try:
        os.add_dll_directory(str(load_dir))
    except (AttributeError, FileNotFoundError, OSError):
        os.environ["PATH"] = str(load_dir) + os.pathsep + os.environ.get("PATH", "")
    _DLL = ctypes.CDLL(str(dll_path))
    _bind(_DLL)
    pix = ctypes.c_int()
    _DLL.voxel_grid(None, None, None, ctypes.byref(pix))
    _PIX = pix.value
    return _DLL


def _bind(dll):
    dll.voxel_create.restype = ctypes.c_void_p
    dll.voxel_destroy.argtypes = [ctypes.c_void_p]
    dll.voxel_set_feed.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    dll.voxel_set_metal.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
    ]
    dll.voxel_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    dll.voxel_run.restype = ctypes.c_int
    dll.voxel_grid.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    dll.voxel_air_e.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    if hasattr(dll, "voxel_cut_e"):
        dll.voxel_cut_e.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
    if hasattr(dll, "voxel_extra_fields"):
        dll.voxel_extra_fields.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
    if hasattr(dll, "voxel_faces"):
        dll.voxel_faces.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
        ]
    if hasattr(dll, "voxel_set_lower_full_gnd"):
        dll.voxel_set_lower_full_gnd.argtypes = [ctypes.c_void_p, ctypes.c_int]
    if hasattr(dll, "voxel_set_sim"):
        dll.voxel_set_sim.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
    if hasattr(dll, "voxel_ntff_get"):
        dll.voxel_ntff_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]
        dll.voxel_ntff_get.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        dll.voxel_ntff_set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    if hasattr(dll, "voxel_leave"):
        dll.voxel_leave.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]


class VoxelFDTD:
    backend = "cpu"

    def __init__(self):
        dll = _load()
        self._dll = dll
        self._ctx = dll.voxel_create()
        if not self._ctx:
            raise RuntimeError("CPU voxel FDTD creation failed")
        self.pix = _PIX

    def close(self):
        if getattr(self, "_ctx", None):
            self._dll.voxel_destroy(self._ctx)
            self._ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def set_feed(self, px: int, py: int):
        self._dll.voxel_set_feed(self._ctx, int(px), int(py))

    def set_lower_full_ground(self, on: bool = True):
        if hasattr(self._dll, "voxel_set_lower_full_gnd"):
            self._dll.voxel_set_lower_full_gnd(self._ctx, 1 if on else 0)

    def run(self, metal: np.ndarray, freqs: np.ndarray):
        # C reads metal[i + PIX*j] (i=x). Flattening numpy [i, j] as is would transpose it, so pass .T.
        metal = np.ascontiguousarray(np.asarray(metal, dtype=np.uint8).T).reshape(-1)
        freqs = np.ascontiguousarray(freqs, dtype=np.float64)
        s11 = np.empty(freqs.size, dtype=np.float64)
        min_db = ctypes.c_double()
        min_hz = ctypes.c_double()
        self._dll.voxel_set_metal(
            self._ctx,
            metal.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            metal.size,
        )
        nused = self._dll.voxel_run(
            self._ctx,
            freqs.size,
            freqs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            s11.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(min_db),
            ctypes.byref(min_hz),
        )
        return s11, float(min_db.value), float(min_hz.value), int(nused)

    def air_e(self) -> np.ndarray:
        nx = ctypes.c_int()
        ny = ctypes.c_int()
        self._dll.voxel_grid(ctypes.byref(nx), ctypes.byref(ny), None, None)
        buf = np.zeros(nx.value * ny.value, dtype=np.float32)
        self._dll.voxel_air_e(
            self._ctx,
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(nx),
            ctypes.byref(ny),
        )
        # C: i + NX*j  → (NX, NY) so [x, y] matches metal
        return buf.reshape((ny.value, nx.value)).T

    def cut_e(self) -> np.ndarray | None:
        if not hasattr(self._dll, "voxel_cut_e"):
            return None
        nx = ctypes.c_int()
        nz = ctypes.c_int()
        self._dll.voxel_grid(ctypes.byref(nx), None, ctypes.byref(nz), None)
        buf = np.zeros(nx.value * nz.value, dtype=np.float32)
        self._dll.voxel_cut_e(
            self._ctx,
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(nx),
            ctypes.byref(nz),
        )
        # C: i + NX*k → (NX, NZ) [x, z]
        return buf.reshape((nz.value, nx.value)).T

    def extra_fields(self) -> dict[str, np.ndarray] | None:
        if not hasattr(self._dll, "voxel_extra_fields"):
            return None
        nx = ctypes.c_int()
        ny = ctypes.c_int()
        nz = ctypes.c_int()
        nx2, ny2, nz2 = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        self._dll.voxel_grid(ctypes.byref(nx2), ctypes.byref(ny2), ctypes.byref(nz2), None)
        nxy = nx2.value * ny2.value
        nxz = nx2.value * nz2.value
        patch = np.zeros(nxy, dtype=np.float32)
        mid = np.zeros(nxy, dtype=np.float32)
        top = np.zeros(nxy, dtype=np.float32)
        cut_max = np.zeros(nxz, dtype=np.float32)
        self._dll.voxel_extra_fields(
            self._ctx,
            patch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            mid.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            top.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cut_max.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(nx),
            ctypes.byref(ny),
            ctypes.byref(nz),
        )
        return {
            "patch": patch.reshape((ny.value, nx.value)).T,
            "mid": mid.reshape((ny.value, nx.value)).T,
            "top": top.reshape((ny.value, nx.value)).T,
            "cut_max": cut_max.reshape((nz.value, nx.value)).T,
        }

    def far_field(self) -> dict | None:
        """v4: NTFF far-field metrics (None if unavailable). Call after run()."""
        import ntff

        try:
            return ntff.far_field(self)
        except Exception as e:  # don't let a metric failure stop the search
            print("[ntff] failed:", e)
            return None

    def faces(self) -> np.ndarray | None:
        # v4: with far field (NTFF): [axis directivity 6, cone ratio 6, 1] (13). Otherwise the old 12 near-field values.
        ff = self.far_field() if hasattr(self._dll, "voxel_ntff_get") else None
        if ff is not None:
            import ntff

            self.last_ff = ff
            return ntff.faces_from_ff(ff)
        """Mean |E|^2 on the 6 boundary faces. Order +x -x +y -y +z -z."""
        if not hasattr(self._dll, "voxel_faces"):
            return None
        buf = np.zeros(12, dtype=np.float32)
        self._dll.voxel_faces(self._ctx, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return buf

    def set_sim(self, t_sec: float, cfl: float, f0_hz: float, fc_hz: float):
        if hasattr(self._dll, "voxel_set_sim"):
            self._dll.voxel_set_sim(
                self._ctx, float(t_sec), float(cfl), float(f0_hz), float(fc_hz)
            )

    def leave(self) -> float | None:
        if not hasattr(self._dll, "voxel_leave"):
            return None
        buf = ctypes.c_float()
        self._dll.voxel_leave(self._ctx, ctypes.byref(buf))
        return float(buf.value)


def rectangle_mask(pix=51, lx_mm=9.0, ly_mm=11.7):
    """Same rectangle patch as validate_s11.py (1 mm pixels)."""
    g = np.zeros((pix, pix), dtype=np.uint8)
    cx = cy = pix // 2
    nx = max(1, int(round(lx_mm)))
    ny = max(1, int(round(ly_mm)))
    x0 = cx - nx // 2
    y0 = cy - ny // 2
    g[x0 : x0 + nx, y0 : y0 + ny] = 1
    return g

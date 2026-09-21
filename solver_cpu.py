# -*- coding: utf-8 -*-
"""CPU voxel FDTD (OpenMP DLL). 연산만 담당. 알고리즘은 main.py."""
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
        raise FileNotFoundError(f"{dll_path} 없음. native/compile.ps1 로 먼저 빌드하세요.")
    load_dir = Path(os.environ.get("TEMP", r"C:\Temp")) / "antenna_voxel_dll"
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


class VoxelFDTD:
    backend = "cpu"

    def __init__(self):
        dll = _load()
        self._dll = dll
        self._ctx = dll.voxel_create()
        if not self._ctx:
            raise RuntimeError("CPU voxel FDTD 생성 실패")
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

    def run(self, metal: np.ndarray, freqs: np.ndarray):
        metal = np.ascontiguousarray(metal, dtype=np.uint8).reshape(-1)
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
        # C: i + NX*j  → (NX, NY) so [x, y] 가 metal 과 같음
        return buf.reshape((ny.value, nx.value)).T


def rectangle_mask(pix=51, lx_mm=9.0, ly_mm=11.7):
    """validate_s11.py 와 같은 사각 패치 (1 mm 픽셀)."""
    g = np.zeros((pix, pix), dtype=np.uint8)
    cx = cy = pix // 2
    nx = max(1, int(round(lx_mm)))
    ny = max(1, int(round(ly_mm)))
    x0 = cx - nx // 2
    y0 = cy - ny // 2
    g[x0 : x0 + nx, y0 : y0 + ny] = 1
    return g

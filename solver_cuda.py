# -*- coding: utf-8 -*-
"""CUDA voxel FDTD. 연산만 담당. 알고리즘은 main.py."""
from __future__ import annotations

import ctypes
import os
import shutil
from pathlib import Path

import numpy as np

from solver_cpu import rectangle_mask

_DLL = None
_PIX = 51
NATIVE = Path(__file__).resolve().parent / "native"


def _load():
    global _DLL, _PIX
    if _DLL is not None:
        return _DLL
    dll_path = NATIVE / "voxel_fdtd_cuda.dll"
    if not dll_path.exists():
        raise FileNotFoundError(
            f"{dll_path} 없음. native/compile_cuda.ps1 로 먼저 빌드하세요."
        )
    os.environ["ANTENNA_NATIVE"] = str(NATIVE)
    load_dir = Path(os.environ.get("TEMP", r"C:\Temp")) / "antenna_voxel_cuda_dll"
    load_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "voxel_fdtd_cuda.dll",
        "fdtd_kernels.cu",
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
    dll_path = load_dir / "voxel_fdtd_cuda.dll"
    try:
        os.add_dll_directory(str(load_dir))
    except (AttributeError, FileNotFoundError, OSError):
        os.environ["PATH"] = str(load_dir) + os.pathsep + os.environ.get("PATH", "")
    _DLL = ctypes.CDLL(str(dll_path))
    _DLL.voxel_create.restype = ctypes.c_void_p
    _DLL.voxel_destroy.argtypes = [ctypes.c_void_p]
    _DLL.voxel_set_feed.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    _DLL.voxel_set_metal.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
    ]
    _DLL.voxel_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    _DLL.voxel_run.restype = ctypes.c_int
    _DLL.voxel_grid.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    _DLL.voxel_air_e.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    pix = ctypes.c_int()
    _DLL.voxel_grid(None, None, None, ctypes.byref(pix))
    _PIX = pix.value
    return _DLL


class VoxelFDTD:
    backend = "cuda"

    def __init__(self):
        dll = _load()
        self._dll = dll
        self._ctx = dll.voxel_create()
        if not self._ctx:
            raise RuntimeError(
                "CUDA voxel FDTD 생성 실패. GPU/VRAM/NVRTC 를 확인하거나 "
                "main.py 에서 BACKEND='cpu' 로 바꾸세요."
            )
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
        return buf.reshape((ny.value, nx.value)).T

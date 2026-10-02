# -*- coding: utf-8 -*-
"""CUDA voxel FDTD. Compute only. The algorithm lives in main.py."""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
from pathlib import Path

import numpy as np

from solver_cpu import rectangle_mask

_DLL = None
_PIX = 51
NATIVE = Path(__file__).resolve().parent / "native"
_ARTIFACTS = (
    "voxel_fdtd_cuda.dll",
    "fdtd_kernels.cu",
    "libgcc_s_seh-1.dll",
    "libstdc++-6.dll",
    "libwinpthread-1.dll",
    "libdl.dll",
)


def _native_bundle_digest() -> str:
    h = hashlib.sha256()
    for name in _ARTIFACTS:
        src = NATIVE / name
        if not src.is_file():
            continue
        st = src.stat()
        h.update(name.encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime_ns)).encode())
    return h.hexdigest()


def _sync_cuda_load_dir(load_dir: Path) -> None:
    """At process start, sync repo native → %TEMP% (no manual cache clearing needed).

    ANTENNA_CUDA_CACHE=keep skips the copy only when the digest matches.
    """
    mode = os.environ.get("ANTENNA_CUDA_CACHE", "refresh").strip().lower()
    load_dir.mkdir(parents=True, exist_ok=True)
    digest = _native_bundle_digest()
    stamp = load_dir / "native_bundle.sha256"
    if mode in ("keep", "reuse") and stamp.is_file():
        if stamp.read_text(encoding="utf-8").strip() == digest:
            missing = [n for n in _ARTIFACTS if (NATIVE / n).is_file() and not (load_dir / n).is_file()]
            if not missing:
                return
    if mode in ("clear", "fresh"):
        shutil.rmtree(load_dir, ignore_errors=True)
        load_dir.mkdir(parents=True, exist_ok=True)
    for name in _ARTIFACTS:
        src = NATIVE / name
        if src.is_file():
            shutil.copy2(src, load_dir / name)
    stamp.write_text(digest + "\n", encoding="utf-8")


def _load():
    global _DLL, _PIX
    if _DLL is not None:
        return _DLL
    src_dll = NATIVE / "voxel_fdtd_cuda.dll"
    if not src_dll.exists():
        raise FileNotFoundError(
            f"{src_dll} not found. Build it first with native/compile_cuda.ps1."
        )
    # NVRTC compiles fdtd_kernels.cu from ANENNA_NATIVE (straight from the repo)
    os.environ["ANTENNA_NATIVE"] = str(NATIVE)
    load_dir = Path(os.environ.get("TEMP", r"C:\Temp")) / "antenna_voxel_cuda_dll_v4"
    _sync_cuda_load_dir(load_dir)
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
    if hasattr(_DLL, "voxel_cut_e"):
        _DLL.voxel_cut_e.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
    if hasattr(_DLL, "voxel_extra_fields"):
        _DLL.voxel_extra_fields.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
    if hasattr(_DLL, "voxel_faces"):
        _DLL.voxel_faces.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
        ]
    if hasattr(_DLL, "voxel_set_lower_full_gnd"):
        _DLL.voxel_set_lower_full_gnd.argtypes = [ctypes.c_void_p, ctypes.c_int]
    if hasattr(_DLL, "voxel_set_sim"):
        _DLL.voxel_set_sim.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
    if hasattr(_DLL, "voxel_leave"):
        _DLL.voxel_leave.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]

    if hasattr(_DLL, "voxel_ntff_get"):
        _DLL.voxel_ntff_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]
        _DLL.voxel_ntff_get.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        _DLL.voxel_ntff_set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
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
                "CUDA voxel FDTD creation failed. Check GPU, VRAM, NVRTC, or "
                "set BACKEND='cpu' in main.py."
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

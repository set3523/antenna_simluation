# -*- coding: utf-8 -*-
"""openEMS DLL 경로. 검증 폴더에서만 쓴다."""
from __future__ import annotations

import os


def register():
    install_path = (
        os.environ.get("CSXCAD_INSTALL_PATH")
        or os.environ.get("OPENEMS_INSTALL_PATH")
        or r"C:\openEMS"
    )
    if not os.path.isdir(install_path):
        raise FileNotFoundError(
            f"openEMS 설치 경로를 찾을 수 없습니다: {install_path}\n"
            "CSXCAD_INSTALL_PATH 또는 OPENEMS_INSTALL_PATH 를 확인하세요."
        )
    os.environ.setdefault("CSXCAD_INSTALL_PATH", install_path)
    os.environ.setdefault("OPENEMS_INSTALL_PATH", install_path)
    try:
        os.add_dll_directory(install_path)
    except (AttributeError, FileNotFoundError):
        pass
    if install_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = install_path + os.pathsep + os.environ.get("PATH", "")
    return install_path

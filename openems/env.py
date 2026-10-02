# -*- coding: utf-8 -*-
"""openEMS DLL paths. Used only in the validation folder."""
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
            f"openEMS install path not found: {install_path}\n"
            "Check CSXCAD_INSTALL_PATH or OPENEMS_INSTALL_PATH."
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

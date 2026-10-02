# -*- coding: utf-8 -*-
"""Validation moved to openems/. Kept for the old path."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "openems" / "run.py"), run_name="__main__")

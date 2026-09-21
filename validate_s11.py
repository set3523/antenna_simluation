# -*- coding: utf-8 -*-
"""검증은 openems/ 로 옮겼다. 예전 경로 유지용."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "openems" / "run.py"), run_name="__main__")

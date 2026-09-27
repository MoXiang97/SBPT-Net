"""Importable adapter for the frozen formal baseline entry on Windows workers.

``multiprocessing.spawn`` must be able to import the module that owns dataset
classes.  Executing the frozen entry in this real module keeps those classes
pickleable without editing the archived paper implementation.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENTRY = (
    ROOT
    / "reference_code"
    / "STS2R_formal_baselines_100ep_w8"
    / "scripts"
    / "03_run_formal_baselines.py"
)

source = ENTRY.read_text(encoding="utf-8-sig")
adapter_file = __file__
__file__ = str(ENTRY)
try:
    exec(compile(source, str(ENTRY), "exec"), globals(), globals())
finally:
    __file__ = adapter_file

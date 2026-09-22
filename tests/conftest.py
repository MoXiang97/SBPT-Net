"""Test-environment compatibility helpers."""

import os


# Windows Conda environments can load separate OpenMP runtimes through PyTorch
# and SciPy. This setting keeps smoke tests from aborting before Python can
# report a normal test result. It is scoped to tests and does not change the
# released model code.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

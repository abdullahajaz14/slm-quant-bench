"""slmbench.adapters: importing this package registers every adapter.

Each adapter module calls @register at import time, filling
base.ADAPTERS; the runner only needs `from .adapters import base`
plus this package import side effect.
"""

from . import base
from . import cuad, hotpotqa, cnndm, truthfulqa, ukps  # noqa: F401  (registration side effect)

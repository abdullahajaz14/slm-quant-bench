"""Make src/ importable when running pytest without `pip install -e .`."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

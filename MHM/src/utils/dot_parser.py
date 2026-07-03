import sys
from pathlib import Path

_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))

from common.utils.dot_parser import LooseDotParser

__all__ = ["LooseDotParser"]

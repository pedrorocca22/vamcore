"""Hace importable la raíz del repo (vamcore) desde las pruebas sin instalar nada."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

"""Recursos incluidos en el paquete (modelos de ejemplo)."""
from pathlib import Path

RESOURCES = Path(__file__).resolve().parent


def example_stl() -> Path:
    """Vasculatura trifurcada de ejemplo (la pieza de referencia de los benchmarks y las pruebas)."""
    return RESOURCES / "trifurcatedvasculature.stl"

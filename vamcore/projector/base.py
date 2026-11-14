"""Interfaz de proyector (docs/MIGRACION.md §4, principio 4): el CAL no sabe qué hay debajo.

Convención: volumen ``x[ix, iy, iz]``, sinograma ``b[u, ángulo, z]``; ambos ``torch.Tensor`` float32 en
``device``. ``adjoint`` es la traspuesta de ``forward`` (⟨Ax, y⟩ = ⟨x, Aᵀy⟩, ver tests/test_projector_*.py).
Ninguna de las dos modifica su entrada.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Projector(Protocol):
    vol_shape: tuple   # (n_x, n_y, n_z)
    sino_shape: tuple  # (n_u, n_angles, n_z)
    device: object

    def forward(self, x): ...

    def adjoint(self, b): ...

    def close(self) -> None: ...

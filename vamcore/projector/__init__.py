"""Proyectores (``base.Projector``). Se importan bajo demanda para que ``import vamcore`` no cargue torch.

- ``"torch"`` (por defecto) — **Prisma**: método de Joseph como matriz dispersa 2D compartida por las capas
  (cuSPARSE), retroproyección por vóxel; admite la absorción de la resina. Funciona en GPU o en CPU.
- ``"torch_grid"``: el mismo operador con ``grid_sample`` y adjunto por autograd. Mucho más lento; solo referencia.
"""
from .._backend import default_device
from .base import Projector  # noqa: F401

BACKENDS = ("torch", "torch_grid")


def make_projector(vol_shape, angles_deg, backend: str = "torch", device=None, **kw):
    device = default_device(device)
    if backend == "torch":
        from .torch_parallel import SparseParallel3D

        return SparseParallel3D(vol_shape, angles_deg, device=device, **kw)
    if backend == "torch_grid":
        from .torch_parallel import TorchParallel3D

        return TorchParallel3D(vol_shape, angles_deg, device=device, **kw)
    raise ValueError(f"backend de proyector desconocido: {backend!r} (usa uno de {BACKENDS})")

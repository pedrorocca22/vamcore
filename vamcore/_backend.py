"""Despacho mínimo NumPy / PyTorch sin importar torch si no está instalado."""
from __future__ import annotations

import numpy as np

try:  # torch es opcional en vamcore (las pruebas ligeras y el CI no lo tienen)
    import torch
except ImportError:  # pragma: no cover
    torch = None


def is_torch(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def default_device(device=None):
    """Dispositivo de cálculo: el indicado, o CUDA si está disponible, o CPU (ruta PyTorch pura, lenta pero válida)."""
    if device is not None:
        return device
    return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"


def to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if is_torch(x) else np.asarray(x)

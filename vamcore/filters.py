"""Filtros rampa para el sinograma (inicialización FBP del CAL). NumPy o torch (FFT en el dispositivo del tensor).

Adaptado de VAMToolbox (GPL-3.0, ``vamtoolbox/util/data.py``: filterSinogram, _get_fourier_filter), que a su vez
sigue a Kak & Slaney, *Principles of Computerized Tomographic Imaging*, cap. 3, ec. 61 (rampa construida en el
espacio real para evitar el sesgo de la componente continua). Mismos resultados; la diferencia es que se filtran
todas las capas Z a la vez en lugar de una a una.

Sinograma ``[u, ángulo]`` o ``[u, ángulo, z]``; se filtra a lo largo de ``u`` (eje 0).
"""
from __future__ import annotations

import numpy as np

from ._backend import is_torch, torch

FILTERS = ("ram-lak", "shepp-logan", "cosine", "hamming", "hanning", "ram-lak_freq", "ram-lak_scikit")
ALIASES = {"hann": "hanning", "ramlak": "ram-lak", "shepplogan": "shepp-logan"}


def normalize_name(name):
    """Nombre canónico del filtro o None (sin filtro). Lanza ValueError si no existe."""
    if name is None or str(name).lower() == "none":
        return None
    n = str(name).lower()
    n = ALIASES.get(n, n)
    if n not in FILTERS:
        raise ValueError(f"Unknown filter: {name}")
    return n


def padded_size(n_u: int) -> int:
    return max(64, int(2 ** np.ceil(np.log2(2 * n_u))))


def fourier_filter(size: int, name: str) -> np.ndarray:
    """Respuesta en frecuencia (longitud ``size``, orden de la FFT) del filtro ``name`` (canónico)."""
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int), np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    h = 2 * np.real(np.fft.fft(f))  # rampa (Ram-Lak)
    if name == "ram-lak":
        pass
    elif name == "shepp-logan":
        omega = np.pi * np.fft.fftfreq(size)[1:]
        h[1:] *= np.sin(omega) / omega
    elif name == "cosine":
        h *= np.fft.fftshift(np.sin(np.linspace(0, np.pi, size, endpoint=False)))
    elif name == "hamming":
        h *= np.fft.fftshift(np.hamming(size))
    elif name == "hanning":
        h *= np.fft.fftshift(np.hanning(size))
    elif name == "ram-lak_freq":
        h /= (np.pi / 4) ** 2
    elif name == "ram-lak_scikit":
        h *= np.pi / 2
    else:  # pragma: no cover - normalize_name lo impide
        raise ValueError(name)
    return h


def filter_sinogram(sino, name):
    """Filtra el sinograma a lo largo de u. NumPy → float64; torch → dtype de entrada (float32 en el CAL)."""
    name = normalize_name(name)
    if name is None:
        return sino
    if sino.ndim not in (2, 3):
        raise ValueError("El sinograma debe tener 2 o 3 dimensiones")
    n_u = sino.shape[0]
    size = padded_size(n_u)
    h = fourier_filter(size, name)
    shape = (size,) + (1,) * (sino.ndim - 1)
    if is_torch(sino):
        ht = torch.as_tensor(h, dtype=sino.dtype, device=sino.device).reshape(shape)
        spec = torch.fft.fft(sino, n=size, dim=0) * ht
        return torch.fft.ifft(spec, dim=0)[:n_u].real.to(sino.dtype)
    spec = np.fft.fft(np.asarray(sino, dtype=float), n=size, axis=0) * h.reshape(shape)
    return np.real(np.fft.ifft(spec, axis=0)[:n_u])

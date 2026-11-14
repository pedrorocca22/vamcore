"""Operaciones elementales: sigmoide, discretización, círculo inscrito y máscaras gel/vacío.

Adaptado de VAMToolbox (GPL-3.0): ``vamtoolbox/util/data.py`` (sigmoid, discretize, clipToCircle) y
``vamtoolbox/geometry.py`` (getCircleMask, getInds). Mismos resultados; funciones sin efectos colaterales salvo
``clip_to_circle(..., inplace=True)``.
"""
from __future__ import annotations

import numpy as np

from ._backend import is_torch, torch


def sigmoid(x, g: float):
    """1 / (1 + exp(-x/g)). NumPy → float64; torch → mismo dtype y dispositivo."""
    if is_torch(x):
        return torch.sigmoid(x / float(g))
    return (1.0 / (1.0 + np.exp(-np.asarray(x) * (1.0 / g)))).astype(float)


def discretize(x, bit_depth: int, value_range, output_dtype=float):
    """Cuantiza a 2**bit_depth niveles equiespaciados en [min, max] (redondeo hacia abajo, como np.digitize)."""
    lo, hi = value_range
    bins = np.linspace(lo, hi, 2**bit_depth, endpoint=True)
    if is_torch(x):
        tb = torch.as_tensor(bins, dtype=x.dtype, device=x.device)
        idx = torch.bucketize(x, tb, right=True) - 1
        return tb[idx]
    idx = np.digitize(x, bins=bins) - 1
    return bins[idx].astype(output_dtype)


def circle_mask(n_x: int, n_y: int, device=None):
    """Máscara booleana [ix, iy] del círculo inscrito (coordenadas normalizadas a [-1, 1], frontera incluida).

    Se calcula siempre en NumPy float64 (la definición de referencia, igual que ``getCircleMask``) y, si se pide
    ``device``, se copia allí. Los vóxeles cuyo centro cae justo sobre la circunferencia dependen del redondeo:
    VAMToolbox los incluía en la ruta torch (float32) y los excluía en la NumPy, así que su bucle y su reconstrucción
    final usaban máscaras distintas. Aquí hay una sola.
    """
    cx = np.linspace(-1, 1, n_x)
    cy = np.linspace(-1, 1, n_y)
    m = (cx[:, None] ** 2 + cy[None, :] ** 2) <= 1
    if device is not None:
        return torch.as_tensor(m, device=device)
    return m


def clip_to_circle(x, inplace: bool = False):
    """Anula lo que queda fuera del círculo inscrito en los dos primeros ejes."""
    if is_torch(x):
        m = circle_mask(x.shape[0], x.shape[1], device=x.device)
        if x.ndim == 3:
            m = m[:, :, None]
        return x.masked_fill_(~m, 0) if inplace else x.masked_fill(~m, 0)
    out = x if inplace else np.array(x, copy=True)
    m = circle_mask(x.shape[0], x.shape[1])
    if out.ndim == 3:
        m = m[:, :, None]
    out[np.broadcast_to(~m, out.shape)] = 0
    return out


def get_inds(target, threshold: float | None = None):
    """Máscaras (gel, vacío) dentro del círculo inscrito.

    threshold=None reproduce ``vamtoolbox.geometry.getInds`` (gel: > 0, vacío: == 0). Con ocupación parcial
    (vamcore.voxelize) usar threshold=0.5: gel ≥ 0.5, vacío < 0.5, que es lo que hace el pipeline.
    """
    t = np.asarray(target)
    m = circle_mask(t.shape[0], t.shape[1])
    if t.ndim == 3:
        m = np.broadcast_to(m[:, :, None], t.shape)
    if threshold is None:
        gel, void = t > 0, t == 0
    else:
        gel, void = t >= threshold, t < threshold
    return gel & m, void & m


def metric_masks(target, band: float = 0.05, min_core_frac: float = 0.25):
    """Máscaras (gel, vacío, info) para evaluar la dosis con ocupación parcial.

    Un vóxel con ocupación entre ``band`` y ``1 − band`` está medio dentro de la pieza: su dosis ideal es
    intermedia y no se puede exigir que quede por encima (gel) ni por debajo (vacío) del umbral. Clasificarlo
    al 50 % inflaba VER, PW y el solape por capa (y castigaba el Score Bio) precisamente cuando el objetivo es
    más fiel. Aquí se excluye de ambos conjuntos (banda de guarda).

    Con objetivos binarios (voxelizador «legacy») el resultado es idéntico a ``get_inds(target)``. Si la pieza
    es tan fina que los vóxeles llenos son menos de ``min_core_frac`` de los que tienen ocupación ≥ 0,5, se vuelve
    al corte en 0,5 (la banda dejaría sin gel que evaluar) y ``info['rule']`` lo indica.
    """
    t = np.asarray(target)
    m = circle_mask(t.shape[0], t.shape[1])
    if t.ndim == 3:
        m = np.broadcast_to(m[:, :, None], t.shape)
    gel, void = (t >= 1 - band) & m, (t <= band) & m
    n_half = int(((t >= 0.5) & m).sum())
    excluded = int((m & ~gel & ~void).sum())
    if n_half and gel.sum() < min_core_frac * n_half:
        gel, void = (t >= 0.5) & m, (t < 0.5) & m
        return gel, void, {"rule": "threshold_0.5 (pieza demasiado fina para la banda)", "band": band,
                           "excluded_partial": 0, "n_gel": int(gel.sum()), "n_void": int(void.sum())}
    return gel, void, {"rule": "guard_band" if excluded else "binary", "band": band,
                       "excluded_partial": excluded, "n_gel": int(gel.sum()), "n_void": int(void.sum())}

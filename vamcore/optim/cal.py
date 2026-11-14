"""CAL (Computed Axial Lithography) en PyTorch sobre la interfaz ``Projector``.

Adaptado de VAMToolbox (GPL-3.0, ``vamtoolbox/optimizer/CAL.py``, ruta CUDA de VAM Studio). Referencias:
Kelly et al., *Science* 363 (2019) 1075; Bhattacharya et al., *Addit. Manuf.* 47 (2021).

Mismo algoritmo, paso a paso:
  b₀ = max(0, filtro(A·t))                       inicialización FBP
  x  = Aᵀb / max                                 dosis normalizada
  repetir:
    Δx = σ((x − d_h)/g) − t                      error tras el umbral sigmoide
    Δb = A·Δx
    b  = max(0, b − lr·[(1−m)·Δb + m·Δb_prev/(1 − m^(k+1))])
    x  = Aᵀb / max
  final: b ← b/max(b), x ← Aᵀb / max

Diferencias con VAMToolbox: todo en la GPU (la inicialización y la reconstrucción final ya no pasan por NumPy),
el proyector es intercambiable, la información de parada se devuelve en el resultado (no se escribe en las opciones)
y el VER de parada puede usar la banda de guarda (``StopCriteria.ver_mask``). Sin regularización TV (el pipeline no
la usa).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .._backend import torch
from ..config import CALOptions
from ..filters import filter_sinogram
from ..ops import discretize
from .stopping import StopRule


@dataclass
class CALResult:
    sinogram: np.ndarray        # [u, ángulo, z], normalizado a [0, 1]
    reconstruction: np.ndarray  # [x, y, z], normalizada a máx 1
    ver: np.ndarray             # VER de parada por iteración (NaN donde no se evaluó)
    stop: dict
    pw: np.ndarray | None = None  # PW por iteración (máx vacío − mín gel; NaN donde no se evaluó)
    times: dict = field(default_factory=dict)


def _ver_masks(t, mode: str):
    if mode == "half":
        gel = t > 0.5
        void = ~gel
        n = t.numel()  # como VAMToolbox: todos los vóxeles del volumen
    elif mode == "guard_band":
        gel, void = t >= 0.95, t <= 0.05
        n = int(gel.sum() + void.sum())
    else:
        raise ValueError(f"ver_mask desconocido: {mode!r}")
    if not bool(gel.any()):
        gel, void, n = t > 0.5, ~(t > 0.5), t.numel()
    return gel, void, float(n)


def _ver_pw(x, gel, void, n):
    g_min = x[gel].min()
    xv = x[void]
    return (xv >= g_min).sum().item() / n, (xv.max() - g_min).item()


def run_cal(target, angles_deg, opts: CALOptions, *, exit_param: float | None = None, projector=None,
            device=None, callback=None, backend: str = "torch", projector_kwargs: dict | None = None,
            init=None) -> CALResult:
    """init: sinograma inicial (forma del proyector) en lugar de la FBP; p. ej. la FBP ideal corregida por el vial."""
    """Optimiza el sinograma para el objetivo ``target[x, y, z]`` (ocupación 0…1). Necesita ``opts.d_h``."""
    if torch is None:
        raise ImportError("run_cal necesita PyTorch")
    if opts.d_h is None:
        raise ValueError("opts.d_h es obligatorio (usa vamcore.dose.suggest_d_h)")
    t0 = time.perf_counter()
    times = {}
    own = projector is None
    if own:
        from ..projector import make_projector

        projector = make_projector(np.shape(target), angles_deg, backend=backend, device=device,
                                   **(projector_kwargs or {}))
    A = projector
    try:
        t = torch.as_tensor(np.asarray(target), dtype=torch.float32, device=A.device)
        n_iter, lr, m, g, d_h = opts.n_iter, opts.learning_rate, opts.momentum, float(opts.sigmoid), float(opts.d_h)

        if init is not None:
            b = torch.clamp(torch.as_tensor(np.asarray(init) if not torch.is_tensor(init) else init,
                                            dtype=torch.float32, device=A.device), min=0)
            if tuple(b.shape) != tuple(A.sino_shape):
                raise ValueError(f"init: forma {tuple(b.shape)}, se esperaba {tuple(A.sino_shape)}")
        else:
            b = torch.clamp(filter_sinogram(A.forward(t), opts.filter), min=0)
        x = A.adjoint(b)
        x = x / x.max()
        gel, void, n = _ver_masks(t, opts.stop.ver_mask)
        ver = np.full(n_iter, np.nan)
        pw = np.full(n_iter, np.nan)
        ver[0], pw[0] = _ver_pw(x, gel, void, n)
        times["init_s"] = time.perf_counter() - t0

        rule = StopRule(opts.stop, n_iter, exit_param)
        delta_b_prev = torch.zeros_like(b)
        it = 0
        for it in range(1, n_iter):
            delta_x = torch.sigmoid((x - d_h) / g) - t
            delta_b = A.forward(delta_x)
            grad = delta_b * (1 - m) + delta_b_prev * m / (1 - m ** (it + 1))
            b = torch.clamp(b - lr * grad, min=0)
            if opts.bit_depth is not None:
                b = discretize(b, opts.bit_depth, [0.0, float(b.max())])
            x = A.adjoint(b)
            x = x / x.max()
            delta_b_prev = delta_b
            if it % opts.ver_every == 0 or it == n_iter - 1:
                ver[it], pw[it] = _ver_pw(x, gel, void, n)
                if callback is not None:
                    callback(it, ver[it])
                if rule.update(it, ver[it], pw[it]):
                    break
        times["loop_s"] = time.perf_counter() - t0 - times["init_s"]

        b = torch.clamp(b, min=0)
        if float(b.max()) > 0:
            b = b / b.max()
        x = A.adjoint(b)
        if float(x.max()) > 0:
            x = x / x.max()
        res = CALResult(b.cpu().numpy(), x.cpu().numpy(), ver, rule.info(it + 1, ver[0]), pw=pw)
        times["total_s"] = time.perf_counter() - t0
        res.times = times
        return res
    finally:
        if own:
            A.close()

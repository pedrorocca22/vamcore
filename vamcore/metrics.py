"""Métricas de dosis: VER, PW, IPDR y CV (globales o por capa Z). NumPy o torch.

Definiciones de VAMToolbox (GPL-3.0, ``vamtoolbox/metrics.py``), que a su vez siguen las publicadas
(Kelly et al., Science 2019; Bhattacharya et al., Addit. Manuf. 2021):

  VER  = nº de vóxeles de vacío con dosis ≥ mínima dosis de gel / (nº gel + nº vacío)
  PW   = máx(dosis de vacío) − mín(dosis de gel)     (negativo = hay ventana de proceso)
  IPDR = máx(dosis de gel) − mín(dosis de gel)
  CV   = desviación típica / media de la dosis de gel

``gel`` y ``void`` son máscaras booleanas con la forma del volumen (ver ``ops.get_inds``).

Diferencias con VAMToolbox: con ``layerwise=True`` VAMToolbox fallaba (máscara 3D sobre un corte 2D) o, en CV,
repetía el valor global en cada capa. Aquí se calcula de verdad por capa; en capas sin gel o sin vacío el valor es
NaN. El denominador del VER por capa es el total del volumen (como en VAMToolbox), de modo que la suma de los
VER por capa es el VER «por capas» global.
"""
from __future__ import annotations

import numpy as np

from ._backend import is_torch, torch


def _f(v):
    return float(v.item()) if is_torch(v) else float(v)


def _layers(recon, gel, void, fn):
    nz = recon.shape[2]
    out = np.full(nz, np.nan)
    for z in range(nz):
        g, v = gel[:, :, z], void[:, :, z]
        if bool(g.any()) and bool(v.any()):
            out[z] = fn(recon[:, :, z], g, v)
    return out


def ver(recon, gel, void, layerwise: bool = False):
    n = _f(gel.sum()) + _f(void.sum())

    def one(r, g, v):
        return _f((r[v] >= r[g].min()).sum()) / n

    return _layers(recon, gel, void, one) if layerwise else one(recon, gel, void)


def pw(recon, gel, void, layerwise: bool = False):
    def one(r, g, v):
        return _f(r[v].max()) - _f(r[g].min())

    return _layers(recon, gel, void, one) if layerwise else one(recon, gel, void)


def ipdr(recon, gel, void, layerwise: bool = False):
    def one(r, g, v):
        return _f(r[g].max()) - _f(r[g].min())

    return _layers(recon, gel, void, one) if layerwise else one(recon, gel, void)


def cv(recon, gel, void=None, layerwise: bool = False):
    def one(r, g, v=None):
        d = r[g]
        if is_torch(d):
            d = d.double()
            return _f(d.std(unbiased=False)) / _f(d.mean())
        return float(np.std(d) / np.mean(d))

    if not layerwise:
        return one(recon, gel)
    void = void if void is not None else ~gel
    nz = recon.shape[2]
    out = np.full(nz, np.nan)
    for z in range(nz):
        g = gel[:, :, z]
        if bool(g.any()):
            out[z] = one(recon[:, :, z], g)
    return out


def all_metrics(recon, gel, void) -> dict:
    return {"ver": ver(recon, gel, void), "pw": pw(recon, gel, void),
            "ipdr": ipdr(recon, gel, void), "cv": cv(recon, gel)}

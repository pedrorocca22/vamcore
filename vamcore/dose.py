"""Umbral de gelificación sugerido (d_h) a partir del grosor de la pieza.

Adaptado de ``vamtoolbox/geometry.py`` (TargetGeometry.suggest_d_h, escrito para VAM Studio sobre VAMToolbox,
GPL-3.0). Mismo criterio: grosor medio = media de la transformada de distancia euclídea dentro del gel
(ocupación ≥ 0,5), y una escalera de d_h calibrada con los benchmarks de morfologías del proyecto.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_D_H = 0.85
# (grosor medio máximo en vóxeles, d_h, etiqueta)
LADDER = ((1.4, 0.82, "CRITICAL (Ultra-thin)"), (1.8, 0.85, "THIN (Fragile)"),
          (3.0, 0.88, "MEDIUM (Balanced)"), (float("inf"), 0.90, "ROBUST (Massive)"))


@dataclass(frozen=True)
class DhSuggestion:
    d_h: float
    mean_thickness_vox: float | None
    tag: str


def mean_thickness(occupancy) -> float | None:
    from scipy import ndimage

    gel = np.asarray(occupancy) >= 0.5
    if not gel.any():
        return None
    return float(np.mean(ndimage.distance_transform_edt(gel)[gel]))


def suggest_d_h(occupancy) -> DhSuggestion:
    t = mean_thickness(occupancy)
    if t is None:
        return DhSuggestion(DEFAULT_D_H, None, "EMPTY")
    for limit, d_h, tag in LADDER:
        if t < limit:
            return DhSuggestion(d_h, t, tag)
    raise AssertionError  # pragma: no cover

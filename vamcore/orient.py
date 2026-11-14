"""Auto-orientación geométrica de la pieza respecto al eje de giro (Z).

Adaptado de ``CAL_run.py`` (``find_optimal_orientation``, VAM Studio, GPL-3.0). Criterio: minimizar el ancho medio
de la pieza proyectada sobre 72 direcciones del plano XY, es decir, hacerla lo más estrecha posible vista desde el
proyector (menos radio alrededor del eje: menos vóxeles, cabe mejor en la zona imprimible y, cuando se modele la
absorción, recorridos de luz más cortos). Búsqueda en rejilla de Rx, Ry ∈ [−90°, 90°] cada 10° y refinado con
Nelder-Mead. El giro en Z no se busca: girar alrededor del eje de rotación no cambia nada físico.

Novedad respecto al CLI: **restricciones**. Las orientaciones que superan la altura máxima de impresión o el radio
máximo (zona imprimible del vial o área de trabajo) se descartan; si ninguna cabe, se devuelve la menos mala y
``feasible = False``. El radio se mide como lo hace el pipeline: desde el centro de la caja envolvente ya orientada.

Convención: rotación en el sistema del STL (Z = eje de giro), R = Rz(rz)·Ry(ry)·Rx(rx), la de
``vamcore.pipeline._center_and_rotate``. Aquí rz = 0 y R = Ry(ry)·Rx(rx).

**Límite:** es solo geométrica; no evalúa la dosis ni el Score Bio.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_DIRECTIONS = 72


def rot_x(a):
    a = np.radians(a)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def rot_y(a):
    a = np.radians(a)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rotation(rx: float, ry: float) -> np.ndarray:
    return rot_y(ry) @ rot_x(rx)


_THETAS = np.linspace(0, np.pi, N_DIRECTIONS, endpoint=False)
_DIRS = np.column_stack([np.cos(_THETAS), np.sin(_THETAS)])


def mean_width(v_xy: np.ndarray) -> float:
    p = v_xy @ _DIRS.T
    return float((p.max(axis=0) - p.min(axis=0)).mean())


def extents_after(v: np.ndarray, R: np.ndarray) -> tuple[float, float]:
    """(altura en Z, radio máximo respecto al centro de la caja en XY) tras aplicar R."""
    w = v @ R.T
    lo, hi = w.min(axis=0), w.max(axis=0)
    c = (lo[:2] + hi[:2]) / 2
    return float(hi[2] - lo[2]), float(np.max(np.hypot(w[:, 0] - c[0], w[:, 1] - c[1])))


@dataclass
class OrientResult:
    rx: float
    ry: float
    rotation: list                 # matriz 3×3 en el sistema del STL
    width_initial_mm: float
    width_mm: float
    reduction_pct: float
    height_mm: float
    radius_mm: float
    feasible: bool
    initial_feasible: bool
    limits: dict = field(default_factory=dict)
    candidates: int = 0
    grid: list | None = None       # [(rx, ry, ancho), …] de la rejilla, con return_grid=True


def find_orientation(vertices, max_height_mm: float | None = None, max_radius_mm: float | None = None,
                     search_range: float = 90.0, step: float = 10.0, refine: bool = True,
                     return_grid: bool = False) -> OrientResult:
    v = np.asarray(vertices, dtype=np.float64)
    v = v - v.mean(axis=0)
    if len(v) > 60000:  # suficiente para medir anchos; acelera mallas grandes
        v = v[np.random.default_rng(0).choice(len(v), 60000, replace=False)]

    def cost(rx, ry):
        R = rotation(rx, ry)
        w = mean_width((v @ R.T)[:, :2])
        h, r = extents_after(v, R)
        excess = 0.0
        if max_height_mm is not None:
            excess += max(0.0, h - max_height_mm)
        if max_radius_mm is not None:
            excess += max(0.0, r - max_radius_mm)
        return w + 1e3 * excess, w, h, r, excess == 0.0

    w0 = mean_width(v[:, :2])
    _, _, h0, r0, ok0 = cost(0.0, 0.0)
    grid = np.arange(-search_range, search_range + 1e-9, step)
    best = (float("inf"), 0.0, 0.0)
    grid_rows = []
    for rx in grid:
        for ry in grid:
            c, w_, _, _, _ = cost(rx, ry)
            if return_grid:
                grid_rows.append((float(rx), float(ry), w_))
            if c < best[0] - 1e-12:
                best = (c, float(rx), float(ry))
    rx, ry = best[1], best[2]
    if refine:
        try:
            from scipy.optimize import minimize

            res = minimize(lambda p: cost(p[0], p[1])[0], x0=[rx, ry], method="Nelder-Mead",
                           options={"xatol": 0.1, "fatol": 1e-6, "maxiter": 300})
            if res.fun < best[0]:
                rx, ry = float(res.x[0]), float(res.x[1])
        except ImportError:  # pragma: no cover
            pass
    # Si no mejora de forma apreciable, no se gira (evita giros espurios por empates numéricos)
    _, w, h, r, ok = cost(rx, ry)
    if ok0 and w >= w0 * (1 - 1e-4):
        rx, ry, w, h, r, ok = 0.0, 0.0, w0, h0, r0, ok0
    return OrientResult(
        rx=rx, ry=ry, rotation=rotation(rx, ry).tolist(), width_initial_mm=w0, width_mm=w,
        reduction_pct=100 * (1 - w / w0) if w0 > 0 else 0.0, height_mm=h, radius_mm=r, feasible=ok,
        initial_feasible=ok0, limits={"max_height_mm": max_height_mm, "max_radius_mm": max_radius_mm},
        candidates=len(grid) ** 2, grid=grid_rows if return_grid else None,
    )

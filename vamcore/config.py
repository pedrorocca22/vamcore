"""Configuración explícita y sin estado global (docs/MIGRACION.md §4, principio 5).

Cada ejecución construye sus propios objetos: no hay diccionarios de clase que se muten (el fallo §3.8 de
VAMToolbox). Unidades en el nombre del campo. Las usa vamcore.optim (CALOptions, StopCriteria); el resto es el contrato que
irá adoptando ``pipeline.py``. Los valores por defecto replican los del pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ProjectorSpec:
    pixel_um: float = 14.3            # tamaño del píxel en el plano del vial
    build_diameter_mm: float = 20.0
    build_height_mm: float = 25.0
    n_angles: int = 360
    telecentric: bool = True          # rayos paralelos (único modelo soportado por ahora)

    def angles_deg(self):
        import numpy as np

        return np.linspace(0.0, 360.0 - 360.0 / self.n_angles, self.n_angles)


@dataclass(frozen=True)
class VoxelGrid:
    pitch_mm: float
    n_xy: int
    n_z: int

    @property
    def shape(self):
        return (self.n_xy, self.n_xy, self.n_z)


@dataclass(frozen=True)
class StopCriteria:
    ver_threshold: float = 0.0005     # para si VER ≤ umbral
    stall_patience: int = 10          # evaluaciones del VER sin mejora (0 = desactivado)
    stall_rel_tol: float = 0.01
    stall_min_iter: int = 50
    max_time_s: float | None = None
    # Máscaras del VER que decide la parada: "half" clasifica al 50 % sobre todo el volumen (idéntico a VAMToolbox);
    # "guard_band" excluye los vóxeles parciales (0,05 < ocupación < 0,95), como las métricas del Score Bio.
    ver_mask: str = "half"
    # Parada por ventana de proceso: si se da, para cuando PW = máx(dosis vacío) − mín(dosis gel) ≤ −pw_margin
    # (sobre las mismas máscaras). En ese modo no se usa el umbral de VER y el estancamiento se mide sobre el PW:
    # para si el mejor PW no baja al menos stall_pw_tol en stall_patience evaluaciones.
    pw_margin: float | None = None
    stall_pw_tol: float = 0.001


@dataclass(frozen=True)
class CALOptions:
    n_iter: int = 250
    learning_rate: float = 0.010
    momentum: float = 0.3
    sigmoid: float = 0.01
    filter: str | None = "cosine"
    d_h: float | None = None          # None → dose.suggest_d_h
    bit_depth: int | None = None
    ver_every: int = 5
    stop: StopCriteria = field(default_factory=StopCriteria)

    @staticmethod
    def for_resolution(res: int) -> "CALOptions":
        """Escalera de iteraciones y learning rate que usa el pipeline según la resolución."""
        if res >= 512:
            return CALOptions(n_iter=800, learning_rate=0.020)
        if res >= 256:
            return CALOptions(n_iter=500, learning_rate=0.015)
        if res <= 64:
            return CALOptions(n_iter=80, learning_rate=0.010)
        return CALOptions(n_iter=250, learning_rate=0.010)

    def with_(self, **kw) -> "CALOptions":
        return replace(self, **kw)


@dataclass(frozen=True)
class PrintJob:
    stl_path: str
    resolution: int = 128             # nº de capas en Z (si no se da voxel_pitch_um)
    voxel_pitch_um: float | None = None
    rotation_deg: tuple = (0.0, 0.0, 0.0)
    projector: ProjectorSpec = field(default_factory=ProjectorSpec)
    cal: CALOptions | None = None     # None → CALOptions.for_resolution(resolution)
    vial: dict | None = None          # parámetros de vamcore.optics.vial.VialOptics

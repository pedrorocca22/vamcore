"""
Corrección de la refracción del vial (impresión sin baño de index matching)
==========================================================================

Modelo
------
Proyector telecéntrico: rayos paralelos, perpendiculares al eje del vial (eje Z). Un vial cilíndrico
de radio exterior R_o y pared de grosor w (radio interior R_i = R_o - w) se comporta como una lente
cilíndrica: desvía los rayos en el plano XY y no en Z.

Por la simetría circular se conserva el invariante  n · b  (índice × parámetro de impacto, fórmula de
Bouguer). Un rayo que sale del proyector a una distancia u del eje recorre la resina:

  * a una distancia  t = n_out · u / n_resin  del eje, y
  * girado un ángulo δ(u) = (i1 - r1) + (i2 - r2) respecto a su dirección original, con
        sin i1 = u / R_o                        (aire  → vidrio, superficie exterior)
        sin r1 = n_out·u / (n_vial · R_o)
        sin i2 = n_out·u / (n_vial · R_i)       (vidrio → resina, superficie interior)
        sin r2 = n_out·u / (n_resin · R_i)

Dentro de la resina (homogénea) el rayo es una recta, es decir, una línea más del sinograma ideal.
Por eso no hace falta cambiar el optimizador: se optimiza como si no hubiera vial y después cada
imagen se remapea. En la convención «parallel3d» del proyector Prisma (dirección del rayo (sin φ, -cos φ, 0),
eje del detector (cos φ, sin φ, 0)), el píxel (θ, u) del proyector debe mostrar el valor del
sinograma ideal en

        φ = θ - δ(u),     t = n_out · u / n_resin

dividido por la transmisión de Fresnel T(u) de las dos interfaces. La relación es invariante si la
máquina invierte a la vez el sentido de giro y el espejo de la imagen (lo único que importa es que
sin vial la proyección ya sea correcta).

Límites del modelo
------------------
Proyector telecéntrico, vial cilíndrico perfecto y centrado en el eje de giro, sin absorción ni
reflexiones múltiples. Los rayos con |u| ≥ u_max no llegan a la resina (salen del vial o sufren
reflexión total), lo que reduce el radio imprimible a  t_max = n_out·u_max / n_resin.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Índices de refracción aproximados a 405 nm. Son orientativos: conviene confirmar con el fabricante.
VIAL_MATERIALS = {
    "borosilicate": 1.483,   # Borofloat 33 / vidrio de laboratorio tipo 3.3 (n_d ≈ 1.472)
    "fused_silica": 1.470,   # cuarzo fundido
    "soda_lime": 1.530,      # vidrio común (n_d ≈ 1.52)
    "pmma": 1.506,           # acrílico (n_d ≈ 1.49)
    "polystyrene": 1.620,    # poliestireno (n_d ≈ 1.59)
}
RESIN_PRESETS = {
    "water": 1.343,
    "gelma": 1.350,          # hidrogeles GelMA 5–10 % (aprox.)
    "acrylate": 1.480,       # resinas acrílicas típicas (PEGDA ≈ 1.47)
}


@dataclass(frozen=True)
class VialOptics:
    """Geometría y materiales del vial. Longitudes en mm."""

    outer_diameter_mm: float
    wall_thickness_mm: float
    n_vial: float
    n_resin: float
    n_outside: float = 1.0   # aire; >1 si hay un baño parcial

    # ── derivados ───────────────────────────────────────────────────────────
    @property
    def r_outer(self) -> float:
        return self.outer_diameter_mm / 2.0

    @property
    def r_inner(self) -> float:
        return self.r_outer - self.wall_thickness_mm

    def validate(self) -> None:
        if self.outer_diameter_mm <= 0:
            raise ValueError("El diámetro exterior del vial debe ser > 0")
        if not (0 <= self.wall_thickness_mm < self.r_outer):
            raise ValueError("El grosor de pared debe estar entre 0 y el radio exterior")
        for name in ("n_vial", "n_resin", "n_outside"):
            if getattr(self, name) < 1.0:
                raise ValueError(f"{name} debe ser ≥ 1")

    @property
    def u_max(self) -> float:
        """Mayor |u| (en el plano del proyector) cuyo rayo llega a la resina."""
        n0, nv, nr = self.n_outside, self.n_vial, self.n_resin
        return min(self.r_outer, nv * self.r_inner / n0, nr * self.r_inner / n0)

    @property
    def printable_radius(self) -> float:
        """Radio (mm, medido en la resina) alcanzable por la luz: la pieza debe caber dentro."""
        return self.n_outside * self.u_max / self.n_resin

    def summary(self) -> dict:
        d = asdict(self)
        d.update(r_inner_mm=self.r_inner, u_max_mm=self.u_max, printable_radius_mm=self.printable_radius,
                 printable_diameter_mm=2 * self.printable_radius)
        return d

    # ── trazado analítico ───────────────────────────────────────────────────
    def ray_map(self, u):
        """
        Para posiciones u (mm, plano del proyector, con signo) devuelve:
          t      : distancia con signo a la que el rayo recorre la resina (mm)
          delta  : giro de la dirección del rayo (rad, con el mismo signo que u)
          T      : transmisión de Fresnel (potencia, luz no polarizada) de ambas interfaces
          valid  : el rayo llega a la resina
        """
        self.validate()
        u = np.asarray(u, dtype=np.float64)
        n0, nv, nr = self.n_outside, self.n_vial, self.n_resin
        Ro, Ri = self.r_outer, self.r_inner
        a = np.abs(u)
        valid = a < self.u_max * (1 - 1e-12)
        a = np.where(valid, a, 0.0)

        i1 = np.arcsin(a / Ro)
        r1 = np.arcsin(n0 * a / (nv * Ro))
        if Ri > 0 and self.wall_thickness_mm > 0:
            i2 = np.arcsin(n0 * a / (nv * Ri))
            r2 = np.arcsin(n0 * a / (nr * Ri))
            T = _fresnel_T(n0, nv, i1, r1) * _fresnel_T(nv, nr, i2, r2)
            delta = (i1 - r1) + (i2 - r2)
        else:  # sin pared: una sola interfaz exterior → resina
            r1 = np.arcsin(n0 * a / (nr * Ro))
            T = _fresnel_T(n0, nr, i1, r1)
            delta = i1 - r1

        t = np.sign(u) * n0 * a / nr
        delta = np.sign(u) * delta
        return np.where(valid, t, 0.0), np.where(valid, delta, 0.0), np.where(valid, T, 0.0), valid


def _fresnel_T(n1, n2, i, r):
    """Transmitancia en potencia (media s/p, luz no polarizada) en una interfaz plana local."""
    ci, cr = np.cos(i), np.cos(r)
    rs = (n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)
    rp = (n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)
    return 1.0 - 0.5 * (rs**2 + rp**2)


def projector_u_grid(vial: VialOptics, pitch_mm: float) -> np.ndarray:
    """Posiciones u (mm) de las columnas de imagen, centradas en el eje y cubriendo el vial completo."""
    half = int(np.ceil(vial.r_outer / pitch_mm))
    return (np.arange(-half, half + 1)) * pitch_mm


def correct_sinogram(
    sino: np.ndarray,
    angles_deg: np.ndarray,
    voxel_pitch_mm: float,
    vial: VialOptics,
    u_pitch_mm: float | None = None,
    min_transmission: float = 0.05,
):
    """
    Remapea un sinograma ideal (sin vial) a las imágenes que debe emitir el proyector.

    Parameters
    ----------
    sino : (nT, nA, nZ) en la convención de CAL/VAMToolbox (detector, ángulo, capa)
    angles_deg : ángulos del sinograma, equiespaciados en [0, 360)
    voxel_pitch_mm : tamaño del vóxel (= separación de detector del sinograma ideal) en la resina
    vial : VialOptics
    u_pitch_mm : separación de columnas en el plano del proyector (por defecto = voxel_pitch_mm)
    min_transmission : por debajo de esta transmisión la columna se apaga (evita amplificar ruido)

    Returns
    -------
    out : (nU, nA, nZ) imágenes corregidas (misma escala relativa que `sino`)
    u : (nU,) posiciones de columna en mm
    info : dict con métricas de la corrección
    """
    sino = np.asarray(sino)
    if sino.ndim == 2:
        sino = sino[:, :, None]
    nT, nA, nZ = sino.shape
    angles_deg = np.asarray(angles_deg, dtype=np.float64)
    d_ang = 360.0 / nA
    if len(angles_deg) != nA or not np.allclose(np.diff(angles_deg), d_ang, atol=1e-6):
        raise ValueError("La corrección del vial necesita ángulos equiespaciados que cubran 360°")
    a0 = angles_deg[0]

    u = projector_u_grid(vial, u_pitch_mm or voxel_pitch_mm)
    t, delta, T, valid = vial.ray_map(u)

    # Columna (fraccionaria) del sinograma ideal y desplazamiento angular fraccionario
    j = t / voxel_pitch_mm + (nT - 1) / 2.0
    inside = valid & (j >= 0) & (j <= nT - 1) & (T >= min_transmission)
    shift = np.degrees(delta) / d_ang          # φ = θ - δ  →  índice k - shift

    out = np.zeros((len(u), nA, nZ), dtype=np.float32)
    lost_energy_cols = int(np.sum(valid & ~inside & (T >= min_transmission)))
    for q in np.flatnonzero(inside):
        j0 = int(np.floor(j[q])); wt = j[q] - j0; j1 = min(j0 + 1, nT - 1)
        col = (1.0 - wt) * sino[j0] + wt * sino[j1]          # (nA, nZ)
        s0 = int(np.floor(shift[q])); ws = shift[q] - s0
        col = (1.0 - ws) * np.roll(col, s0, axis=0) + ws * np.roll(col, s0 + 1, axis=0)
        out[q] = col / T[q]
    # Normalización relativa: el haz central (T(0)) conserva su escala
    T0 = vial.ray_map(np.array([0.0]))[2][0]
    out *= T0

    info = {
        "vial": vial.summary(),
        "u_pitch_mm": float(u_pitch_mm or voxel_pitch_mm),
        "n_columns": int(len(u)),
        "active_columns": int(inside.sum()),
        "max_delta_deg": float(np.degrees(np.abs(delta[inside]).max())) if inside.any() else 0.0,
        "min_transmission": float(T[inside].min()) if inside.any() else 0.0,
        "volume_radius_mm": float((nT - 1) / 2.0 * voxel_pitch_mm),
        "uncovered_columns": lost_energy_cols,
        "angle_offset_deg": float(a0),
    }
    return out, u, info


def check_fit(part_radius_mm: float, vial: VialOptics) -> None:
    """Lanza ValueError si la pieza (radio máximo respecto al eje de giro) no cabe en la zona imprimible."""
    if part_radius_mm > vial.printable_radius:
        raise ValueError(
            f"La pieza no cabe: su radio respecto al eje es {part_radius_mm:.2f} mm y la zona imprimible "
            f"de este vial es {vial.printable_radius:.2f} mm (Ø {2 * vial.printable_radius:.2f} mm). "
            "Usa un vial mayor, un líquido exterior de índice más alto o reorienta la pieza."
        )

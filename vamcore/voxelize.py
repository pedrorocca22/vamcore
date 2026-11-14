"""
Voxelizador por secciones (exacto, con ocupación parcial y en unidades físicas)
===============================================================================

Para cada capa Z se corta la malla con planos horizontales, se obtienen los segmentos de la sección y
se rellenan por líneas de barrido:

* **Exacto en X**: en cada línea de barrido se calcula la longitud exacta de cada intervalo relleno que
  cae dentro de cada vóxel (no se muestrea por puntos).
* **Ocupación parcial**: Y y Z se submuestrean (por defecto 4 sublíneas y 2 subplanos por vóxel), así
  que cada vóxel guarda la fracción de su volumen que está dentro de la pieza (0…1). Las superficies
  curvas y las paredes finas dejan de tener escalones de un vóxel.
* **Robusto**: regla de la semirrecta semiabierta en planos y líneas (un vértice o arista justo en el
  plano o en la línea nunca se cuenta dos veces). Con normales coherentes usa la regla del **número de
  giro** (no nulo), que rellena correctamente cuerpos solapados; si no, par-impar.
* **Ligero**: procesa plano a plano; la memoria es la del volumen de salida más la de una sección.

Convención de salida (idéntica a vamtoolbox.voxelize.voxelizeTarget): ``array[ix, iy, iz]`` con
``x = (ix - (n-1)/2)·pitch``, ``y = (iy - (n-1)/2)·pitch`` (eje de giro en el centro del array) y
``z`` creciente con ``iz``. La malla debe venir ya centrada en el eje de giro (como hace el pipeline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class VoxelGrid:
    pitch_mm: float
    n_xy: int
    n_z: int
    z_start_mm: float          # cota del borde inferior de la capa 0
    supersample_xy: int
    supersample_z: int
    fill_rule: str

    @property
    def radius_mm(self) -> float:
        return self.n_xy * self.pitch_mm / 2

    def x_coords(self) -> np.ndarray:
        return (np.arange(self.n_xy) - (self.n_xy - 1) / 2) * self.pitch_mm

    def z_coords(self) -> np.ndarray:
        return self.z_start_mm + (np.arange(self.n_z) + 0.5) * self.pitch_mm

    def as_dict(self) -> dict:
        d = asdict(self)
        d["radius_mm"] = self.radius_mm
        return d


def native_pitch_mm(projector_pixel_um: float, n_resin: float | None = None, n_outside: float = 1.0) -> float:
    """
    Separación entre rayos del proyector *dentro de la resina* (mm).

    Sin vial (o con baño de index matching) los rayos están separados un píxel. Con la corrección del vial
    sin baño, la refracción los junta en el plano horizontal: un rayo a distancia u recorre la resina a
    n_out·u/n_resina, así que la separación efectiva es pixel·n_out/n_resina. Vóxeles mucho más finos que
    esto no aportan detalle que el proyector pueda reproducir.
    """
    p = projector_pixel_um / 1000.0
    return p * n_outside / n_resin if n_resin else p


# ── diagnóstico ───────────────────────────────────────────────────────────────
def mesh_report(mesh) -> dict:
    """Comprobaciones baratas de la malla (trimesh.Trimesh)."""
    import trimesh

    rep = {
        "triangles": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "extents_mm": [float(v) for v in mesh.extents],
    }
    try:
        labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
        rep["bodies"] = int(labels.max() + 1) if len(labels) else 0
    except Exception:
        rep["bodies"] = None
    rep["volume_mm3"] = float(mesh.volume) if rep["watertight"] else None
    rep["warnings"] = []
    if not rep["watertight"]:
        rep["warnings"].append("La malla no es estanca (tiene huecos): la voxelización puede perder material en esas zonas.")
    if not rep["winding_consistent"]:
        rep["warnings"].append("Normales incoherentes: se usa la regla par-impar (cuerpos solapados quedarían huecos).")
    if rep["bodies"] and rep["bodies"] > 1:
        rep["warnings"].append(f"La malla tiene {rep['bodies']} cuerpos separados.")
    return rep


# ── sección de la malla en un plano ───────────────────────────────────────────
class _Slicer:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, face_normals: np.ndarray | None):
        tri = vertices[faces].astype(np.float64)                     # (M, 3, 3)
        tz = tri[:, :, 2]
        order = np.argsort(tz.min(axis=1))
        self.tri = tri[order]
        self.tzmin = tz.min(axis=1)[order]
        self.tzmax = tz.max(axis=1)[order]
        self.nxy = None if face_normals is None else face_normals[order, :2].astype(np.float64)

    def segments(self, h: float):
        """Segmentos (S, 2, 2) de la sección z = h y su normal exterior 2D (S, 2) si se conoce."""
        k = np.searchsorted(self.tzmin, h, side="left")              # tzmin < h
        sel = np.flatnonzero(self.tzmax[:k] >= h)                    # y además algún vértice ≥ h
        if len(sel) == 0:
            return np.zeros((0, 2, 2)), None
        tri = self.tri[sel]
        above = tri[:, :, 2] >= h                                    # semiabierto: ningún vértice "en" el plano
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            cross = above[:, a] != above[:, b]
            za, zb = tri[:, a, 2], tri[:, b, 2]
            t = np.where(cross, (h - za) / np.where(cross, zb - za, 1.0), 0.0)
            p = tri[:, a, :2] + t[:, None] * (tri[:, b, :2] - tri[:, a, :2])
            pts.append((cross, p))
        # cada triángulo que corta tiene exactamente dos aristas que cruzan
        c = np.stack([pc[0] for pc in pts], axis=1)                  # (S, 3)
        P = np.stack([pc[1] for pc in pts], axis=1)                  # (S, 3, 2)
        first = np.argmax(c, axis=1)
        second = 2 - np.argmax(c[:, ::-1], axis=1)
        idx = np.arange(len(sel))
        seg = np.stack([P[idx, first], P[idx, second]], axis=1)      # (S, 2, 2)
        n2 = None if self.nxy is None else self.nxy[sel]
        return seg, n2


# ── relleno de una sección por líneas de barrido ─────────────────────────────
def _fill_section(seg, n2, n, pitch, sy, rule):
    """Cobertura (n, n) en [0,1] de la sección: filas = iy, columnas = ix."""
    cov = np.zeros((n, n), dtype=np.float64)
    if len(seg) == 0:
        return cov, 0
    x1, y1 = seg[:, 0, 0], seg[:, 0, 1]
    x2, y2 = seg[:, 1, 0], seg[:, 1, 1]
    if rule == "nonzero":
        # orientar cada segmento para que su normal derecha (dy, -dx) apunte hacia fuera
        dx, dy = x2 - x1, y2 - y1
        flip = (dy * n2[:, 0] - dx * n2[:, 1]) < 0
        x1, x2 = np.where(flip, x2, x1), np.where(flip, x1, x2)
        y1, y2 = np.where(flip, y2, y1), np.where(flip, y1, y2)
    ymin, ymax = np.minimum(y1, y2), np.maximum(y1, y2)
    nr = n * sy
    dy_s = pitch / sy
    y0 = -n * pitch / 2
    r_lo = np.clip(np.ceil((ymin - y0) / dy_s - 0.5), 0, nr).astype(np.int64)
    r_hi = np.clip(np.ceil((ymax - y0) / dy_s - 0.5), 0, nr).astype(np.int64)   # exclusivo
    cnt = r_hi - r_lo
    keep = cnt > 0
    if not keep.any():
        return cov, 0
    x1, y1, x2, y2, r_lo, cnt = x1[keep], y1[keep], x2[keep], y2[keep], r_lo[keep], cnt[keep]
    seg_id = np.repeat(np.arange(len(cnt)), cnt)
    rows = r_lo[seg_id] + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    yr = y0 + (rows + 0.5) * dy_s
    xs = x1[seg_id] + (yr - y1[seg_id]) * (x2[seg_id] - x1[seg_id]) / (y2[seg_id] - y1[seg_id])
    wind = np.sign(y2[seg_id] - y1[seg_id])

    o = np.lexsort((xs, rows))
    rows, xs, wind = rows[o], xs[o], wind[o]
    # posición de cada cruce dentro de su fila
    starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
    row_start = np.repeat(starts, np.diff(np.r_[starts, len(rows)]))
    if rule == "nonzero":
        cw = np.cumsum(wind)
        cw = cw - np.r_[0, cw][row_start]                    # giro acumulado dentro de la fila
        inside_after = cw != 0
        row_end_open = cw[np.r_[starts[1:] - 1, len(rows) - 1]] != 0
    else:
        k = np.arange(len(rows)) - row_start
        inside_after = (k % 2) == 0
        row_end_open = (np.diff(np.r_[starts, len(rows)]) % 2) == 1
    open_rows = int(row_end_open.sum())

    same_row_next = np.r_[rows[1:] == rows[:-1], False]
    q = np.flatnonzero(inside_after & same_row_next)
    if len(q) == 0:
        return cov, open_rows
    iy = rows[q] // sy
    a = np.clip((xs[q] - y0) / pitch, 0, n)                  # x en unidades de vóxel (X0 = Y0 = -n·p/2)
    b = np.clip((xs[q + 1] - y0) / pitch, 0, n)
    ok = b > a
    iy, a, b = iy[ok], a[ok], b[ok]
    cs = np.minimum(np.floor(a).astype(np.int64), n - 1)
    ce = np.minimum(np.floor(b).astype(np.int64), n - 1)
    one = cs == ce
    np.add.at(cov, (iy[one], cs[one]), b[one] - a[one])
    m = ~one
    np.add.at(cov, (iy[m], cs[m]), cs[m] + 1 - a[m])
    np.add.at(cov, (iy[m], ce[m]), b[m] - ce[m])
    D = np.zeros((n, n + 1))
    np.add.at(D, (iy[m], cs[m] + 1), 1.0)
    np.add.at(D, (iy[m], ce[m]), -1.0)
    cov += np.cumsum(D, axis=1)[:, :n]
    return cov / sy, open_rows


# ── API principal ─────────────────────────────────────────────────────────────
def voxelize_mesh(
    mesh,
    pitch_mm: float,
    radius_mm: float | None = None,
    supersample_xy: int = 4,
    supersample_z: int = 2,
    pad_voxels: int = 2,
    fill_rule: str = "auto",
    dtype=np.float32,
):
    """
    Voxeliza una malla (trimesh.Trimesh, centrada en el eje de giro, Z = eje del vial).

    Returns
    -------
    occupancy : ndarray (n_xy, n_xy, n_z) en [0, 1], indexado [ix, iy, iz]
    grid : VoxelGrid
    stats : dict (volumen voxelizado, filas abiertas, regla usada…)
    """
    if pitch_mm <= 0:
        raise ValueError("pitch_mm debe ser > 0")
    sy, sz = max(1, int(supersample_xy)), max(1, int(supersample_z))
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces)
    if rule_needs_normals := (fill_rule in ("auto", "nonzero")):
        consistent = bool(mesh.is_winding_consistent)
        rule = "nonzero" if (fill_rule == "nonzero" or consistent) else "evenodd"
    else:
        rule = "evenodd"
    normals = np.asarray(mesh.face_normals) if rule == "nonzero" else None
    if rule == "nonzero" and mesh.is_watertight and mesh.volume < 0:
        normals = -normals                                   # malla con normales hacia dentro

    r_part = float(np.max(np.hypot(V[:, 0], V[:, 1])))
    R = max(radius_mm or 0.0, r_part) + pad_voxels * pitch_mm
    n = 2 * int(math.ceil(R / pitch_mm)) + 1                 # impar: el eje cae en el centro de un vóxel
    zmin, zmax = float(V[:, 2].min()), float(V[:, 2].max())
    nz = int(math.ceil((zmax - zmin) / pitch_mm - 1e-9)) + 2 * pad_voxels
    z_start = (zmin + zmax) / 2 - nz * pitch_mm / 2
    grid = VoxelGrid(pitch_mm, n, nz, z_start, sy, sz, rule)

    slicer = _Slicer(V, F, normals)
    out = np.zeros((n, n, nz), dtype=dtype)
    open_rows = 0
    for k in range(nz):
        acc = np.zeros((n, n))
        for j in range(sz):
            h = z_start + (k + (j + 0.5) / sz) * pitch_mm
            if h < zmin or h > zmax:
                continue
            seg, n2 = slicer.segments(h)
            cov, op = _fill_section(seg, n2, n, pitch_mm, sy, rule)
            acc += cov
            open_rows += op
        out[:, :, k] = (acc / sz).T                          # filas iy, columnas ix → [ix, iy]
    np.clip(out, 0.0, 1.0, out=out)

    vox_volume = float(out.sum(dtype=np.float64) * pitch_mm**3)
    stats = {
        "grid": grid.as_dict(),
        "fill_rule": rule,
        "voxel_volume_mm3": vox_volume,
        "open_scanlines": int(open_rows),
        "partial_voxels": int(np.count_nonzero((out > 0.02) & (out < 0.98))),
        "solid_voxels": int(np.count_nonzero(out >= 0.5)),
    }
    if mesh.is_watertight and abs(mesh.volume) > 0:
        stats["volume_error_pct"] = 100.0 * (vox_volume - abs(mesh.volume)) / abs(mesh.volume)
    return out, grid, stats

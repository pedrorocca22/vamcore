"""Pruebas del voxelizador por secciones (vamcore/voxelize.py). Ligeras: mallas pequeñas, < 10 s.

Ejecutar desde la raíz:  python -m pytest tests/test_slicevox.py -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

ROOT = Path(__file__).resolve().parents[1]
from vamcore import voxelize as sv  # noqa: E402


def centered(m):
    m = m.copy()
    c = m.bounds.mean(axis=0)
    m.apply_translation([-c[0], -c[1], 0.0])
    return m


def box(x0, x1, y0, y1, z0, z1):
    b = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return b


# ── volumen frente a referencias analíticas ─────────────────────────────────
@pytest.mark.parametrize("pitch", [0.25, 0.1])
def test_sphere_volume_and_range(pitch):
    m = trimesh.creation.icosphere(subdivisions=4, radius=3.0)
    occ, grid, st = sv.voxelize_mesh(m, pitch)
    assert occ.dtype == np.float32 and occ.min() >= 0 and occ.max() <= 1
    assert grid.n_xy % 2 == 1                                  # eje en el centro de un vóxel
    assert abs(st["volume_error_pct"]) < (1.0 if pitch > 0.2 else 0.3)
    c = grid.n_xy // 2
    assert occ[c, c, grid.n_z // 2] == pytest.approx(1.0)
    assert st["fill_rule"] == "nonzero" and st["open_scanlines"] == 0


def test_thin_tube_wall_keeps_its_volume():
    # pared de 0.15 mm con vóxeles de 0.1 mm: un muestreo binario por centros la pierde o la duplica
    tube = trimesh.creation.annulus(r_min=2.0, r_max=2.15, height=3.0, sections=256)
    occ, grid, st = sv.voxelize_mesh(tube, 0.1)
    assert abs(st["volume_error_pct"]) < 2.0
    assert st["partial_voxels"] > st["solid_voxels"]           # casi toda la pared es ocupación parcial


def test_overlapping_bodies_are_united_with_nonzero_rule():
    a = box(-2, 1, -1, 1, 0, 2)
    b = box(-1, 2, -1, 1, 0, 2)                                # solapan en x ∈ [-1, 1]
    m = trimesh.util.concatenate([a, b])                       # sin unión booleana, como muchos STL de CAD
    union_volume = 4 * 2 * 2
    occ, grid, st = sv.voxelize_mesh(m, 0.1)
    assert st["fill_rule"] == "nonzero"
    vol = occ.sum() * 0.1**3
    assert vol == pytest.approx(union_volume, rel=0.01)
    occ_eo, _, _ = sv.voxelize_mesh(m, 0.1, fill_rule="evenodd")
    assert occ_eo.sum() * 0.1**3 == pytest.approx(union_volume - 2 * 2 * 2, rel=0.02)  # par-impar vacía el solape


def test_open_mesh_does_not_crash_and_is_reported():
    m = trimesh.creation.icosphere(subdivisions=3, radius=2.0)
    m.update_faces(np.arange(len(m.faces)) != 7)                # quitar un triángulo → malla abierta
    rep = sv.mesh_report(m)
    assert not rep["watertight"] and rep["warnings"]
    occ, grid, st = sv.voxelize_mesh(m, 0.1)
    assert np.isfinite(occ).all() and occ.sum() > 0


def test_vertices_exactly_on_planes_and_scanlines():
    # caja alineada con la rejilla: caras, aristas y vértices caen exactamente en planos y líneas de muestreo
    m = box(-1.0, 1.0, -1.0, 1.0, 0.0, 2.0)
    occ, grid, st = sv.voxelize_mesh(m, 0.25, supersample_xy=4, supersample_z=2)
    assert st["volume_error_pct"] == pytest.approx(0.0, abs=1e-6)
    assert st["open_scanlines"] == 0


# ── orientación: [ix, iy, iz] con x en el eje 0 (sin reflejo) ────────────────
def L_shape():
    # pieza quiral: brazo largo hacia +x, brazo corto hacia +y, más alta por un lado
    return trimesh.util.concatenate([box(0, 4, 0, 1, 0, 3), box(0, 1, 1, 2.5, 0, 1.5)])


def phys_centroid(occ, grid):
    x = grid.x_coords()
    z = grid.z_coords()
    w = occ.sum()
    return (np.einsum("ijk,i->", occ, x) / w, np.einsum("ijk,j->", occ, x) / w, np.einsum("ijk,k->", occ, z) / w)


def test_axes_follow_physical_coordinates():
    m = L_shape()
    occ, grid, st = sv.voxelize_mesh(m, 0.1)
    cx, cy, cz = phys_centroid(occ, grid)
    ref = m.center_mass
    assert cx == pytest.approx(ref[0], abs=0.02)
    assert cy == pytest.approx(ref[1], abs=0.02)
    assert cz == pytest.approx(ref[2], abs=0.02)


# ── escala física ─────────────────────────────────────────────────────────────
def test_native_pitch():
    assert sv.native_pitch_mm(14.3) == pytest.approx(0.0143)
    assert sv.native_pitch_mm(14.3, n_resin=1.35) == pytest.approx(0.0143 / 1.35)

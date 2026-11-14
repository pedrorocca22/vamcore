"""vamcore.orient: auto-orientación geométrica con restricciones de altura y radio. Pura (numpy + trimesh)."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
from vamcore.orient import extents_after, find_orientation, rotation  # noqa: E402


def tilted_cylinder(deg, axis=(1, 0, 0), height=12.0, radius=1.0):
    c = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
    c.apply_transform(trimesh.transformations.rotation_matrix(np.radians(deg), axis))
    return np.asarray(c.vertices)


def axis_angle_to_z(v_rot):
    # eje principal de la nube de puntos (cilindro largo) frente a Z
    w = v_rot - v_rot.mean(axis=0)
    _, _, vt = np.linalg.svd(w, full_matrices=False)
    return np.degrees(np.arccos(np.clip(abs(vt[0][2]), 0.0, 1.0)))  # el redondeo puede dar 1 + ε → NaN


@pytest.mark.parametrize("deg,axis", [(35, (1, 0, 0)), (-50, (0, 1, 0)), (25, (1, 1, 0))])
def test_recovers_vertical_axis(deg, axis):
    v = tilted_cylinder(deg, axis)
    r = find_orientation(v)
    assert r.feasible and r.reduction_pct > 30
    assert axis_angle_to_z(v @ np.array(r.rotation).T) < 0.5


def test_already_optimal_is_left_alone():
    r = find_orientation(tilted_cylinder(0))
    assert r.rx == 0 and r.ry == 0 and r.reduction_pct == 0


def test_height_limit_is_respected():
    v = tilted_cylinder(60, height=30.0)        # de pie mediría 30 mm > 25 mm
    r = find_orientation(v, max_height_mm=25.0)
    assert r.feasible and r.height_mm <= 25.0 + 1e-6
    free = find_orientation(v)
    assert free.height_mm > 25.0                 # sin límite elegiría ponerlo vertical


def test_radius_limit_and_infeasible_case():
    v = tilted_cylinder(0, height=12.0, radius=3.0)
    ok = find_orientation(v, max_radius_mm=5.0)
    assert ok.feasible and ok.radius_mm <= 5.0
    bad = find_orientation(v, max_radius_mm=1.0)  # no cabe de ninguna forma
    assert not bad.feasible


def test_rotation_convention_matches_pipeline():
    # el pipeline aplica R = Rz(rz)·Ry(ry)·Rx(rx); con rz = 0 debe ser exactamente rotation(rx, ry)
    a, b = 23.0, -41.0
    rx, ry = np.radians(a), np.radians(b)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    np.testing.assert_allclose(rotation(a, b), Ry @ Rx, atol=1e-12)
    h, r = extents_after(np.array([[0, 0, -1.0], [0, 0, 1.0], [2, 0, 0]]), np.eye(3))
    assert h == 2.0 and r == pytest.approx(1.0)  # caja x ∈ [0, 2] → centro x = 1 → radio 1

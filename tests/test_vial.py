"""Pruebas de vamcore/optics/vial.py (ligeras: numpy puro, < 1 s).

Ejecutar desde la raíz:  python -m pytest tests/test_vial.py -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# vamcore no importa ASTRA ni torch: se puede importar directamente (tests/conftest.py añade la raíz).
from vamcore.optics import vial as vial_mod  # noqa: E402

VialOptics = vial_mod.VialOptics

TYPICAL = VialOptics(outer_diameter_mm=20.0, wall_thickness_mm=1.0, n_vial=1.483, n_resin=1.35)


# ── trazador independiente (Snell vectorial) ──────────────────────────────────
def _hit_circle(p, d, R):
    b = p @ d
    c = p @ p - R * R
    disc = b * b - c
    if disc < 0:
        return None
    s = -b - np.sqrt(disc)          # primera intersección (desde fuera)
    return p + s * d if s > 0 else None


def _refract(d, normal, n1, n2):
    eta = n1 / n2
    cosi = -normal @ d
    k = 1 - eta**2 * (1 - cosi**2)
    if k < 0:
        return None                 # reflexión total
    return eta * d + (eta * cosi - np.sqrt(k)) * normal


def trace(u, theta, v: VialOptics):
    """Rayo del proyector a ángulo theta (rad) y desplazamiento u → (phi, t) de la recta en la resina."""
    d = np.array([np.sin(theta), -np.cos(theta)])          # dirección ASTRA parallel3d
    e = np.array([np.cos(theta), np.sin(theta)])           # eje del detector
    p = u * e - 100.0 * d
    if v.wall_thickness_mm > 0:
        surfaces = ((v.r_outer, v.n_outside, v.n_vial), (v.r_inner, v.n_vial, v.n_resin))
    else:
        surfaces = ((v.r_outer, v.n_outside, v.n_resin),)
    for R, n1, n2 in surfaces:
        hit = _hit_circle(p, d, R)
        if hit is None:
            return None
        d = _refract(d, hit / R, n1, n2)
        if d is None:
            return None
        p = hit
    phi = np.arctan2(d[0], -d[1])
    t = p @ np.array([np.cos(phi), np.sin(phi)])
    return phi, t


# ── modelo analítico ──────────────────────────────────────────────────────────
def test_center_ray_is_undeviated():
    t, delta, T, valid = TYPICAL.ray_map(np.array([0.0]))
    assert valid[0] and t[0] == 0 and delta[0] == 0 and 0.9 < T[0] < 1.0


def test_index_matched_is_identity():
    v = VialOptics(20.0, 1.0, n_vial=1.4, n_resin=1.4, n_outside=1.4)
    u = np.linspace(-8.9, 8.9, 41)                       # el hueco interior tiene radio 9 mm
    t, delta, T, valid = v.ray_map(u)
    assert valid.all()
    np.testing.assert_allclose(t, u, atol=1e-12)
    np.testing.assert_allclose(delta, 0, atol=1e-12)
    np.testing.assert_allclose(T, 1, atol=1e-12)


def test_odd_symmetry():
    u = np.linspace(0.1, 9.0, 30)
    tp, dp, Tp, _ = TYPICAL.ray_map(u)
    tn, dn, Tn, _ = TYPICAL.ray_map(-u)
    np.testing.assert_allclose(tn, -tp)
    np.testing.assert_allclose(dn, -dp)
    np.testing.assert_allclose(Tn, Tp)


@pytest.mark.parametrize("vial", [
    TYPICAL,
    VialOptics(16.0, 2.0, n_vial=1.62, n_resin=1.48),        # poliestireno + acrilato
    VialOptics(25.0, 0.0, n_vial=1.5, n_resin=1.343),        # sin pared (película fina)
    VialOptics(20.0, 1.5, n_vial=1.47, n_resin=1.36, n_outside=1.333),  # baño parcial de agua
])
def test_analytic_matches_vector_snell_tracer(vial):
    u = np.linspace(-0.98, 0.98, 49) * vial.u_max
    t, delta, T, valid = vial.ray_map(u)
    for ui, ti, di, ok in zip(u, t, delta, valid):
        assert ok
        res = trace(ui, 0.0, vial)
        assert res is not None
        phi, tt = res
        assert phi == pytest.approx(-di, abs=1e-9)        # φ = θ - δ con θ = 0
        assert tt == pytest.approx(ti, abs=1e-9)


def test_rays_beyond_u_max_do_not_reach_resin():
    v = TYPICAL
    _, _, _, valid = v.ray_map(np.array([v.u_max * 1.001, v.r_outer * 0.999]))
    assert not valid[0]
    # el trazador independiente coincide en que no llega (o llega solo si u_max == r_outer)
    if v.u_max < v.r_outer * 0.999:
        assert trace(v.r_outer * 0.999, 0.0, v) is None


def test_printable_radius_typical_vial():
    # Vial 20 mm, pared 1 mm, borosilicato, GelMA: la luz solo alcanza ~Ø14.8 mm de los 18 mm interiores
    assert TYPICAL.printable_radius == pytest.approx(10.0 / 1.35, rel=1e-9)
    with pytest.raises(ValueError, match="no cabe"):
        vial_mod.check_fit(8.0, TYPICAL)
    vial_mod.check_fit(7.0, TYPICAL)


# ── remapeo del sinograma ─────────────────────────────────────────────────────
BLOBS = [((1.5, -0.8), 0.5, 1.0), ((-2.5, 1.2), 0.35, 0.7), ((0.4, 3.1), 0.6, 0.5)]  # (centro, sigma, amplitud)


def radon_blobs(phi, t):
    """Sinograma analítico de manchas gaussianas en la convención de ASTRA (línea p·e(φ) = t)."""
    out = np.zeros(np.broadcast(phi, t).shape)
    for (cx, cy), s, A in BLOBS:
        tc = cx * np.cos(phi) + cy * np.sin(phi)
        out += A * np.sqrt(2 * np.pi) * s * np.exp(-((t - tc) ** 2) / (2 * s * s))
    return out


def test_correct_sinogram_samples_the_physical_line():
    v = TYPICAL
    pitch = 0.02
    nA = 720
    ang = np.arange(nA) * 360.0 / nA
    nT = 2 * int(round(9.0 / pitch)) + 1
    tt = (np.arange(nT) - (nT - 1) / 2) * pitch
    sino = radon_blobs(np.radians(ang)[None, :], tt[:, None])[:, :, None]

    out, u, info = vial_mod.correct_sinogram(sino, ang, pitch, v)
    T0 = v.ray_map(np.array([0.0]))[2][0]
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(300):
        q = rng.integers(len(u)); k = rng.integers(nA)
        res = trace(u[q], np.radians(ang[k]), v)
        if res is None or out[q, k, 0] == 0:
            continue
        phi, t = res
        T = v.ray_map(np.array([u[q]]))[2][0]
        expected = radon_blobs(np.array(phi), np.array(t)) / T * T0
        errs.append(abs(out[q, k, 0] - expected))
    assert len(errs) > 150
    assert max(errs) / sino.max() < 0.01          # < 1 % del máximo (error de interpolación)
    assert info["active_columns"] > 0 and info["max_delta_deg"] > 1


def test_correct_sinogram_identity_when_index_matched():
    v = VialOptics(20.0, 1.0, n_vial=1.45, n_resin=1.45, n_outside=1.45)
    pitch = 0.05
    nA = 180
    ang = np.arange(nA) * 2.0
    nT = 2 * int(round(10.0 / pitch)) + 1
    tt = (np.arange(nT) - (nT - 1) / 2) * pitch
    sino = radon_blobs(np.radians(ang)[None, :], tt[:, None])[:, :, None]
    out, u, _ = vial_mod.correct_sinogram(sino, ang, pitch, v)
    np.testing.assert_allclose(u, tt, atol=1e-9)
    np.testing.assert_allclose(out[1:-1], sino[1:-1], rtol=1e-5, atol=1e-6)


def test_rejects_non_uniform_angles():
    with pytest.raises(ValueError):
        vial_mod.correct_sinogram(np.zeros((5, 3, 1)), np.array([0, 10, 30]), 0.1, TYPICAL)

"""Refracción del vial dentro del modelo (docs/MIGRACION.md F6c): operador G (imagen del proyector → rayos en la resina)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from vamcore.optics.vial import VialOptics, correct_sinogram  # noqa: E402
from vamcore.optics.vial_operator import VialProjector, VialResampler  # noqa: E402
from vamcore.projector import make_projector  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TYPICAL = VialOptics(outer_diameter_mm=20.0, wall_thickness_mm=1.0, n_vial=1.483, n_resin=1.35)
MATCHED = VialOptics(outer_diameter_mm=20.0, wall_thickness_mm=1.0, n_vial=1.0, n_resin=1.0)


def smooth_sino(n_t, ang, pitch, n_z=2):
    t = (np.arange(n_t) - (n_t - 1) / 2) * pitch
    phi = np.radians(ang)
    b = np.zeros((n_t, len(ang), n_z), np.float32)
    for z in range(n_z):  # proyecciones de manchas gaussianas (suaves en φ y t)
        for (cx, cy, s) in ((1.5, -1.0, 1.2), (-2.0, 0.5, 0.9), (0.3, 2.2, 0.7)):
            tc = cx * np.cos(phi)[None] + cy * np.sin(phi)[None]
            b[:, :, z] += np.exp(-((t[:, None] - tc) ** 2) / (2 * s ** 2)) * (1 + 0.3 * z)
    return b


def test_g_undoes_correct_sinogram():
    ang = np.arange(0, 360, 1.0); pitch = 0.1; n_t = 141  # resina de ±7 mm, dentro de la zona imprimible
    b = smooth_sino(n_t, ang, pitch)
    p, _, _ = correct_sinogram(b, ang, pitch, TYPICAL)
    R = VialResampler(TYPICAL, ang, pitch, n_t, device=DEV)
    assert R.n_u == p.shape[0]
    back = R.apply(torch.tensor(p, device=DEV)).cpu().numpy()  # C multiplica por T0/T y G por T/T0: se cancelan
    inner = np.abs((np.arange(n_t) - (n_t - 1) / 2) * pitch) < 6.5
    err = np.abs(back[inner] - b[inner]).max() / b.max()
    assert err < 0.02, err


def test_index_matched_is_identity():
    ang = np.arange(0, 360, 2.0); pitch = 0.1; n_t = 101  # n_t impar: la rejilla t coincide con columnas u
    R = VialResampler(MATCHED, ang, pitch, n_t, device=DEV)
    p = torch.rand((R.n_u, len(ang), 3), device=DEV)
    b = R.apply(p).cpu().numpy()
    off = (R.n_u - n_t) // 2
    np.testing.assert_allclose(b, p[off:off + n_t].cpu().numpy(), rtol=1e-6, atol=1e-6)


def test_rays_match_vector_snell_tracer():
    import importlib.util, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("tv", Path(__file__).with_name("test_vial.py"))
    tv = importlib.util.module_from_spec(spec); sys.modules["tv"] = tv; spec.loader.exec_module(tv)
    ang = np.arange(0, 360, 1.0); pitch = 0.1; n_t = 141
    R = VialResampler(TYPICAL, ang, pitch, n_t, device=DEV)
    # imagen del proyector suave conocida f(θ, u): G·f en (φ_k, t_j) = f(θ*, u*)·T/T0 con (θ*, u*) el píxel cuyo rayo
    # (trazado con Snell vectorial) es exactamente (φ_k, t_j)
    th = np.radians(ang)[None, :, None]; u = R.u[:, None, None]
    f = (np.cos(th) * np.exp(-(u - 2) ** 2 / 8) + 1.2).astype(np.float32)
    b = R.apply(torch.tensor(np.broadcast_to(f, (R.n_u, len(ang), 1)).copy(), device=DEV)).cpu().numpy()[:, :, 0]
    T0 = TYPICAL.ray_map(np.array([0.0]))[2][0]
    rng = np.random.default_rng(0)
    for _ in range(40):
        k = int(rng.integers(len(ang))); j = int(rng.integers(15, n_t - 15))
        t = (j - (n_t - 1) / 2) * pitch
        u_star = TYPICAL.n_resin * t / TYPICAL.n_outside
        _, delta, T, _ = TYPICAL.ray_map(np.array([u_star]))
        th_star = np.radians(ang[k]) + delta[0]
        phi_tr, t_tr = tv.trace(u_star, th_star, TYPICAL)
        assert np.isclose(np.angle(np.exp(1j * (phi_tr - np.radians(ang[k])))), 0, atol=1e-9)
        assert t_tr == pytest.approx(t, abs=1e-9)
        expected = (np.cos(th_star) * np.exp(-(u_star - 2) ** 2 / 8) + 1.2) * T[0] / T0
        assert b[j, k] == pytest.approx(expected, rel=0.01)


def test_adjoint_exact():
    ang = np.arange(0, 360, 3.0)
    R = VialResampler(TYPICAL, ang, 0.15, 81, device=DEV)
    g = torch.Generator().manual_seed(1)
    p = torch.rand((R.n_u, len(ang), 2), generator=g).to(DEV); b = torch.rand((81, len(ang), 2), generator=g).to(DEV)
    lhs = float((R.apply(p).double() * b.double()).sum()); rhs = float((p.double() * R.adjoint(b).double()).sum())
    assert abs(lhs - rhs) / abs(lhs) < 1e-5


def test_vial_projector_matches_base_when_index_matched():
    ang = np.arange(0, 360, 2.0); n = 61
    base = make_projector((n, n, 2), ang, backend="torch", device=DEV)
    VP = VialProjector(base, VialResampler(MATCHED, ang, 0.2, n, device=DEV))
    g = torch.Generator().manual_seed(2)
    x = torch.rand((n, n, 2), generator=g).to(DEV)
    p = torch.zeros(VP.sino_shape, device=DEV); off = (VP.sino_shape[0] - n) // 2
    p[off:off + n] = torch.rand((n, len(ang), 2), generator=g).to(DEV)
    assert torch.allclose(VP.adjoint(p), base.adjoint(p[off:off + n]), rtol=1e-5, atol=1e-4)
    assert torch.allclose(VP.forward(x)[off:off + n], base.forward(x), rtol=1e-5, atol=1e-4)

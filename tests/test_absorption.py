"""Absorción de la resina (Beer–Lambert) en el proyector propio (docs/MIGRACION.md F6b). torch en CPU o GPU."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from vamcore.projector import make_projector  # noqa: E402
from vamcore.projector.torch_parallel import path_in_resin  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def disc(n, R):
    c = (n - 1) / 2
    i, j = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing="ij")
    return torch.tensor(((i ** 2 + j ** 2) <= R ** 2).astype(np.float32)[:, :, None], device=DEV)


def test_path_length_geometry():
    R = 10.0
    t = lambda v: torch.tensor(float(v), dtype=torch.float64)
    # φ = 0: rayo r = (0, −1), la luz entra por y = +R
    assert float(path_in_resin(t(0), t(0), t(0), t(1), R)) == pytest.approx(R)
    assert float(path_in_resin(t(0), t(5), t(0), t(1), R)) == pytest.approx(5.0)    # cerca de la entrada
    assert float(path_in_resin(t(0), t(-5), t(0), t(1), R)) == pytest.approx(15.0)  # lado opuesto
    assert float(path_in_resin(t(12), t(0), t(0), t(1), R)) == 0.0                  # fuera de la resina


def test_forward_matches_analytic_homogeneous_cylinder():
    # resina = disco de radio R lleno: b(u) = ∫_{-h}^{h} e^{−α(s+h)} ds = (1 − e^{−2αh}) / α,  h = √(R² − u²)
    n, R, a = 129, 40.0, 0.03
    P = make_projector((n, n, 1), [0.0, 37.0, 90.0, 233.0], backend="torch", device=DEV, alpha=a, r_resin=R)
    b = P.forward(disc(n, R))[:, :, 0].cpu().numpy()
    u = np.arange(n) - (n - 1) / 2
    h = np.sqrt(np.clip(R ** 2 - u ** 2, 0, None))
    ref = (1 - np.exp(-2 * a * h)) / a
    inner = np.abs(u) < R - 3
    for k in range(4):
        np.testing.assert_allclose(b[inner, k], ref[inner], rtol=0.03, atol=0.3)


def test_dose_matches_direct_sum():
    # sinograma uniforme: dosis(p) = Σ_φ e^{−α L(p, φ)}
    n, R, a = 65, 30.0, 0.05
    ang = np.linspace(0, 358, 180)
    P = make_projector((n, n, 1), ang, backend="torch", device=DEV, alpha=a, r_resin=R)
    x = P.adjoint(torch.ones(P.sino_shape, device=DEV))[:, :, 0].cpu().numpy()
    c = (n - 1) / 2
    phi = np.radians(ang)
    for ix, iy in [(32, 32), (32, 50), (14, 20), (45, 40)]:
        X, Y = ix - c, iy - c
        pr = X * np.sin(phi) - Y * np.cos(phi)
        L = pr + np.sqrt(R ** 2 - (X * X + Y * Y) + pr ** 2)
        assert x[ix, iy] == pytest.approx(np.exp(-a * L).sum(), rel=1e-4)
    assert x[32, 32] == pytest.approx(len(ang) * np.exp(-a * R), rel=1e-4)


def test_light_direction_single_angle():
    # φ = 0: la luz viaja hacia −y y entra por +y → el lado +y recibe e^{αR} veces más dosis que el simétrico
    n, R, a = 65, 30.0, 0.05
    P = make_projector((n, n, 1), [0.0], backend="torch", device=DEV, alpha=a, r_resin=R)
    x = P.adjoint(torch.ones(P.sino_shape, device=DEV))[:, :, 0].cpu().numpy()
    c = 32
    assert x[c, c + 15] / x[c, c - 15] == pytest.approx(np.exp(a * 30), rel=1e-4)


def test_adjoint_exact_with_absorption_and_zero_alpha_is_unchanged():
    shape, ang = (40, 40, 3), np.linspace(0, 354, 60)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(shape, generator=g).to(DEV); y = torch.rand((40, 60, 3), generator=g).to(DEV)
    P = make_projector(shape, ang, backend="torch", device=DEV, backprojector="exact", alpha=0.04, r_resin=25.0)
    lhs = float((P.forward(x).double() * y.double()).sum())
    rhs = float((x.double() * P.adjoint(y).double()).sum())
    assert abs(lhs - rhs) / abs(lhs) < 1e-5
    P0 = make_projector(shape, ang, backend="torch", device=DEV)
    Pz = make_projector(shape, ang, backend="torch", device=DEV, alpha=0.0, r_resin=25.0)
    # mismo operador; el autoajuste puede elegir otro bloque z y cambiar el orden de suma de cuSPARSE
    assert torch.allclose(P0.forward(x), Pz.forward(x), rtol=1e-6, atol=1e-6)
    assert torch.allclose(P0.adjoint(y), Pz.adjoint(y), rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError):
        make_projector(shape, ang, backend="torch", device=DEV, alpha=0.1)  # falta r_resin

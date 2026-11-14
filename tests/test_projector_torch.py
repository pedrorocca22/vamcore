"""Prisma, el proyector propio (docs/MIGRACION.md F6a): convención de ejes (fuente puntual), integral de línea de un disco,
traspuesta exacta (backprojector="exact"), retroproyección por vóxel casi adjunta, sin efectos colaterales. Su
equivalencia con ASTRA (6e-5) se midió en F6 antes de retirarlo. Corre con torch en CPU o GPU.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from vamcore.projector import Projector, make_projector  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _rand(shape, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(shape, generator=g, dtype=torch.float32).to(DEV)


@pytest.mark.parametrize("backend", ["torch", "torch_grid"])
@pytest.mark.parametrize("shape,n_angles", [((33, 33, 5), 61), ((48, 48, 7), 90), ((40, 40, 3), 360)])
def test_exact_adjoint(shape, n_angles, backend):
    kw = {"backprojector": "exact"} if backend == "torch" else {}
    P = make_projector(shape, np.linspace(0, 360 - 360 / n_angles, n_angles), backend=backend, device=DEV, **kw)
    assert isinstance(P, Projector)
    x, y = _rand(shape, 1), _rand(P.sino_shape, 2)
    lhs = float((P.forward(x).double() * y.double()).sum())
    rhs = float((x.double() * P.adjoint(y).double()).sum())
    assert abs(lhs - rhs) / abs(lhs) < 1e-5  # exacto salvo redondeo float32 (ASTRA: hasta 9e-5)


def test_point_source_lands_where_geometry_says():
    # rayo (sin φ, −cos φ), detector (cos φ, sin φ): un punto en (X, Y) cae en u = X·cos φ + Y·sin φ
    n = 65; c = (n - 1) / 2
    angles = np.array([0.0, 30.0, 90.0, 135.0, 200.0, 300.0])
    x = torch.zeros(n, n, 1, device=DEV); x[40, 22, 0] = 1.0
    P = make_projector((n, n, 1), angles, backend="torch", device=DEV)
    b = P.forward(x)[:, :, 0].cpu().numpy()
    cen = (b * np.arange(n)[:, None]).sum(0) / b.sum(0) - c
    X, Y = 40 - c, 22 - c
    exp = X * np.cos(np.radians(angles)) + Y * np.sin(np.radians(angles))
    # Joseph interpola linealmente entre vóxeles vecinos: el centro de masas de un vóxel puntual se desvía hasta
    # ~0,3 vóxeles de la posición geométrica (es propio del método; ASTRA daba exactamente los mismos valores).
    np.testing.assert_allclose(cen, exp, atol=0.3)


def test_line_integral_of_uniform_disc():
    # disco de radio R: integral a lo largo de la cuerda que pasa a distancia u del centro = 2·sqrt(R² − u²)
    n = 129; c = (n - 1) / 2; R = 40.0
    ii, jj = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing="ij")
    disc = torch.tensor(((ii ** 2 + jj ** 2) <= R ** 2).astype(np.float32)[:, :, None], device=DEV)
    P = make_projector((n, n, 1), [0.0, 37.0, 90.0], backend="torch", device=DEV)
    b = P.forward(disc)[:, :, 0].cpu().numpy()
    u = np.arange(n) - c
    ref = 2 * np.sqrt(np.clip(R ** 2 - u ** 2, 0, None))
    inner = np.abs(u) < R - 3
    for k in range(3):
        assert np.abs(b[inner, k] - ref[inner]).max() < 1.5  # error de discretización del borde: ≤ 1,5 vóxeles


def test_no_side_effects_and_shapes():
    P = make_projector((24, 24, 4), np.arange(0, 360, 10.0), backend="torch", device=DEV)
    x = _rand((24, 24, 4), 3); xc = x.clone()
    b = P.forward(x)
    assert torch.equal(x, xc) and tuple(b.shape) == P.sino_shape
    bc = b.clone(); P.adjoint(b)
    assert torch.equal(b, bc)
    with pytest.raises(ValueError):
        P.forward(torch.zeros(8, 8, 8, device=DEV))


def test_voxel_backprojector_is_nearly_adjoint():
    # no es la traspuesta exacta, pero ⟨Ax, y⟩ ≈ ⟨x, Bᵀy⟩
    P = make_projector((48, 48, 4), np.linspace(0, 358, 180), backend="torch", device=DEV)
    x, y = _rand((48, 48, 4), 7) * P._mask, _rand(P.sino_shape, 8)
    lhs = float((P.forward(x).double() * y.double()).sum())
    rhs = float((x.double() * P.adjoint(y).double()).sum())
    assert abs(lhs - rhs) / abs(lhs) < 1e-3

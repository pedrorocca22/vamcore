"""vamcore: módulos puros (filtros, métricas, máscaras, d_h, configuración). La paridad con VAMToolbox se verificó en
la fase F1 (docs/MIGRACION.md) antes de retirarlo; aquí quedan las pruebas propias, con valores calculados a mano."""
import numpy as np
import pytest

from vamcore import config, dose, filters, metrics, ops

RNG = np.random.default_rng(1234)


def _target(shape=(24, 24, 10), p=0.35, frac=False):
    t = (RNG.random(shape) < p).astype(np.float32)
    if frac:  # ocupación parcial como la de vamcore.voxelize
        t = np.clip(t * RNG.uniform(0.2, 1.0, shape), 0, 1).astype(np.float32)
    return t


# ── propias de vamcore (corren en el CI) ──────────────────────────────────────

def test_filters_names_and_aliases():
    assert filters.normalize_name("HANN") == "hanning"
    assert filters.normalize_name(None) is None and filters.normalize_name("none") is None
    with pytest.raises(ValueError):
        filters.normalize_name("bogus")


def test_ramp_filter_kills_dc_and_is_linear():
    s = RNG.random((40, 7, 3))
    a = filters.filter_sinogram(s, "ram-lak")
    b = filters.filter_sinogram(2 * s, "ram-lak")
    np.testing.assert_allclose(b, 2 * a, rtol=1e-12, atol=1e-12)
    # filtrar cada capa por separado da lo mismo que filtrar el volumen entero
    np.testing.assert_allclose(filters.filter_sinogram(s[:, :, 1], "ram-lak"), a[:, :, 1], atol=1e-12)
    h = filters.fourier_filter(128, "ram-lak")
    assert abs(h[0]) < 0.6 and h[64] == pytest.approx(h.max())  # rampa: mínima en DC, máxima en Nyquist


def test_metrics_by_hand():
    recon = np.zeros((5, 5, 1))
    target = np.zeros((5, 5, 1)); target[1:4, 1:4] = 1
    recon[1:4, 1:4] = 0.9; recon[2, 2] = 1.0           # gel: mín 0,9, máx 1,0
    recon[0, 2] = 0.95; recon[4, 2] = 0.5               # dos vóxeles de vacío dentro del círculo
    gel, void = ops.get_inds(target)
    n = gel.sum() + void.sum()
    assert metrics.ver(recon, gel, void) == pytest.approx(1 / n)
    assert metrics.pw(recon, gel, void) == pytest.approx(0.95 - 0.9)
    assert metrics.ipdr(recon, gel, void) == pytest.approx(0.1)
    assert metrics.cv(recon, gel) == pytest.approx(np.std(recon[gel]) / np.mean(recon[gel]))


def test_layerwise_metrics_are_per_layer():
    t = _target((20, 20, 6)); r = RNG.random(t.shape)
    gel, void = ops.get_inds(t)
    per = metrics.ver(r, gel, void, layerwise=True)
    assert per.shape == (6,)
    # cada capa usa su propio mínimo de gel, así que la suma no supera… ni coincide en general con el global,
    # pero cada valor debe ser el de esa capa calculado aislado (con el denominador global)
    n = gel.sum() + void.sum()
    for z in range(6):
        g, v = gel[:, :, z], void[:, :, z]
        assert per[z] == pytest.approx(np.sum(r[:, :, z][v] >= r[:, :, z][g].min()) / n)
    cvz = metrics.cv(r, gel, layerwise=True)
    assert len(set(np.round(cvz, 12))) > 1  # VAMToolbox repetía el valor global en todas las capas


def test_get_inds_threshold_for_fractional_occupancy():
    t = _target(frac=True)
    gel, void = ops.get_inds(t, threshold=0.5)
    m = ops.circle_mask(*t.shape[:2])[:, :, None]
    np.testing.assert_array_equal(gel, (t >= 0.5) & m)
    np.testing.assert_array_equal(void, (t < 0.5) & m)
    assert not (gel & void).any()


def test_metric_masks_binary_equals_get_inds():
    t = _target()
    gel, void, info = ops.metric_masks(t)
    g0, v0 = ops.get_inds(t)
    np.testing.assert_array_equal(gel, g0)
    np.testing.assert_array_equal(void, v0)
    assert info["rule"] == "binary" and info["excluded_partial"] == 0


def test_metric_masks_guard_band_excludes_partial_voxels():
    t = np.zeros((21, 21, 4), dtype=np.float32)
    t[6:15, 6:15] = 1.0
    t[5, 6:15] = 0.4; t[15, 6:15] = 0.7                    # bordes parciales
    gel, void, info = ops.metric_masks(t)
    assert info["rule"] == "guard_band" and info["excluded_partial"] == 2 * 9 * 4
    assert not gel[5].any() and not void[5, 6:15].any() and not void[15, 6:15].any()
    # una dosis ideal (= ocupación) no tiene solape con la banda; al 50 % sí lo tenía
    assert metrics.ver(t, gel, void) == 0 and metrics.pw(t, gel, void) == pytest.approx(-1.0)


def test_metric_masks_thin_part_falls_back_to_half():
    t = np.zeros((21, 21, 3), dtype=np.float32)
    t[:, 10] = 0.6                                         # lámina sin ningún vóxel lleno
    gel, void, info = ops.metric_masks(t)
    assert info["rule"].startswith("threshold_0.5")
    np.testing.assert_array_equal(gel, ops.get_inds(t, 0.5)[0])


def test_clip_to_circle_not_inplace_by_default():
    x = np.ones((9, 9, 2))
    y = ops.clip_to_circle(x)
    assert x.min() == 1 and y[0, 0, 0] == 0 and y[4, 4, 1] == 1


def test_suggest_d_h_ladder():
    solid = np.ones((30, 30, 30), dtype=np.float32)
    assert dose.suggest_d_h(solid).d_h == 0.90
    thin = np.zeros((30, 30, 30), dtype=np.float32); thin[:, 14, :] = 1  # lámina de 1 vóxel
    s = dose.suggest_d_h(thin)
    assert s.d_h == 0.82 and s.mean_thickness_vox == pytest.approx(1.0)
    assert dose.suggest_d_h(np.zeros((4, 4, 4))).d_h == dose.DEFAULT_D_H


def test_config_is_per_instance():
    a = config.CALOptions().with_(learning_rate=0.05)
    b = config.CALOptions()
    assert a.learning_rate == 0.05 and b.learning_rate == 0.010
    assert config.CALOptions.for_resolution(512).n_iter == 800
    assert config.ProjectorSpec(n_angles=4).angles_deg().tolist() == [0, 90, 180, 270]


def test_filter_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    s = RNG.random((64, 30, 5))
    ref = filters.filter_sinogram(s, "cosine")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    got = filters.filter_sinogram(torch.tensor(s, device=dev), "cosine").cpu().numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)


def test_discretize_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    x = RNG.normal(size=(200,))
    np.testing.assert_array_equal(ops.discretize(torch.tensor(x), 3, [-1.0, 1.0]).numpy(),
                                  ops.discretize(x, 3, [-1.0, 1.0]))

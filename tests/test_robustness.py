"""vamcore.robustness: tolerancia de exposición (puro), perturbaciones (torch CPU) y prueba completa (GPU)."""
import json
from pathlib import Path

import numpy as np
import pytest

from vamcore.robustness import RobustnessSpec, exposure_tolerance, sample_scenarios


def test_exposure_tolerance_formula():
    t = exposure_tolerance(0.88, 0.80)
    r = 0.88 / 0.80
    assert t["tolerance"] == pytest.approx((r - 1) / (r + 1))
    # con el umbral centrado, ±ε de dosis lleva justo a los bordes
    T, e = t["threshold_rel"], t["tolerance"]
    assert 0.88 * (1 - e) == pytest.approx(T, rel=1e-9) and 0.80 * (1 + e) == pytest.approx(T, rel=1e-9)
    assert exposure_tolerance(0.80, 0.88)["tolerance"] == 0.0           # sin ventana
    assert exposure_tolerance(0.9, 0.0)["tolerance"] == 1.0


def test_scenarios_are_reproducible_and_scaled():
    a = sample_scenarios(RobustnessSpec(n_scenarios=5, seed=3), 36)
    b = sample_scenarios(RobustnessSpec(n_scenarios=5, seed=3), 36)
    assert [s["axis_offset_um"] for s in a] == [s["axis_offset_um"] for s in b]
    z = sample_scenarios(RobustnessSpec(n_scenarios=3, axis_offset_um=0, speed_err_pct=0, angle_jitter_deg=0,
                                        blur_um=0, flicker_pct=0, dose_err_pct=0), 10)
    assert all(s["axis_offset_um"] == 0 and s["dose_err"] == 0 and not s["flicker"].any() for s in z)


torch = pytest.importorskip("torch")
from vamcore.robustness import blur_uz, shift_u  # noqa: E402


def test_shift_u_integer_and_fractional():
    b = torch.rand(20, 3, 4, dtype=torch.float64)
    s = shift_u(b, 3)
    assert torch.equal(s[3:], b[:-3]) and torch.equal(s[:3], torch.zeros_like(s[:3]))
    h = shift_u(b, 0.5)
    assert torch.allclose(h[1:], 0.5 * (b[1:] + b[:-1]))
    assert torch.equal(shift_u(b, 0.0), b)


def test_blur_preserves_mass_and_identity():
    b = torch.zeros(41, 2, 41, dtype=torch.float64)
    b[20, :, 20] = 1.0
    y = blur_uz(b, 2.0)
    assert float(y.sum()) == pytest.approx(2.0, rel=1e-9)
    assert float(y[20, 0, 20]) < 0.1 and float(y[22, 0, 20]) > 0
    assert torch.equal(blur_uz(b, 0.0), b)


def test_robustness_on_golden():
    from vamcore.ops import metric_masks
    from vamcore.robustness import robustness_test

    g = np.load(Path(__file__).resolve().parent / "golden" / "vasculature_64.npz")
    meta = json.loads(str(g["meta"]))
    na = meta["n_angles"]
    angles = np.linspace(0, 360 - 360 / na, na)
    gel, void, _ = metric_masks(g["target"])
    gmin, vmax = float(g["recon"][gel].min()), float(g["recon"][void].max())
    zero = RobustnessSpec(n_scenarios=3, axis_offset_um=0, speed_err_pct=0, angle_jitter_deg=0, blur_um=0,
                          flicker_pct=0, dose_err_pct=0)
    r = robustness_test(g["sino"], g["target"], angles, 0.3, 300.0, gmin, vmax, zero)
    if gmin > vmax:  # con ventana, la pila nominal forma la pieza exacta
        assert r["nominal_pass"]
    assert all(s["shape_error_um"] == pytest.approx(r["scenarios"][0]["shape_error_um"]) for s in r["scenarios"])
    if gmin > vmax:
        assert r["pass_rate"] == 1.0 and r["shape_error_um_mean"] == pytest.approx(0.0, abs=1e-9)
    # perturbaciones grandes: más error que sin perturbar
    big = RobustnessSpec(n_scenarios=4, axis_offset_um=600, dose_err_pct=30, blur_um=600)
    rb = robustness_test(g["sino"], g["target"], angles, 0.3, 300.0, gmin, vmax, big)
    assert rb["shape_error_um_mean"] > r["shape_error_um_mean"] and rb["pass_rate"] < 1.0

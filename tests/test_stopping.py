"""vamcore.optim.stopping.StopRule: modo VER (como VAMToolbox) y modo margen de ventana de proceso. Puro: corre en CI."""
import pytest

from vamcore.config import StopCriteria
from vamcore.optim.stopping import StopRule


def feed(rule, seq, start=5, step=5):
    """seq: lista de (ver, pw); devuelve la iteración en que para o None."""
    it = start
    for ver, pw in seq:
        if rule.update(it, ver, pw):
            return it
        it += step
    return None


def test_ver_threshold_mode_matches_old_semantics():
    r = StopRule(StopCriteria(ver_threshold=0.0005, stall_patience=10), n_iter=500)
    assert feed(r, [(0.01, 0.1), (0.001, 0.05), (0.0004, 0.02)]) == 15
    assert r.reason == "threshold"


def test_ver_mode_ignores_pw():
    r = StopRule(StopCriteria(ver_threshold=None, stall_patience=0), n_iter=500)
    assert feed(r, [(0.01, -0.5)] * 5) is None and r.reason == "max_iter"


def test_pw_margin_stops_only_when_window_reached():
    r = StopRule(StopCriteria(ver_threshold=0.0005, stall_patience=10, pw_margin=0.03), n_iter=500)
    # VER ya es 0 (con banda pasa enseguida) pero el margen aún no: NO debe parar por umbral
    seq = [(0.0, 0.02), (0.0, -0.005), (0.0, -0.02), (0.0, -0.031)]
    assert feed(r, seq) == 20
    info = r.info(21, 0.1)
    assert r.reason == "pw_margin" and info["best_pw"] == pytest.approx(-0.031) and info["pw_margin"] == 0.03


def test_pw_mode_stalls_when_window_stops_growing():
    crit = StopCriteria(stall_patience=4, stall_min_iter=0, pw_margin=0.05, stall_pw_tol=0.001)
    r = StopRule(crit, n_iter=500)
    seq = [(0.0, 0.01), (0.0, -0.01), (0.0, -0.02)] + [(0.0, -0.0205)] * 10
    stop = feed(r, seq)
    assert r.reason == "stall" and stop is not None
    # mejora sostenida de > tol por evaluación: no se estanca
    r2 = StopRule(crit, n_iter=500)
    assert feed(r2, [(0.0, 0.01 - 0.002 * k) for k in range(20)]) is None


def test_pw_mode_respects_min_iter_and_requires_pw():
    r = StopRule(StopCriteria(stall_patience=2, stall_min_iter=100, pw_margin=0.05), n_iter=500)
    assert feed(r, [(0.0, -0.01)] * 10) is None  # iteraciones 5…50 < 100: no hay estancamiento
    with pytest.raises(ValueError):
        StopRule(StopCriteria(pw_margin=0.03), 10).update(5, 0.0)
    with pytest.raises(ValueError):
        StopRule(StopCriteria(pw_margin=0.0), 10)

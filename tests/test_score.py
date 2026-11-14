"""vamcore.pipeline._score: el Score Bio queda en 0–100 y el PW solo penaliza si hay solape (PW > 0).

Se extrae la función del fuente para no importar el pipeline (matplotlib, torch): corre en el CI.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

SRC = (Path(__file__).resolve().parents[1] / "vamcore" / "pipeline.py").read_text(encoding="utf-8")
_body = ast.parse(SRC).body
_nodes = [n for n in _body if (isinstance(n, ast.FunctionDef) and n.name in ("_score", "_verdict"))
          or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "VERDICTS" for t in n.targets))]
_ns = {"np": np}
exec(compile(ast.Module(_nodes, []), "_score", "exec"), _ns)
score_fn = _ns["_score"]

GOOD = dict(ver=0.0, cv=0.02, ipdr=0.12, pct_layers_overlap=0.0, error_improvement=0.1, n_valid_iters=10)


def bio(**kw):
    return score_fn({**GOOD, **kw}, 500)[1]


def test_negative_pw_is_not_a_bonus():
    assert bio(pw=-0.06) == pytest.approx(100.0)
    assert bio(pw=-0.5) <= 100.0


def test_positive_pw_penalises_monotonically():
    assert bio(pw=0.0) == pytest.approx(100.0)
    assert 100.0 > bio(pw=0.004) > bio(pw=0.02) > bio(pw=0.2)
    assert bio(pw=0.02) == pytest.approx(100 * np.exp(-0.06))


def test_range_and_verdict():
    s, b, verdict, _ = score_fn({**GOOD, "pw": 2.0, "ver": 0.5, "cv": 0.5, "ipdr": 0.9, "pct_layers_overlap": 90}, 500)
    assert 0.0 <= b <= 100.0 and verdict in ("POBRE", "FALLIDO")


# ── Nota combinada (vamcore.robustness.reliability_score) ─────────────────────
from vamcore.robustness import reliability_score  # noqa: E402


def test_reliability_score_full_and_partial():
    assert reliability_score(100, 5.0, 1.0)["score"] == pytest.approx(100.0)
    assert reliability_score(100, 8.0, 1.0)["score"] == pytest.approx(100.0)        # tolerancia > ref: cuenta entera
    r = reliability_score(100, 2.5, 0.75)
    assert r["score"] == pytest.approx(100 * (0.6 + 0.2 * 0.5 + 0.2 * 0.75))
    assert reliability_score(90, 5.0, 1.0)["score"] == pytest.approx(90.0)          # la dosis sigue mandando


def test_reliability_score_without_robustness_renormalises():
    r = reliability_score(100, 2.5, None)
    assert r["robustness_factor"] is None and r["weights"][2] == 0
    assert r["score"] == pytest.approx(100 * (0.75 + 0.25 * 0.5))
    assert reliability_score(100, 5.0, None)["score"] == pytest.approx(100.0)


def test_reliability_score_bounds_and_monotonic():
    assert reliability_score(100, 0.0, 0.0)["score"] == pytest.approx(60.0)
    assert reliability_score(100, 1.0, 0.5)["score"] < reliability_score(100, 2.0, 0.5)["score"]
    assert reliability_score(100, 2.0, 0.5)["score"] < reliability_score(100, 2.0, 0.9)["score"]
    assert 0 <= reliability_score(100, float("nan"), float("nan"))["score"] <= 100

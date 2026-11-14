"""Las referencias doradas de F0 (tests/golden/*_64.npz) están completas y son coherentes. Solo numpy."""
import json
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).resolve().parent / "golden"
FILES = sorted(GOLDEN.glob("*_64.npz"))
MODELS = {"vasculature", "cube", "sphere", "hollow_cylinder"}


def test_all_models_present():
    assert {f.stem.rsplit("_", 1)[0] for f in FILES} == MODELS


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_golden_structure(path):
    d = np.load(path)
    t, s, x, ver = d["target"], d["sino"], d["recon"], d["ver"]
    meta = json.loads(str(d["meta"]))
    n_xy, _, n_z = t.shape
    assert t.shape[0] == t.shape[1] and x.shape == t.shape
    assert s.shape == (n_xy, meta["n_angles"], n_z)
    assert 0 <= t.min() and t.max() <= 1 and np.isfinite(s).all() and s.min() >= 0
    assert np.nanmax(x) == pytest.approx(1.0, rel=1e-5)
    it = np.where(~np.isnan(ver))[0]
    assert it[0] == 0 and meta["stop_info"]["iterations"] <= len(ver)
    assert meta["metrics"]["ver"] >= 0 and meta["resolution"] == 64

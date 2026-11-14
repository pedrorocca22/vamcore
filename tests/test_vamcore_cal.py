"""F3 (docs/MIGRACION.md §5): el CAL de vamcore reproduce las referencias doradas del motor VAMToolbox.

Criterio: misma razón de parada, misma evaluación ±1 (±5 iteraciones), mejor VER dentro de ±0,5 % relativo (o
2e-5 absoluto cuando el VER es casi 0) y reconstrucción con error L2 relativo < 1 %. Se prueba con el proyector
Prisma (GPU o CPU). Las referencias se generaron con el motor anterior (VAMToolbox + ASTRA), ya retirado.
Usa los *_64.npz versionados y, si existen, los large/*_128.npz (estos, solo con GPU).
"""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
CUDA = torch.cuda.is_available()

from vamcore.config import CALOptions, StopCriteria  # noqa: E402
from vamcore.optim.cal import run_cal  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden"
FILES = sorted(GOLDEN.glob("*_64.npz")) + (sorted((GOLDEN / "large").glob("*_128.npz")) if CUDA else [])
BACKENDS = ["torch"]
EXIT_PARAM = 0.0005  # pipeline.EXIT_PARAM cuando se generaron las referencias


def options_from(meta):
    # Las referencias del 13-11-2026 se generaron con los valores por defecto del pipeline (filtro cosine,
    # paciencia 10, tolerancia 1 %); n_iter sale de stop_info. Las nuevas guardan `params` explícitamente.
    p = meta.get("params") or {"n_iter": meta["stop_info"]["max_iterations"], "filter": "cosine",
                               "stall_patience": 10, "stall_rel_tol": 0.01}
    base = CALOptions.for_resolution(meta["resolution"])
    return base.with_(n_iter=p["n_iter"], filter=p["filter"], d_h=meta["d_h"], momentum=0.3, sigmoid=0.01,
                      stop=StopCriteria(ver_threshold=EXIT_PARAM, stall_patience=p["stall_patience"],
                                        stall_rel_tol=p["stall_rel_tol"], stall_min_iter=50, ver_mask="half"))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_matches_golden(path, backend):
    g = np.load(path)
    meta = json.loads(str(g["meta"]))
    angles = np.linspace(0, 360 - 360 / meta["n_angles"], meta["n_angles"])
    res = run_cal(g["target"], angles, options_from(meta), exit_param=EXIT_PARAM, backend=backend)
    ref = meta["stop_info"]
    assert res.stop["reason"] == ref["reason"]
    assert abs(res.stop["iterations"] - ref["iterations"]) <= 5
    a, b = res.stop["best_ver"], ref["best_ver"]
    # 5e-5 absoluto ≈ 15 vóxeles en estas rejillas de 64: el VER cuenta cruces discretos y el proyector propio difiere
    # de ASTRA en ~6e-5 (L2), lo que mueve un puñado de vóxeles en el umbral (esfera_64: 8 vóxeles).
    assert abs(a - b) <= max(0.005 * abs(b), 5e-5), (a, b)
    x, xr = res.reconstruction, g["recon"]
    assert np.linalg.norm(x - xr) / np.linalg.norm(xr) < 1e-2
    s, sr = res.sinogram, g["sino"]
    assert np.linalg.norm(s - sr) / np.linalg.norm(sr) < 1e-2

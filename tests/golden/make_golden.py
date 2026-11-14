"""Genera referencias «doradas» con el motor de vamcore (proyector Prisma).

Las referencias versionadas (tests/golden/*_64.npz y large/*_128.npz) se generaron el 13-11-2026 con el motor
anterior (VAMToolbox + ASTRA, ya retirado) y son las que usa tests/test_vamcore_cal.py: no se sobrescriben salvo con
--overwrite. Por defecto este script escribe en tests/golden/regenerated/.

Para cada modelo (vasculatura de ejemplo, cubo, esfera, cilindro hueco) y resolución guarda un .npz con: objetivo
(ocupación), sinograma optimizado, reconstrucción, curva de VER (NaN donde no se evaluó) y metadatos (JSON).

Uso, desde la raíz del repo:
    python tests/golden/make_golden.py --res 64
    python tests/golden/make_golden.py --res 64 128 --overwrite
"""
import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
GOLDEN = Path(__file__).resolve().parent


def make_models(tmp: Path) -> dict:
    import trimesh

    models = {"vasculature": ROOT / "vamcore/resources/trifurcatedvasculature.stl"}
    cube = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=4.0)
    # cilindro hueco como malla única y estanca: anillo extruido
    ring = trimesh.creation.annulus(r_min=2.5, r_max=4.0, height=10.0, sections=64)
    for name, m in (("cube", cube), ("sphere", sphere), ("hollow_cylinder", ring)):
        assert m.is_watertight, name
        p = tmp / f"{name}.stl"
        m.export(p)
        models[name] = p
    return models


def run_one(stl: Path, res: int, out_dir: Path) -> dict:
    """El mismo camino que vamcore.pipeline.run (sin vial ni absorción), guardando los arrays intermedios."""
    from vamcore import metrics
    from vamcore.config import CALOptions, StopCriteria
    from vamcore.ops import metric_masks
    from vamcore.optim.cal import run_cal
    from vamcore.pipeline import EXIT_PARAM, STALL_MIN_ITER, STALL_PATIENCE, STALL_REL_TOL, _center_and_rotate, _Target
    from vamcore.voxelize import voxelize_mesh

    out_dir.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    mesh = _center_and_rotate(stl, 0.0, 0.0, 0.0, out_dir / "model_oriented.stl")
    occ, _, _ = voxelize_mesh(mesh, float(mesh.extents[2]) / res)
    tg = _Target(occ)
    d_h = float(tg.suggest_d_h())
    n_angles = 360
    angles = np.linspace(0, 360 - 360 / n_angles, n_angles)
    opts = CALOptions.for_resolution(res).with_(
        d_h=d_h, momentum=0.3, sigmoid=0.01, filter="cosine",
        stop=StopCriteria(ver_threshold=EXIT_PARAM, stall_patience=STALL_PATIENCE, stall_rel_tol=STALL_REL_TOL,
                          stall_min_iter=STALL_MIN_ITER, ver_mask="half"))
    r = run_cal(tg.array, angles, opts, exit_param=EXIT_PARAM)
    wall = time.perf_counter() - t
    gel, void, _ = metric_masks(tg.array)
    meta = {
        "model": stl.stem, "resolution": res, "metrics": metrics.all_metrics(r.reconstruction, gel, void),
        "params": {"n_iter": opts.n_iter, "filter": opts.filter, "stall_patience": STALL_PATIENCE,
                   "stall_rel_tol": STALL_REL_TOL},
        "stop_info": r.stop, "d_h": d_h, "n_angles": n_angles, "wall_s": wall,
        "engine": "vamcore+prisma", "python": platform.python_version(),
    }
    try:
        import torch
        meta.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:
        pass
    return dict(target=np.asarray(tg.array, np.float32), sino=r.sinogram.astype(np.float32),
                recon=r.reconstruction.astype(np.float32), ver=np.asarray(r.ver, np.float64),
                meta=np.array(json.dumps(meta, default=float, ensure_ascii=False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs="*", default=[64, 128])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true", help="sobrescribe las referencias versionadas")
    args = ap.parse_args()
    base = GOLDEN if args.overwrite else GOLDEN / "regenerated"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        models = make_models(td)
        for name, stl in models.items():
            if args.models and name not in args.models:
                continue
            for res in args.res:
                dest = base / (f"{name}_{res}.npz" if res <= 64 else f"large/{name}_{res}.npz")
                dest.parent.mkdir(parents=True, exist_ok=True)
                data = run_one(stl, res, td / f"run_{name}_{res}")
                np.savez_compressed(dest, **data)
                meta = json.loads(str(data["meta"]))
                print(f"[golden] {dest.relative_to(ROOT)} · objetivo {data['target'].shape} · sino {data['sino'].shape}"
                      f" · VER {meta['metrics'].get('ver'):.5f} · parada {meta['stop_info'].get('reason')}"
                      f" tras {meta['stop_info'].get('iterations')} it · {meta['wall_s']:.1f} s", flush=True)


if __name__ == "__main__":
    main()

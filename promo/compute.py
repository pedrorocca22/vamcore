"""Promo video, stage 1: run the real VAMCORE pipeline once and keep what the video shows.

    python promo/compute.py            # 512³, 360 angles (GPU, a few minutes)
    python promo/compute.py --res 128  # quick check

Everything shown in the video comes from here: the auto-orientation of a tilted input, the hero job (vial inside the
model, robustness test), dose snapshots after k iterations (separate runs of the same job with `iterations=k`), the
vial ray map and the exported frames. Output: promo/data/ (not versioned). Nothing is sent to any device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402

import vamcore  # noqa: E402
from vamcore import pipeline  # noqa: E402
from vamcore.ops import metric_masks  # noqa: E402
from vamcore.optics.vial import RESIN_PRESETS, VIAL_MATERIALS, VialOptics  # noqa: E402
from vamcore.orient import find_orientation  # noqa: E402
from vamcore.resources import example_stl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--angles", type=int, default=360)
ap.add_argument("--tilt", type=float, default=35.0, help="tilt applied to the example input (degrees about X)")
ap.add_argument("--snapshots", default="1,6,16,41,81", help="iteration counts for the dose snapshots")
ap.add_argument("--out", default=str(ROOT / "promo" / "data"))
args = ap.parse_args()

OUT = Path(args.out)
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
t_start = time.perf_counter()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / "vamcore").rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


# ── input: the bundled vasculature, tilted for the orientation demo ───────────
src = example_stl()
mesh = trimesh.load(src, force="mesh")
mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(args.tilt), [1, 0, 0]))
tilted = OUT / "input_tilted.stl"
mesh.export(tilted)

vial = dict(outer_diameter_mm=20.0, wall_thickness_mm=1.0,
            n_vial=VIAL_MATERIALS["borosilicate"], n_resin=RESIN_PRESETS["gelma"])
optics = VialOptics(**vial)

# ── 01 · orientation ──────────────────────────────────────────────────────────
t = time.perf_counter()
o = find_orientation(mesh.vertices, max_height_mm=pipeline.BUILD_H_MM, max_radius_mm=optics.printable_radius)
t_orient = time.perf_counter() - t
print(f"[orient] rx {o.rx:.2f}° ry {o.ry:.2f}° width {o.width_initial_mm:.2f} → {o.width_mm:.2f} mm "
      f"(−{o.reduction_pct:.1f} %) feasible={o.feasible}")

# ── hero job ──────────────────────────────────────────────────────────────────
cap: dict = {}
job = OUT / "job"
res = pipeline.run(tilted, job, rx=o.rx, ry=o.ry, rz=0.0, resolution=args.res, num_angles=args.angles,
                   cuda=True, vial=vial, vial_mode="model", robustness=True, capture=cap)
if not res["success"]:
    sys.exit(f"pipeline failed: {res['error']}")
m = res["metrics"]
print(f"[job] Score Bio {res['score_bio']:.1f} · VER {m['ver']:.5f} · opt {res['elapsed_time']:.1f} s · "
      f"total {res['total_time']:.1f} s")

target = np.asarray(cap["target"], dtype=np.float32)
recon = np.asarray(cap["reconstruction"], dtype=np.float32)
nx, ny, nz = target.shape
y_mid = int(np.argmax((target >= 0.5).sum(axis=(0, 2))))  # vertical plane through the most gel
z_idx = np.linspace(0, nz - 1, 48).round().astype(int)
gel, void, minfo = metric_masks(target)
g_vals, v_vals = recon[gel], recon[void]
bins = np.linspace(0, 1, 161)
rob = res.get("robustness") or {}

np.savez_compressed(
    OUT / "arrays.npz",
    target_xz=target[:, y_mid, :].astype(np.float16),
    recon_xz=recon[:, y_mid, :].astype(np.float16),
    target_z=(target[:, :, z_idx] * 255).round().astype(np.uint8),
    z_idx=z_idx,
    hist_gel=np.histogram(g_vals, bins)[0], hist_void=np.histogram(v_vals, bins)[0], hist_bins=bins,
    ver_curve=np.asarray(cap["ver"], dtype=np.float64),
)

# ── 04 · dose snapshots: the same job stopped after k iterations ──────────────
snaps = []
for k in [int(s) for s in args.snapshots.split(",")]:
    c: dict = {}
    d = OUT / f"snap_{k}"
    r = pipeline.run(tilted, d, rx=o.rx, ry=o.ry, rz=0.0, resolution=args.res, num_angles=args.angles, cuda=True,
                     vial=vial, vial_mode="model", robustness=False, iterations=k, capture=c)
    if not r["success"]:
        sys.exit(f"snapshot {k} failed: {r['error']}")
    x = np.asarray(c["reconstruction"], dtype=np.float32)
    np.save(OUT / f"snap_{k}_xz.npy", x[:, y_mid, :].astype(np.float16))
    snaps.append({"iterations": k, "ver": float(r["metrics"]["ver"]), "score_bio": float(r["score_bio"])})
    shutil.rmtree(d)  # its frames are not needed
    print(f"[snapshot] {k} iterations · VER {snaps[-1]['ver']:.5f}")

# ── 03 · vial ray map ─────────────────────────────────────────────────────────
u = np.linspace(-optics.r_outer, optics.r_outer, 61)
rt, rdelta, rT, rvalid = optics.ray_map(u)

frames = sorted(job.glob("frame_*.png"))
data = {
    "generated": datetime.now().isoformat(timespec="seconds"),
    "machine": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__, "python": platform.python_version(), "vamcore": vamcore.__version__},
    "provenance": {"input_stl": str(src.name), "input_sha256": sha256(src), "tilted_sha256": sha256(tilted),
                   "vamcore_code_sha256": code_hash(), "call": "vamcore.pipeline.run",
                   "params": {"resolution": args.res, "num_angles": args.angles, "vial": vial, "vial_mode": "model",
                              "tilt_deg": args.tilt}},
    "orient": {"rx": o.rx, "ry": o.ry, "rotation": o.rotation, "width_initial_mm": o.width_initial_mm,
               "width_mm": o.width_mm, "reduction_pct": o.reduction_pct, "feasible": o.feasible,
               "height_mm": o.height_mm, "radius_mm": o.radius_mm, "time_s": t_orient},
    "job": {"score_bio": res["score_bio"], "verdict": res["verdict"], "elapsed_s": res["elapsed_time"],
            "total_s": res["total_time"], "stage_times": res.get("stage_times", {}),
            "metrics": {k: m.get(k) for k in ("ver", "pw", "cv", "ipdr", "exposure_tolerance_pct", "iterations_run",
                                              "stop_reason", "d_h")},
            "grid": [nx, ny, nz], "y_mid": y_mid, "metric_rule": minfo.get("rule"),
            "gel_min": float(g_vals.min()), "void_max": float(v_vals.max()),
            "frames": len(frames), "frames_dir": str(job),
            "frame_size": list(Image.open(frames[0]).size) if frames else None},
    "robustness": {"n_pass": rob.get("n_pass"), "n_scenarios": rob.get("n_scenarios"),
                   "threshold_rel": rob.get("threshold_rel"), "shape_tol_um": rob.get("shape_tol_um"),
                   "shape_error_um_p90": rob.get("shape_error_um_p90"),
                   "scenarios": [{"pass": s["pass"], "shape_error_um": s["shape_error_um"]}
                                 for s in rob.get("scenarios", [])]},
    "snapshots": snaps,
    "vial": {**optics.summary(), "ray_u": u.tolist(), "ray_t": rt.tolist(), "ray_delta": rdelta.tolist(),
             "ray_T": rT.tolist(), "ray_valid": rvalid.tolist()},
    "historical": {"voxelize_512_previous_s": 195.5, "voxelize_512_previous_note":
                   "previous GPU voxelizer, RTX 4060 Ti, 13-11-2026 (docs/MIGRACION.md)"},
    "compute_wall_s": None,
}
data["compute_wall_s"] = time.perf_counter() - t_start
(OUT / "data.json").write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")
print(f"[done] {len(frames)} frames · data in {OUT} · {data['compute_wall_s']:.0f} s")

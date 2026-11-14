"""
vamcore.pipeline — de un STL a la pila de imágenes del proyector (antes cal_core.py)
====================================================================================
Orquesta todo el proceso de VAMCORE:

  orientar → voxelizar (por secciones, ocupación parcial) → d_h (EDT) → CAL en PyTorch con el proyector Prisma
  (absorción de la resina y refracción del vial dentro del modelo) → métricas (banda de guarda) → tolerancia de
  exposición y prueba de robustez → Score Bio → fotogramas a escala física → result.json, CSV e imágenes.

Uso:
  from vamcore.pipeline import run
  result = run("modelo.stl", "salida/", resolution=256, vial=dict(outer_diameter_mm=20, wall_thickness_mm=1,
               n_vial=1.483, n_resin=1.35))
  # o desde la línea de comandos:  python -m vamcore modelo.stl salida/ --res 256

Dependencias: numpy, scipy, trimesh, torch (CUDA si hay GPU; si no, CPU), matplotlib, Pillow y OpenCV (exportación).
Partes del cálculo de métricas y puntuación derivan de VAMToolbox (GPL-3.0); ver la cabecera de cada módulo.
"""

import matplotlib

matplotlib.use("Agg")

import sys
import csv
import time
import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from vamcore import dose as _dose
from vamcore import metrics as _metrics
from vamcore.ops import clip_to_circle as _clip_to_circle
from vamcore.ops import get_inds as _get_inds


class _Target:
    """Objetivo voxelizado: array de ocupación recortado al círculo inscrito y máscaras gel/vacío."""

    def __init__(self, occupancy, gel_threshold=0.5):
        a = np.atleast_3d(np.asarray(occupancy))
        self.gel_inds, self.void_inds = _get_inds(a, gel_threshold)
        self.array = _clip_to_circle(a)

    def suggest_d_h(self):
        s = _dose.suggest_d_h(self.array)
        if s.mean_thickness_vox is not None:
            print(f"       [Auto-DH] mean_thickness={s.mean_thickness_vox:.2f} voxels")
        print(f"       [Auto-DH] suggested d_h={s.d_h:.2f} ({s.tag})")
        return s.d_h

# =============================================================================
# PARÁMETROS FIJOS
# =============================================================================

RESOLUTION = 128
NUM_ANGLES = 360
N_ITER = 250
FILTER = "cosine"
D_L = 0.70
CUDA = True
EXIT_PARAM = 0.0005

# Convergencia early-stop
EARLY_STOP_THRESHOLD = 1e-4
EARLY_STOP_PATIENCE = 5

# Parada por estancamiento DENTRO del bucle CAL (el VER se evalúa cada 5 iteraciones):
# para si en STALL_PATIENCE evaluaciones (= 5·STALL_PATIENCE iteraciones) el mejor VER no baja un STALL_REL_TOL relativo.
STALL_PATIENCE = 10
STALL_REL_TOL = 0.01
STALL_MIN_ITER = 50

# =============================================================================
# CONVERSIÓN SINOGRAMA → BITPLANES 1-BIT (para proyección a alta velocidad)
# =============================================================================

# Constantes físicas del proyector (actualizar tras calibración)
PROJECTOR_PIXEL_UM = 14.3  # µm — portrait, ×3.5 acercado
BUILD_D_MM = 20.0  # mm — diámetro máximo de impresión
BUILD_H_MM = 25.0  # mm — altura máxima de impresión
PROJ_FRAME_W = int(BUILD_D_MM * 1000 / PROJECTOR_PIXEL_UM)  # 874 px
PROJ_FRAME_H = int(BUILD_H_MM * 1000 / PROJECTOR_PIXEL_UM)  # 1748 px

# =============================================================================
# PREPARACIÓN DE GEOMETRÍA
# =============================================================================


def _Rx(a):
    a = np.radians(a)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _Ry(a):
    a = np.radians(a)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _Rz(a):
    a = np.radians(a)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _center_and_rotate(stl_path, rx, ry, rz, out_path):
    """
    Carga el STL, aplica rotación y centra el modelo en el origen.
    Escribe el resultado en out_path (el STL original no se toca) para que
    TargetGeometry lo vea ya transformado — por eso rot_angles en TargetGeometry debe ser [0,0,0].
    """
    import trimesh

    mesh = trimesh.load(str(stl_path), force="mesh")

    # 1. Rotación (Rx -> Ry -> Rz)
    if rx != 0 or ry != 0 or rz != 0:
        R = _Rz(rz) @ _Ry(ry) @ _Rx(rx)
        R4 = np.eye(4)
        R4[:3, :3] = R
        mesh.apply_transform(R4)

    # 2. Centrado
    mesh.vertices -= mesh.bounding_box.centroid

    mesh.export(str(out_path))
    return mesh


# =============================================================================
# MÉTRICAS GLOBALES
# =============================================================================


def _compute_metrics(target_geo, recon_array, error_array, elapsed):
    m = {}
    gd = recon_array[target_geo.gel_inds]
    vd = recon_array[target_geo.void_inds]

    try:
        m["cv"] = float(np.std(gd) / np.mean(gd)) if np.mean(gd) > 0 else float("inf")
    except:
        m["cv"] = float("inf")
    gi, vi = target_geo.gel_inds, target_geo.void_inds
    for key, fn in (("ver", _metrics.ver), ("pw", _metrics.pw), ("ipdr", _metrics.ipdr)):
        try:
            m[key] = float(fn(recon_array, gi, vi))
        except Exception:
            m[key] = float("inf")

    m["min_gel_dose"] = float(np.min(gd)) if len(gd) > 0 else 0.0
    m["max_gel_dose"] = float(np.max(gd)) if len(gd) > 0 else 0.0
    m["mean_gel_dose"] = float(np.mean(gd)) if len(gd) > 0 else 0.0
    m["max_void_dose"] = float(np.max(vd)) if len(vd) > 0 else 0.0
    m["mean_void_dose"] = float(np.mean(vd)) if len(vd) > 0 else 0.0
    m["dose_contrast"] = (
        m["mean_gel_dose"] / m["max_void_dose"]
        if m["max_void_dose"] > 0
        else float("inf")
    )

    ve = error_array[~np.isnan(error_array)]
    m["final_error"] = float(ve[-1]) if len(ve) > 0 else float("inf")
    m["initial_error"] = float(ve[0]) if len(ve) > 0 else float("inf")
    m["error_improvement"] = float(m["initial_error"] - m["final_error"])
    m["n_valid_iters"] = int(len(ve))

    conv = -1
    if len(ve) > EARLY_STOP_PATIENCE:
        for i in range(EARLY_STOP_PATIENCE, len(ve)):
            w = ve[i - EARLY_STOP_PATIENCE : i]
            if len(w) > 1:
                rc = np.abs(np.diff(w))
                if np.all(rc / max(np.max(np.abs(w)), 1e-10) < EARLY_STOP_THRESHOLD):
                    conv = i - EARLY_STOP_PATIENCE
                    break
    m["converged_iter"] = conv
    m["elapsed_time"] = elapsed
    return m


# =============================================================================
# ANÁLISIS POR CAPA Z
# =============================================================================


def _compute_layer_metrics(target_geo, recon_array):
    if recon_array.ndim != 3:
        return {}

    cv_z, pw_z = [], []
    # Mismas máscaras que las métricas globales (con banda de guarda si hay ocupación parcial)
    for z in range(recon_array.shape[2]):
        gd = recon_array[:, :, z][target_geo.gel_inds[:, :, z]]
        vd = recon_array[:, :, z][target_geo.void_inds[:, :, z]]
        if len(gd) < 5 or len(vd) < 5:
            cv_z.append(float("nan"))
            pw_z.append(float("nan"))
            continue
        mg = float(np.mean(gd))
        cv_z.append(float(np.std(gd) / mg) if mg > 0 else float("nan"))
        pw_z.append(float(gd.min() - vd.max()))

    vcv = np.array([v for v in cv_z if not np.isnan(v)])
    vpw = np.array([v for v in pw_z if not np.isnan(v)])

    r = {"n_active_layers": int(len(vcv)), "_cv_per_layer": cv_z, "_pw_per_layer": pw_z}

    if len(vcv) > 0:
        r.update(
            {
                "cv_z_mean": float(np.mean(vcv)),
                "cv_z_std": float(np.std(vcv)),
                "cv_z_p95": float(np.percentile(vcv, 95)),
                "cv_z_worst": float(np.max(vcv)),
            }
        )
    if len(vpw) > 0:
        r.update(
            {
                "pw_z_mean": float(np.mean(vpw)),
                "pw_z_std": float(np.std(vpw)),
                "pw_z_p5": float(np.percentile(vpw, 5)),
                "pw_z_worst": float(np.min(vpw)),
                "pct_layers_overlap": float(np.mean(vpw < 0) * 100),
            }
        )
    return r


# =============================================================================
# SCORE Y VEREDICTO
# =============================================================================


VERDICTS = {85: "EXCELENTE", 70: "BUENO", 55: "ACEPTABLE", 40: "POBRE", 0: "FALLIDO"}


def _verdict(score):
    return next(v for thr, v in sorted(VERDICTS.items(), reverse=True) if score >= thr)


def _score(metrics, max_iter):
    score, penalties = 100.0, []

    ver = metrics.get("ver", 1.0)
    if ver > 0.3:
        p = min(40, ver * 100)
        score -= p
        penalties.append(f"VER alto ({ver:.3f}): -{p:.0f}")
    elif ver > 0.1:
        p = ver * 50
        score -= p
        penalties.append(f"VER moderado ({ver:.3f}): -{p:.0f}")

    cv = metrics.get("cv", 1.0)
    if cv > 0.3:
        p = min(25, cv * 50)
        score -= p
        penalties.append(f"CV alto ({cv:.3f}): -{p:.0f}")
    elif cv > 0.15:
        p = cv * 20
        score -= p
        penalties.append(f"CV moderado ({cv:.3f}): -{p:.0f}")

    ipdr = metrics.get("ipdr", 1.0)
    if ipdr > 0.5:
        p = min(15, ipdr * 20)
        score -= p
        penalties.append(f"IPDR alto ({ipdr:.3f}): -{p:.0f}")

    pct = metrics.get("pct_layers_overlap", 0) or 0
    if pct > 20:
        p = min(10, pct * 0.3)
        score -= p
        penalties.append(f"Solapamiento {pct:.0f}% capas: -{p:.0f}")

    score = max(0.0, min(100.0, score))

    pw = metrics.get("pw", 0)
    # PW > 0 = solape gel/vacío → penaliza; PW ≤ 0 = hay ventana de proceso → sin penalización (antes exp(-3·PW)
    # con PW negativo multiplicaba por > 1 y el Score Bio podía pasar de 100).
    bio = score * float(np.exp(-3.0 * max(0.0, pw)) * np.exp(-1.5 * max(0.0, ipdr - 0.25)))
    bio = max(0.0, min(100.0, bio))

    verdict = _verdict(bio)

    imp = metrics.get("error_improvement", 1.0)
    iters = metrics.get("n_valid_iters", 0)
    if (imp < 0.05 or iters >= max_iter) and bio < 70:
        penalties.insert(
            0,
            "AVISO: El optimizador no convergió — considera rotar el modelo y repetir",
        )

    metrics["score_bio"] = bio
    metrics["bio_multiplier"] = bio / score if score > 0 else 0.0
    return score, bio, verdict, penalties


# =============================================================================
# EXPORTACIÓN DE FRAMES
# =============================================================================
def _export_frames_physical(
    sino_xaz, save_dir, voxel_pitch_um, frame_w=PROJ_FRAME_W, frame_h=PROJ_FRAME_H,
    pixel_um=PROJECTOR_PIXEL_UM, col_pitch_um=None,
):
    """
    Exporta los frames 8-bit a resolución física del proyector con recorte ajustado.

    sino_xaz : array (n_x, n_angles, n_z) — convención CAL. Puede ser el sinograma ideal o el ya
               corregido por el vial (en ese caso n_x cubre el diámetro del vial y no es cuadrado).
    Escala horizontal y vertical por separado (antes se forzaba un cuadrado max(n_z, n_x)), centra
    la imagen en el lienzo y, si no cabe, recorta simétricamente. col_pitch_um: separación de columnas en el plano del
    proyector si difiere del vóxel (vial dentro del modelo: pitch·n_resina/n_exterior); en vertical, el vóxel. Calcula la caja envolvente global
    para recortar el negro y ahorrar transferencia.
    """
    import cv2
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    import os

    save_dir = Path(save_dir)

    sino_arr = np.array(sino_xaz, dtype=np.float32)
    if sino_arr.ndim == 3:
        sino_arr = sino_arr.transpose(1, 2, 0)  # (n_angles, n_z, n_x)
    elif sino_arr.ndim == 2:
        sino_arr = sino_arr.T[:, np.newaxis, :]  # (n_angles, 1, n_x)
    n_angles, n_z, n_x = sino_arr.shape

    s_max = sino_arr.max()
    if s_max > 1e-10:
        sino_arr /= s_max

    # ── Geometría física ─────────────────────────────────────────────────────
    scale = voxel_pitch_um / pixel_um
    scale_x = (col_pitch_um or voxel_pitch_um) / pixel_um
    full_w = max(1, int(round(n_x * scale_x)))
    full_h = max(1, int(round(n_z * scale)))

    # origen en la imagen escalada (recorte si es mayor que el lienzo) y destino en el lienzo
    src_x0 = max(0, (full_w - frame_w) // 2)
    src_y0 = max(0, (full_h - frame_h) // 2)
    dst_x0 = max(0, (frame_w - full_w) // 2)
    dst_y0 = max(0, (frame_h - full_h) // 2)
    cw, ch = min(full_w, frame_w), min(full_h, frame_h)

    print(
        f"[frames] {n_angles} ángulos | scale={scale:.2f}× | "
        f"imagen={full_w}×{full_h}px | canvas={frame_w}×{frame_h}px"
    )

    def _place(i):
        frame_phys = cv2.resize(sino_arr[i], (full_w, full_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((frame_h, frame_w), dtype=np.float32)
        canvas[dst_y0 : dst_y0 + ch, dst_x0 : dst_x0 + cw] = frame_phys[
            src_y0 : src_y0 + ch, src_x0 : src_x0 + cw
        ]
        return canvas

    # ── PASO 1: bounding box global ──────────────────────────────────────────
    g_rmin, g_rmax, g_cmin, g_cmax = frame_h, 0, frame_w, 0
    for i in range(n_angles):
        nz = np.nonzero(_place(i) > 0.0)
        if len(nz[0]) == 0:
            continue
        g_rmin, g_rmax = min(g_rmin, int(nz[0].min())), max(g_rmax, int(nz[0].max()))
        g_cmin, g_cmax = min(g_cmin, int(nz[1].min())), max(g_cmax, int(nz[1].max()))

    if g_rmax < g_rmin or g_cmax < g_cmin:
        g_rmin, g_rmax, g_cmin, g_cmax = 0, frame_h - 1, 0, frame_w - 1

    MARGIN = 8
    g_rmin = max(0, g_rmin - MARGIN)
    g_rmax = min(frame_h - 1, g_rmax + MARGIN)
    g_cmin = max(0, g_cmin - MARGIN)
    g_cmax = min(frame_w - 1, g_cmax + MARGIN)

    crop_h = g_rmax - g_rmin + 1
    crop_w = g_cmax - g_cmin + 1
    saving = (1 - (crop_h * crop_w) / (frame_h * frame_w)) * 100

    # ── PASO 2: frames recortados ────────────────────────────────────────────
    frames_list = []
    for i in range(n_angles):
        crop_data = _place(i)[g_rmin : g_rmax + 1, g_cmin : g_cmax + 1]
        frames_list.append((i, (crop_data * 255).clip(0, 255).astype(np.uint8)))

    def _save_single_frame(args):
        i, img_8bit = args
        Image.fromarray(img_8bit).save(save_dir / f"frame_{i:04d}.png")

    n_workers = min(32, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(executor.map(_save_single_frame, frames_list))

    print(f"[frames] Bounding box crop: {crop_w}×{crop_h}px | Ahorro: {saving:.1f}%")

    return {
        "crop_x": g_cmin,
        "crop_y": g_rmin,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "full_w": frame_w,
        "full_h": frame_h,
        "pixel_um": pixel_um,
        "savings_pct": round(saving, 1),
    }


# =============================================================================


def _save_slices(recon_array, output_dir, name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Reconstrucción: {name}", fontsize=13, fontweight="bold")
    if recon_array.ndim == 3:
        ny, nx, nz = recon_array.shape
        for ax, sl, title in [
            (axes[0], recon_array[:, :, nz // 2], f"XY z={nz // 2}"),
            (axes[1], recon_array[ny // 2, :, :], f"XZ y={ny // 2}"),
            (axes[2], recon_array[:, nx // 2, :], f"YZ x={nx // 2}"),
        ]:
            ax.imshow(sl, cmap="CMRmap", interpolation="antialiased")
            ax.set_title(title)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        Path(output_dir) / "reconstruction_slices.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def _save_layer_profile(layer_metrics, output_dir, name):
    cv = np.array(layer_metrics.get("_cv_per_layer", []))
    pw = np.array(layer_metrics.get("_pw_per_layer", []))
    if len(cv) == 0:
        return
    z = np.arange(len(cv))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Análisis por capa Z: {name}", fontsize=12, fontweight="bold")

    ax1.plot(z, cv, color="steelblue", linewidth=1.2, label="CV(z)")
    if not np.all(np.isnan(cv)):
        p95 = np.nanpercentile(cv, 95)
        ax1.axhline(
            p95, color="orange", linestyle="--", alpha=0.8, label=f"P95={p95:.3f}"
        )
        ax1.fill_between(z, 0, cv, where=(cv > p95), color="orange", alpha=0.3)
    ax1.set_xlabel("Capa Z")
    ax1.set_ylabel("CV")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(z, pw, color="seagreen", linewidth=1.2, label="PW(z)")
    ax2.axhline(0, color="red", linewidth=0.8, alpha=0.6)
    ax2.fill_between(
        z, pw, 0, where=(pw < 0), color="red", alpha=0.3, label="Solapamiento"
    )
    pct = layer_metrics.get("pct_layers_overlap", 0) or 0
    ax2.set_title(f"PW por capa — {pct:.1f}% solapamiento")
    ax2.set_xlabel("Capa Z")
    ax2.set_ylabel("PW")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "layer_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_histogram(recon_array, target_geo, output_dir, name):
    fig, ax = plt.subplots(figsize=(8, 4))
    gd = recon_array[target_geo.gel_inds]
    vd = recon_array[target_geo.void_inds]
    top = max(np.max(gd), np.max(vd)) * 1.1
    bins = np.linspace(0, top, 80)
    ax.hist(
        gd.flatten(), bins=bins, alpha=0.6, color="green", label="Gel", density=True
    )
    ax.hist(vd.flatten(), bins=bins, alpha=0.6, color="red", label="Void", density=True)
    ax.set_xlabel("Dosis normalizada")
    ax.set_ylabel("Densidad")
    ax.set_title(f"Distribución de dosis: {name}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "dose_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CSV DE MÉTRICAS
# =============================================================================

FIELDNAMES = [
    "timestamp",
    "model",
    "run_name",
    "method",
    "resolution",
    "num_angles",
    "cuda",
    "filter",
    "n_iter",
    "d_h",
    "d_l",
    "rot_rx",
    "rot_ry",
    "rot_rz",
    "score",
    "score_bio",
    "bio_multiplier",
    "verdict",
    "ver",
    "cv",
    "pw",
    "ipdr",
    "min_gel_dose",
    "max_gel_dose",
    "mean_gel_dose",
    "max_void_dose",
    "mean_void_dose",
    "dose_contrast",
    "final_error",
    "initial_error",
    "error_improvement",
    "n_valid_iters",
    "converged_iter",
    "n_active_layers",
    "cv_z_mean",
    "cv_z_std",
    "cv_z_p95",
    "cv_z_worst",
    "pw_z_mean",
    "pw_z_std",
    "pw_z_p5",
    "pw_z_worst",
    "pct_layers_overlap",
    "elapsed_time",
    "export_time",
    "total_time",
    "penalties",
    "error",
]


def _write_csv(row_data, filepath):
    exists = Path(filepath).exists()
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
        w.writerow(row_data)


def _fmt(v):
    if isinstance(v, float):
        return "" if (v != v or v == float("inf")) else f"{v:.6f}"
    return str(v) if v is not None else ""


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================


def run(
    stl_path,
    output_dir=None,
    *,
    rx=0.0,
    ry=0.0,
    rz=0.0,
    num_angles=None,
    resolution=None,
    filter_type=None,
    method="CAL",
    image_dims=(1024, 1024),
    cuda=None,
    iterations=None,
    learning_rate=None,
    exit_param=None,
    stall_patience=None,
    stall_rel_tol=None,
    vial=None,
    projector_pixel_um=None,
    voxel_pitch_um=None,
    stop_ver_mask="half",
    stop_pw_margin=None,
    robustness=True,
    absorption_cm=0.0,
    vial_mode="model",
    capture=None,
):
    """
    Ejecuta la optimización CAL completa sobre un STL.

    vial : None | dict | vamcore.optics.vial.VialOptics
        Si se indica, las imágenes se corrigen por la refracción del vial (impresión sin baño de
        index matching). dict con claves de VialOptics: outer_diameter_mm, wall_thickness_mm,
        n_vial, n_resin y opcionalmente n_outside.
    projector_pixel_um : tamaño del píxel del proyector en el plano del vial (por defecto PROJECTOR_PIXEL_UM).
    voxel_pitch_um : tamaño de vóxel explícito en µm. Si no se da: alto_Z / resolution.
    stop_ver_mask : máscaras del VER que decide la parada: "half" (al 50 %) o "guard_band" (excluye los vóxeles
        parciales, como el Score Bio).
    stop_pw_margin : si se da (δ > 0), para cuando la ventana de proceso PW ≤ −δ en lugar de por el umbral de VER
        (pensado para usarse con stop_ver_mask="guard_band").
    absorption_cm : coeficiente de absorción de la resina α (cm⁻¹, Beer–Lambert). 0 = sin absorción. La luz entra por
        la pared interior del vial (o por el borde del área de trabajo si no hay vial).
    vial_mode : con vial, "model" (por defecto: la refracción del vial va dentro del modelo y el CAL optimiza
        directamente la imagen que emite el proyector) o "post" (se optimiza el sinograma ideal y después se
        corrigen las imágenes).
    capture : dict opcional que se rellena con los arrays intermedios (target, sinogram = imagen del proyector,
        reconstruction, ver, d_h, n_iter, learning_rate, filter, voxel_pitch_mm, angles). Para depurar o para el vídeo.
    robustness : True (por defecto) para estimar la tolerancia de exposición y ejecutar la prueba de robustez
        (vamcore.robustness); False la omite; o un RobustnessSpec.
    """
    if str(method).upper() != "CAL":
        raise ValueError(f"Método no soportado: {method!r} (solo CAL)")
    if vial_mode not in ("post", "model"):
        raise ValueError(f"vial_mode debe ser 'post' o 'model', no {vial_mode!r}")
    absorption_cm = float(absorption_cm or 0.0)
    if absorption_cm < 0:
        raise ValueError("absorption_cm debe ser ≥ 0")
    from vamcore.optics.vial import VialOptics, check_fit, correct_sinogram

    pixel_um = float(projector_pixel_um) if projector_pixel_um else PROJECTOR_PIXEL_UM
    if vial is not None and not isinstance(vial, VialOptics):
        vial = VialOptics(**vial)
    if vial is not None:
        vial.validate()
    stl_path = Path(stl_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    use_cuda = CUDA if cuda is None else cuda

    res = resolution if resolution is not None else RESOLUTION
    n_angles = num_angles if num_angles is not None else NUM_ANGLES
    filt = filter_type if filter_type is not None else FILTER

    # Escalera dinámica de iteraciones y Learning Rate por resolución
    if iterations is not None:
        n_iter_dynamic = iterations
        lr_dynamic = 0.015 if iterations >= 300 else 0.020
    elif res >= 512:
        n_iter_dynamic = 800
        lr_dynamic = 0.020
    elif res >= 256:
        n_iter_dynamic = 500
        lr_dynamic = 0.015
    elif res <= 64:
        n_iter_dynamic = 80
        lr_dynamic = 0.010
    else:
        n_iter_dynamic = 250
        lr_dynamic = 0.010

    if learning_rate is not None:
        lr_dynamic = learning_rate

    if output_dir is None:
        output_dir = stl_path.parent / f"cal_result_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    result = {
        "success": False,
        "run_name": "",
        "output_dir": str(output_dir),
        "metrics": {},
        "orient": {"rx": rx, "ry": ry, "rz": rz},
        "score": 0.0,
        "score_bio": 0.0,
        "verdict": "ERROR",
        "penalties": [],
        "elapsed_time": 0.0,
        "export_time": 0.0,
        "total_time": 0.0,
        "error": None,
    }

    try:
        # ── 1. Preparar Geometría (Manual) ────────────────────────────────────
        oriented_stl = output_dir / "model_oriented.stl"
        stage_times = {}  # segundos por etapa (orientación, voxelización, d_h, optimización, vial, fotogramas)
        _t = time.perf_counter()
        mesh_o = _center_and_rotate(stl_path, rx, ry, rz, oriented_stl)
        stage_times["orient"] = time.perf_counter() - _t

        # Tamaño físico del vóxel: alto_Z / resolución (como voxelizeTarget), o el indicado en µm
        if voxel_pitch_um:
            voxel_pitch_mm = float(voxel_pitch_um) / 1000.0
        else:
            voxel_pitch_mm = float(mesh_o.extents[2]) / res
        part_radius_mm = float(np.max(np.hypot(mesh_o.vertices[:, 0], mesh_o.vertices[:, 1])))
        if vial is not None:
            check_fit(part_radius_mm, vial)  # falla antes de optimizar si no cabe

        from vamcore.voxelize import mesh_report, native_pitch_mm, voxelize_mesh

        mesh_info = mesh_report(mesh_o)
        native_mm = native_pitch_mm(pixel_um, vial.n_resin if vial else None, vial.n_outside if vial else 1.0)

        # ── 2. Voxelización ───────────────────────────────────────────────────
        t_vox = time.perf_counter()
        occ, vgrid, vox_stats = voxelize_mesh(mesh_o, voxel_pitch_mm)
        # El optimizador usa la ocupación fraccional; las máscaras iniciales clasifican gel/vacío al 50 %
        target_geo = _Target(occ, gel_threshold=0.5)
        vox_stats["voxelizer"] = "slices"
        vox_stats["time_s"] = time.perf_counter() - t_vox
        vox_stats["pitch_vs_native"] = voxel_pitch_mm / native_mm
        print(
            f"[voxel] vóxel {voxel_pitch_mm * 1000:.1f} µm "
            f"({vox_stats['pitch_vs_native']:.1f}× la separación de rayos en la resina, {native_mm * 1000:.1f} µm) · "
            f"rejilla {target_geo.array.shape} · {vox_stats['time_s']:.2f} s"
            + (f" · error de volumen {vox_stats['volume_error_pct']:+.3f} %" if "volume_error_pct" in vox_stats else "")
        )
        for w in mesh_info["warnings"]:
            print(f"[malla] ⚠ {w}")

        # ── 3. d_h via EDT ────────────────────────────────────────────────────
        _t = time.perf_counter()
        d_h = float(target_geo.suggest_d_h())
        stage_times["d_h"] = time.perf_counter() - _t

        # ── 4. Geometría de proyección ────────────────────────────────────────
        angles = np.linspace(0, 360 - 360 / n_angles, n_angles)

        # ── 5. Optimización ───────────────────────────────────────────────
        run_name = f"{method}_rx{rx:.0f}_ry{ry:.0f}_dh{d_h:.3f}_dl{D_L:.2f}_{filt}"
        result["run_name"] = run_name

        exit_p = exit_param if exit_param is not None else EXIT_PARAM
        # stall_patience=0 desactiva la parada por estancamiento
        stall_p = STALL_PATIENCE if stall_patience is None else int(stall_patience)
        stall_tol = STALL_REL_TOL if stall_rel_tol is None else float(stall_rel_tol)

        # Dispositivo: GPU si se pide y la hay; si no, CPU (ruta PyTorch pura, lenta)
        from vamcore._backend import default_device

        calc_device = default_device(None) if use_cuda else "cpu"
        if calc_device == "cpu":
            print("[CAL] sin GPU: ruta PyTorch en CPU (lenta)")
        # Absorción: α por vóxel y radio de la resina en vóxeles (entrada de la luz: pared interior del vial)
        r_resin_mm = float(vial.r_inner) if vial is not None else BUILD_D_MM / 2
        proj_kw = ({"alpha": absorption_cm / 10.0 * voxel_pitch_mm, "r_resin": r_resin_mm / voxel_pitch_mm}
                   if absorption_cm > 0 else {})
        if absorption_cm > 0:
            print(f"[absorción] α ={absorption_cm:g} cm⁻¹ · resina Ø{2 * r_resin_mm:.2f} mm · "
                  f"transmisión hasta el eje {np.exp(-absorption_cm / 10.0 * r_resin_mm) * 100:.1f} %")
        pw_curve = None
        vial_in_model = vial is not None and vial_mode == "model"
        vial_R = None
        t_opt = time.perf_counter()
        # CAL con el proyector Prisma
        from types import SimpleNamespace

        from vamcore.config import CALOptions, StopCriteria
        from vamcore.optim.cal import run_cal

        copts = CALOptions(
            n_iter=n_iter_dynamic, learning_rate=lr_dynamic, momentum=0.3, sigmoid=0.01, filter=filt, d_h=d_h,
            stop=StopCriteria(ver_threshold=exit_p, stall_patience=stall_p,
                              stall_rel_tol=stall_tol, stall_min_iter=STALL_MIN_ITER,
                              ver_mask=stop_ver_mask,
                              pw_margin=float(stop_pw_margin) if stop_pw_margin else None),
        )
        if vial_in_model:
            # F6c: el CAL optimiza la imagen del proyector con el vial dentro del modelo (VialProjector = A ∘ G)
            import torch

            from vamcore.filters import filter_sinogram
            from vamcore.optics.vial_operator import VialProjector, VialResampler
            from vamcore.projector import make_projector

            _tv = time.perf_counter()
            base = make_projector(target_geo.array.shape, angles, device=calc_device, **proj_kw)
            # columnas del proyector con paso pitch·n_r/n₀: cada rayo de la resina cae en una columna (sin
            # submuestreo en u, cuya traspuesta reparte el gradiente en peine y hacía divergir el CAL)
            u_pitch = voxel_pitch_mm * vial.n_resin / vial.n_outside
            vial_R = VialResampler(vial, angles, voxel_pitch_mm, target_geo.array.shape[0], u_pitch_mm=u_pitch,
                                   device=calc_device)
            VP = VialProjector(base, vial_R)
            stage_times["vial"] = time.perf_counter() - _tv
            # arranque: la FBP ideal corregida por el vial (ya es una imagen de proyector razonable)
            b_id = torch.clamp(filter_sinogram(base.forward(torch.as_tensor(target_geo.array, dtype=torch.float32,
                                                                            device=base.device)), filt), min=0)
            p0, _, _ = correct_sinogram(b_id.cpu().numpy(), angles, voxel_pitch_mm, vial, u_pitch_mm=u_pitch)
            try:
                cres = run_cal(target_geo.array, angles, copts, exit_param=exit_p, projector=VP, init=p0)
            finally:
                VP.close()
            print(f"[vial] dentro del modelo: el CAL optimizó la imagen del proyector ({vial_R.n_u} columnas)")
        else:
            cres = run_cal(target_geo.array, angles, copts, exit_param=exit_p, device=calc_device,
                           projector_kwargs=proj_kw)
        opt_result = (SimpleNamespace(array=cres.sinogram), SimpleNamespace(array=cres.reconstruction), cres.ver)
        stop_info = cres.stop
        pw_curve = cres.pw
        elapsed = time.perf_counter() - t_opt
        stop_info["engine"] = "vamcore"
        stop_info["projector"] = "prisma"
        stop_info["absorption_cm"] = absorption_cm
        stop_info["vial_mode"] = ("model" if vial_R is not None else "post") if vial is not None else None
        stop_info["ver_mask"] = stop_ver_mask
        if stop_info:
            print(
                f"[CAL] parada: {stop_info['reason']} tras {stop_info['iterations']}/{stop_info['max_iterations']} "
                f"iteraciones · mejor VER {stop_info['best_ver']:.5f} · {elapsed:.1f} s"
            )

        opt_sino, opt_recon, error_array = opt_result[0], opt_result[1], opt_result[2]
        if capture is not None:
            capture.update(target=np.asarray(target_geo.array), sinogram=np.asarray(opt_sino.array),
                           reconstruction=np.asarray(opt_recon.array), ver=np.asarray(error_array), d_h=d_h,
                           n_iter=n_iter_dynamic, learning_rate=lr_dynamic, filter=filt,
                           voxel_pitch_mm=voxel_pitch_mm, angles=angles)

        # ── 6. Exportar frames ────────────────────────────────────────────────
        t_exp = time.perf_counter()

        voxel_pitch_um = voxel_pitch_mm * 1000.0
        print(f"[export] alto Z={mesh_o.extents[2]:.2f}mm / {res} → vóxel={voxel_pitch_um:.1f}µm")

        frame_w = int(BUILD_D_MM * 1000 / pixel_um)
        frame_h = int(BUILD_H_MM * 1000 / pixel_um)
        sino_out = np.asarray(opt_sino.array)
        vial_info = None
        if vial is not None and vial_R is not None:
            # F6c: el resultado ya es la imagen del proyector; no se corrige después
            vial_info = dict(vial_R.info)
            frame_w = max(frame_w, int(np.ceil(vial.outer_diameter_mm * 1000 / pixel_um)))
        elif vial is not None:
            _t = time.perf_counter()
            sino_out, _, vial_info = correct_sinogram(sino_out, angles, voxel_pitch_mm, vial)
            vial_info["mode"] = "post"
            stage_times["vial"] = time.perf_counter() - _t
            frame_w = max(frame_w, int(np.ceil(vial.outer_diameter_mm * 1000 / pixel_um)))
            print(
                f"[vial] Ø{vial.outer_diameter_mm} mm, pared {vial.wall_thickness_mm} mm, "
                f"n_vial={vial.n_vial}, n_resina={vial.n_resin} → zona imprimible "
                f"Ø{2 * vial.printable_radius:.2f} mm · desviación máx. {vial_info['max_delta_deg']:.1f}°"
            )

        _t = time.perf_counter()
        crop_metadata = _export_frames_physical(
            sino_out, output_dir, voxel_pitch_um, frame_w, frame_h, pixel_um,
            col_pitch_um=(vial_R.u_pitch * 1000.0) if vial_R is not None else None,
        )
        stage_times["frames"] = time.perf_counter() - _t
        export_time = time.perf_counter() - t_exp
        stage_times["voxelize"] = vox_stats["time_s"]
        stage_times["optimize"] = elapsed
        _ev = np.where(~np.isnan(np.asarray(error_array, dtype=float)))[0]
        ver_curve = {"iter": _ev.tolist(), "ver": [float(error_array[i]) for i in _ev]}
        if pw_curve is not None:
            ver_curve["pw"] = [float(pw_curve[i]) for i in _ev]

        # ── 7. Métricas ───────────────────────────────────────────────────────
        # Banda de guarda: con ocupación parcial los vóxeles de borde (0,05 < occ < 0,95) no cuentan ni como gel
        # ni como vacío (su dosis ideal es intermedia). El CAL ya terminó: esto solo afecta a la evaluación.
        from vamcore.ops import metric_masks

        target_geo.gel_inds, target_geo.void_inds, metric_mask_info = metric_masks(target_geo.array)
        metrics = _compute_metrics(target_geo, opt_recon.array, error_array, elapsed)
        metrics["metric_masks"] = metric_mask_info

        # ── 7b. Tolerancia de exposición y prueba de robustez (vamcore.robustness) ──
        from vamcore.robustness import RobustnessSpec, exposure_tolerance, robustness_test

        _tol = exposure_tolerance(metrics.get("min_gel_dose", 0.0), metrics.get("max_void_dose", 1.0))
        metrics["exposure_tolerance_pct"] = 100 * _tol["tolerance"]
        robust = None
        if robustness:
            _t = time.perf_counter()
            spec = robustness if isinstance(robustness, RobustnessSpec) else RobustnessSpec()
            try:
                robust = robustness_test(np.asarray(opt_sino.array), target_geo.array, angles, voxel_pitch_mm,
                                         float(mesh_o.area), metrics["min_gel_dose"], metrics["max_void_dose"], spec,
                                         device=calc_device, projector_kwargs=proj_kw,
                                         image_to_resin=vial_R.apply if vial_R is not None else None,
                                         image_pitch_mm=vial_R.u_pitch if vial_R is not None else None)
                metrics["robust_pass_rate"] = robust["pass_rate"]
                metrics["shape_error_um_p90"] = robust["shape_error_um_p90"]
                print(f"[robustez] tolerancia de exposición ±{metrics['exposure_tolerance_pct']:.1f} % · "
                      f"{robust['n_pass']}/{robust['n_scenarios']} escenarios correctos · error de forma p90 "
                      f"{robust['shape_error_um_p90']:.1f} µm")
            except Exception as e:  # la prueba es informativa: nunca tumba el trabajo
                robust = {"error": f"{type(e).__name__}: {e}"}
                print(f"[robustez] no se pudo ejecutar: {robust['error']}")
            stage_times["robustness"] = time.perf_counter() - _t
        metrics["export_time"] = export_time
        # n_valid_iters cuenta EVALUACIONES del VER (una cada 5 iteraciones); estas son las iteraciones reales
        metrics["iterations_run"] = stop_info.get("iterations")
        metrics["stop_reason"] = stop_info.get("reason")
        metrics["total_time"] = time.perf_counter() - t_total

        layer_m = _compute_layer_metrics(target_geo, opt_recon.array)
        metrics.update({k: v for k, v in layer_m.items() if not k.startswith("_")})

        score, score_bio, verdict, penalties = _score(metrics, n_iter_dynamic)

        # Score Bio final = nota de dosis simulada × factor de fiabilidad (tolerancia de exposición y robustez)
        from vamcore.robustness import reliability_score

        score_breakdown = reliability_score(score_bio, metrics.get("exposure_tolerance_pct"),
                                            metrics.get("robust_pass_rate"))
        metrics["score_dose"] = score_bio
        score_bio = score_breakdown["score"]
        metrics["score_bio"] = score_bio
        verdict = _verdict(score_bio)
        if score_breakdown["tolerance_factor"] < 1:
            penalties.append(f"Tolerancia de exposición ±{metrics['exposure_tolerance_pct']:.1f} % "
                             f"(< ±{score_breakdown['tol_ref_pct']:.0f} %)")
        rf = score_breakdown["robustness_factor"]
        if rf is not None and rf < 1 and robust and "n_pass" in robust:
            penalties.append(f"Robustez {robust['n_pass']}/{robust['n_scenarios']} escenarios dentro de tolerancia")

        # ── 8. Visualizaciones ────────────────────────────────────────────────
        _save_slices(opt_recon.array, output_dir, run_name)
        _save_layer_profile(layer_m, output_dir, run_name)
        _save_histogram(opt_recon.array, target_geo, output_dir, run_name)

        # ── 9. CSV ────────────────────────────────────────────────────────────
        csv_path = output_dir / "metricas.csv"
        row = {
            "timestamp": timestamp,
            "model": stl_path.name,
            "run_name": run_name,
            "method": method,
            "resolution": res,
            "num_angles": n_angles,
            "cuda": use_cuda,
            "filter": filt,
            "n_iter": n_iter_dynamic,
            "d_h": _fmt(d_h),
            "d_l": _fmt(D_L),
            "rot_rx": _fmt(rx),
            "rot_ry": _fmt(ry),
            "rot_rz": _fmt(rz),
            "score": _fmt(score),
            "score_bio": _fmt(score_bio),
            "bio_multiplier": _fmt(metrics.get("bio_multiplier", 0)),
            "verdict": verdict,
            "penalties": " | ".join(penalties),
            "error": "",
        }
        for key in [
            "ver",
            "cv",
            "pw",
            "ipdr",
            "min_gel_dose",
            "max_gel_dose",
            "mean_gel_dose",
            "max_void_dose",
            "mean_void_dose",
            "dose_contrast",
            "final_error",
            "initial_error",
            "error_improvement",
            "n_valid_iters",
            "converged_iter",
            "n_active_layers",
            "cv_z_mean",
            "cv_z_std",
            "cv_z_p95",
            "cv_z_worst",
            "pw_z_mean",
            "pw_z_std",
            "pw_z_p5",
            "pw_z_worst",
            "pct_layers_overlap",
            "elapsed_time",
            "export_time",
            "total_time",
        ]:
            row[key] = _fmt(metrics.get(key, ""))

        _write_csv(row, csv_path)

        # ── 10. JSON de resultado (para la interfaz) ──────────────────────────
        summary = {
            "success": True,
            "timestamp": timestamp,
            "run_name": run_name,
            "model": stl_path.name,
            "output_dir": str(output_dir),
            "params": {
                "method": method,
                "resolution": res,
                "num_angles": n_angles,
                "filter": filt,
                "n_iter": n_iter_dynamic,
            "stall_patience": stall_p,
            "stall_rel_tol": stall_tol,
            "engine": stop_info.get("engine"),
            "projector": stop_info.get("projector"),
            "absorption_cm": stop_info.get("absorption_cm"),
            "vial_mode": stop_info.get("vial_mode"),
            "stop_ver_mask": stop_info.get("ver_mask"),
            "stop_pw_margin": stop_info.get("pw_margin"),
                "d_h": round(d_h, 4),
                "d_l": D_L,
            },
            "crop_metadata": crop_metadata,
            "stage_times": {k: round(v, 4) for k, v in stage_times.items()},
            "robustness": robust,
            "score_breakdown": score_breakdown,
            "ver_curve": ver_curve,
            "geometry": {
                "voxel_pitch_um": round(voxel_pitch_um, 3),
                "projector_pixel_um": pixel_um,
                "native_pitch_um": round(native_mm * 1000, 3),
                "part_radius_mm": round(part_radius_mm, 4),
            },
            "mesh": mesh_info,
            "voxelization": vox_stats,
            "vial": vial_info,
            "orient": result["orient"],
            "score": round(score, 2),
            "score_bio": round(score_bio, 2),
            "verdict": verdict,
            "penalties": penalties,
            "metrics": {
                k: (round(v, 6) if isinstance(v, float) and v == v else v)
                for k, v in metrics.items()
                if not k.startswith("_")
            },
            "files": {
                "frames_dir": str(output_dir),
                "metrics_csv": str(csv_path),
                "reconstruction_slices": str(output_dir / "reconstruction_slices.png"),
                "layer_profile": str(output_dir / "layer_profile.png"),
                "dose_histogram": str(output_dir / "dose_histogram.png"),
            },
        }
        with open(output_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        result.update(
            {
                "success": True,
                "metrics": metrics,
                "score": score,
                "score_bio": score_bio,
                "verdict": verdict,
                "penalties": penalties,
                "elapsed_time": elapsed,
                "export_time": export_time,
                "total_time": metrics["total_time"],
                "stage_times": stage_times,
                "ver_curve": ver_curve,
                "robustness": robust,
                "score_breakdown": score_breakdown,
                "vial": vial_info,
                "geometry": summary["geometry"],
                "mesh": mesh_info,
                "voxelization": vox_stats,
            }
        )

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()
        try:
            _write_csv(
                {k: "" for k in FIELDNAMES}
                | {
                    "timestamp": timestamp,
                    "model": stl_path.name,
                    "error": result["error"],
                },
                output_dir / "metricas.csv",
            )
        except Exception:
            pass

    return result

"""¿Cuánto error real aguanta una pila de imágenes? Tolerancia de exposición y prueba de robustez.

El Score Bio dice si, en una simulación ideal, la dosis separa la pieza del vacío. Aquí se estima cuánto puede
desviarse la máquina antes de que la pieza falle:

1. **Tolerancia de exposición.** Con un umbral de gelificación T de la resina, la pieza sale bien mientras toda la
   dosis del gel quede por encima de T y toda la del vacío por debajo. Si la dosis real es s·x (s = error de potencia
   del LED o de tiempo), hace falta T/g_min ≤ s < T/v_max. El margen relativo es r = g_min/v_max y, con T centrado
   (T = 2·g_min·v_max/(g_min + v_max)), la máquina tolera un error simétrico de ±ε con (1+ε)/(1−ε) = r, es decir
   ε = (r − 1)/(r + 1). Sin ventana (r ≤ 1), ε = 0.

2. **Prueba de robustez.** Se reconstruye la dosis de la pila final con perturbaciones realistas, todas con el mismo
   umbral fijo T (el centrado de la nominal), y se cuenta cuántos escenarios siguen formando la pieza:
   desplazamiento del eje de giro en la imagen, error de velocidad y fluctuación de ángulo de la plataforma,
   desenfoque del proyector, parpadeo del LED por fotograma y error global de dosis. Para cada escenario se da el
   error de forma equivalente en µm: (vóxeles que faltan + vóxeles de más) · volumen del vóxel / área de la pieza,
   es decir, el grosor medio de material que sobra o falta sobre la superficie. Un escenario es «correcto» si ese
   error no supera la tolerancia geométrica (``shape_tol_um``; por defecto medio vóxel).

**Límites (no es una garantía):** no modela la absorción de la resina (llegará con el proyector propio, F6), la
difusión ni la inhibición por oxígeno, ni la respuesta no lineal del material. Las magnitudes por defecto de las
perturbaciones son hipótesis razonables, no medidas de una máquina: hay que ajustarlas con impresiones reales.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from ._backend import torch


# ── 1. Tolerancia de exposición ───────────────────────────────────────────────
def exposure_tolerance(gel_min: float, void_max: float) -> dict:
    """Tolerancia simétrica de dosis (±fracción) y umbral relativo óptimo, a partir de la dosis normalizada."""
    if not (gel_min > 0) or not (void_max >= 0):
        return {"tolerance": 0.0, "ratio": float("nan"), "threshold_rel": float("nan")}
    if void_max == 0:
        return {"tolerance": 1.0, "ratio": float("inf"), "threshold_rel": gel_min / 2}
    r = gel_min / void_max
    eps = max(0.0, (r - 1) / (r + 1))
    # umbral centrado para un error lineal ±ε: g(1−ε) = T = v(1+ε) → T = 2gv/(g+v) (media armónica)
    thr = 2 * gel_min * void_max / (gel_min + void_max)
    return {"tolerance": eps, "ratio": r, "threshold_rel": thr}


# ── 2. Prueba de robustez ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class RobustnessSpec:
    """Magnitudes de las perturbaciones (hipótesis por defecto; ajustar a la máquina real)."""
    n_scenarios: int = 20
    seed: int = 0
    axis_offset_um: float = 20.0       # desplazamiento máx. del eje de giro en la imagen (uniforme ±)
    speed_err_pct: float = 0.5         # error de velocidad de giro, σ (%)
    angle_jitter_deg: float = 0.2      # fluctuación de ángulo por fotograma, σ (°)
    blur_um: float = 15.0              # desenfoque del proyector, σ máx. de la gaussiana (uniforme 0…)
    flicker_pct: float = 2.0           # parpadeo del LED por fotograma, σ (%)
    dose_err_pct: float = 5.0          # error global de dosis máx. (uniforme ±)
    shape_tol_um: float | None = None  # error de forma admisible (µm); None → medio vóxel (lo que la simulación resuelve)


def sample_scenarios(spec: RobustnessSpec, n_angles: int) -> list[dict]:
    rng = np.random.default_rng(spec.seed)
    out = []
    for k in range(spec.n_scenarios):
        out.append({
            "axis_offset_um": float(rng.uniform(-1, 1) * spec.axis_offset_um),
            "speed_err": float(rng.normal(0, spec.speed_err_pct / 100)),
            "angle_jitter_deg": rng.normal(0, spec.angle_jitter_deg, n_angles),
            "blur_um": float(rng.uniform(0, 1) * spec.blur_um),
            "flicker": rng.normal(0, spec.flicker_pct / 100, n_angles),
            "dose_err": float(rng.uniform(-1, 1) * spec.dose_err_pct / 100),
        })
    return out


def shift_u(b, d: float):
    """Desplaza el sinograma ``b[u, ángulo, z]`` d vóxeles a lo largo de u (interpolación lineal, ceros fuera)."""
    n = b.shape[0]
    src = torch.arange(n, device=b.device, dtype=torch.float64) - d
    i0 = torch.floor(src)
    w = (src - i0).to(b.dtype)[:, None, None]
    i0 = i0.long()
    i1 = i0 + 1
    v0 = ((i0 >= 0) & (i0 < n)).to(b.dtype)[:, None, None]
    v1 = ((i1 >= 0) & (i1 < n)).to(b.dtype)[:, None, None]
    return (1 - w) * v0 * b[i0.clamp(0, n - 1)] + w * v1 * b[i1.clamp(0, n - 1)]


def _gauss_kernel(sigma: float, dtype, device):
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum(), r


def blur_uz(b, sigma_vox: float):
    """Desenfoque gaussiano separable en u y z (plano de la imagen) de σ vóxeles."""
    if sigma_vox < 0.05:
        return b
    k, r = _gauss_kernel(sigma_vox, b.dtype, b.device)
    # Convolución separable como suma de copias desplazadas (bordes a cero). El núcleo es pequeño (r ≈ 3σ, pocos
    # vóxeles), así que son 2·(2r+1) operaciones elementales. No se usa conv1d: sin cuDNN es lenta (>1 s a 256) y
    # con cuDNN la DLL falla en algunos entornos Windows («Cannot load symbol cudnnGetVersion»), lo que mata el proceso.
    return _conv_axis(_conv_axis(b, k, r, 0), k, r, 2)


def _conv_axis(b, k, r, axis):
    pad = [0, 0] * b.ndim
    pad[2 * (b.ndim - 1 - axis)] = pad[2 * (b.ndim - 1 - axis) + 1] = r
    p = torch.nn.functional.pad(b, pad)
    n = b.shape[axis]
    out = torch.zeros_like(b)
    for j in range(2 * r + 1):
        out += k[j] * p.narrow(axis, j, n)
    return out


def robustness_test(sinogram, target, angles_deg, pitch_mm: float, area_mm2: float, gel_min: float,
                    void_max: float, spec: RobustnessSpec = RobustnessSpec(), device=None,
                    backend: str = "torch", projector_kwargs: dict | None = None, image_to_resin=None,
                    image_pitch_mm: float | None = None) -> dict:
    """image_to_resin: si el sinograma es la imagen del proyector (vial dentro del modelo, F6c), la función que la
    lleva a los rayos de la resina (``VialResampler.apply``). Las perturbaciones se aplican sobre la imagen."""
    """Prueba de robustez de la pila final ``sinogram[u, ángulo, z]`` (la ideal, sin corrección del vial)."""
    if torch is None:
        raise ImportError("robustness_test necesita PyTorch")
    from .ops import circle_mask
    from .projector import make_projector

    from ._backend import default_device

    device = default_device(device)
    tol = exposure_tolerance(gel_min, void_max)
    T = tol["threshold_rel"]
    t = np.asarray(target)
    m = np.broadcast_to(circle_mask(t.shape[0], t.shape[1])[:, :, None], t.shape)
    dev = torch.device(device)
    gel = torch.as_tensor((t >= 0.95) & m, device=dev)
    void = torch.as_tensor((t <= 0.05) & m, device=dev)
    b0 = torch.as_tensor(np.asarray(sinogram), dtype=torch.float32, device=dev)
    angles = np.asarray(angles_deg, dtype=float)
    vox_mm3 = pitch_mm ** 3

    pk = dict(projector_kwargs or {})
    with make_projector(t.shape, angles, backend=backend, device=device, **pk) as P:
        # proyector propio: sin autoajuste por escenario y solo la retroproyección (la prueba no proyecta)
        kw = {**pk, "z_chunk": P.z_chunk, "forward": False} if hasattr(P, "z_chunk") else pk
        to_resin = image_to_resin or (lambda v: v)
        k = float(P.adjoint(to_resin(b0)).max())  # normalización nominal: la misma para todos los escenarios

        def evaluate(x):
            printed = x >= T
            miss = int((gel & ~printed).sum())
            extra = int((void & printed).sum())
            err = (miss + extra) * vox_mm3 / area_mm2 * 1000 if area_mm2 > 0 else float("nan")
            return miss, extra, err

        tol_um = spec.shape_tol_um if spec.shape_tol_um is not None else 500 * pitch_mm
        nominal = evaluate(P.adjoint(to_resin(b0)) / k)

    scen = sample_scenarios(spec, len(angles))
    rows = []
    for s in scen:
        ip = image_pitch_mm or pitch_mm  # paso de las columnas de la imagen en el plano del proyector
        b = shift_u(b0, s["axis_offset_um"] / 1000 / ip)
        b = blur_uz(b, s["blur_um"] / 1000 / ip)
        b = b * torch.as_tensor(1 + s["flicker"], dtype=b.dtype, device=dev)[None, :, None]
        ang = angles * (1 + s["speed_err"]) + s["angle_jitter_deg"]
        with make_projector(t.shape, ang, backend=backend, device=device, **kw) as Pp:
            x = Pp.adjoint(to_resin(b)) / k * (1 + s["dose_err"])
        miss, extra, err_um = evaluate(x)
        ok = err_um <= tol_um
        rows.append({"pass": bool(ok), "missing": miss, "extra": extra, "shape_error_um": err_um,
                     "axis_offset_um": s["axis_offset_um"], "speed_err_pct": 100 * s["speed_err"],
                     "blur_um": s["blur_um"], "dose_err_pct": 100 * s["dose_err"]})

    errs = np.array([r["shape_error_um"] for r in rows])
    n_pass = sum(r["pass"] for r in rows)
    worst = max(rows, key=lambda r: r["shape_error_um"]) if rows else None
    return {
        "exposure_tolerance_pct": 100 * tol["tolerance"],
        "threshold_rel": T,
        "nominal_pass": nominal[2] <= tol_um, "nominal_shape_error_um": nominal[2], "shape_tol_um": tol_um,
        "n_scenarios": len(rows), "n_pass": int(n_pass), "pass_rate": n_pass / len(rows) if rows else float("nan"),
        "shape_error_um_mean": float(errs.mean()) if len(errs) else float("nan"),
        "shape_error_um_p90": float(np.percentile(errs, 90)) if len(errs) else float("nan"),
        "worst": worst, "scenarios": rows, "spec": asdict(spec),
        "not_modeled": (["difusión", "inhibición por oxígeno", "respuesta no lineal"]
                        + ([] if pk.get("alpha") else ["absorción de la resina"]) + [
                        "corrección del vial (se evalúa la pila ideal)"]),
    }


# ── 3. Nota combinada (Score Bio) ─────────────────────────────────────────────
TOL_REF_PCT = 5.0                  # tolerancia de exposición que cuenta entera (±5 %)
WEIGHTS = (0.6, 0.2, 0.2)          # dosis simulada, tolerancia de exposición, robustez


def reliability_score(dose_score: float, tol_pct: float | None, pass_rate: float | None,
                      tol_ref_pct: float = TOL_REF_PCT, weights=WEIGHTS) -> dict:
    """Score Bio = Q · (w_q + w_t·T + w_r·R), en 0–100.

    Q = nota de dosis simulada (separación gel/vacío en el modelo ideal, 0–100); T = min(1, tolerancia/tol_ref);
    R = fracción de escenarios de robustez correctos. Sin prueba de robustez (R desconocido) su peso se reparte en
    proporción entre los otros dos. Solo da 100 con dosis perfecta, tolerancia ≥ tol_ref y todos los escenarios bien.
    """
    wq, wt, wr = weights
    T = 0.0 if tol_pct is None or not np.isfinite(tol_pct) else min(1.0, max(0.0, tol_pct / tol_ref_pct))
    if pass_rate is None or not np.isfinite(pass_rate):
        s = wq + wt
        wq, wt, wr = wq / s, wt / s, 0.0
        R = None
    else:
        R = min(1.0, max(0.0, float(pass_rate)))
    factor = wq + wt * T + wr * (R or 0.0)
    score = max(0.0, min(100.0, float(dose_score) * factor))
    return {"score": score, "dose_score": float(dose_score), "tolerance_factor": T, "robustness_factor": R,
            "factor": factor, "weights": [wq, wt, wr], "tol_ref_pct": tol_ref_pct}

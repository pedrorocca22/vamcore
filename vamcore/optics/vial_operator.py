"""Refracción del vial dentro del modelo directo (docs/MIGRACION.md F6c, variante 6c-i).

Hasta ahora el CAL optimizaba el sinograma «ideal» b(φ, t) (rayos rectos y paralelos dentro de la resina) y después
``correct_sinogram`` deformaba las imágenes para el vial. Aquí el CAL optimiza **directamente la imagen p(θ, u) que
emite el proyector**:

* **G** (imagen del proyector → rayos en la resina). El píxel (θ, u) produce dentro de la resina la recta
  (φ = θ − δ(u), t = n₀·u/n_r) con transmisión T(u) (``VialOptics.ray_map``). Muestreando en la rejilla de la resina:
        b(φ, t) = p(θ = φ + δ(u), u) · T(u)/T₀,     u = n_r·t/n₀
  con interpolación bilineal en (θ, u) (θ periódico). Es la relación inversa de ``correct_sinogram``: G(C(b)) ≈ b.
  El jacobiano ∂(φ, t)/∂(θ, u) = n₀/n_r es constante: solo cambia la escala global, que el CAL normaliza.
* **Gᵀ**: su traspuesta exacta (matriz dispersa explícita).
* **VialProjector**: proyector compuesto con la interfaz ``Projector``: dosis = Aᵀ·(G·p) y gradiente = Gᵀ·(A·Δx).
  El CAL no cambia: solo recibe este proyector y su «sinograma» es la imagen del proyector.

Ventajas frente a la corrección posterior: positividad, nivel máximo y discretización se aplican sobre los píxeles
reales; no hace falta normalizar con T(u) al final; las columnas que no llegan a la resina (|u| ≥ u_max) no se usan
porque G no las ve; y las métricas de dosis ya incluyen el vial. G es igual para todas las capas z (el vial no desvía
en Z), así que es una matriz 2D pequeña.
"""
from __future__ import annotations

import warnings

import numpy as np

from .._backend import torch
from .vial import VialOptics, projector_u_grid


class VialResampler:
    def __init__(self, vial: VialOptics, angles_deg, voxel_pitch_mm: float, n_t: int, device=None,
                 u_pitch_mm: float | None = None, min_transmission: float = 0.05):
        if torch is None:
            raise ImportError("VialResampler necesita PyTorch")
        vial.validate()
        ang = np.asarray(angles_deg, dtype=np.float64)
        n_a = len(ang)
        d_ang = 360.0 / n_a
        if not np.allclose(np.diff(ang), d_ang, atol=1e-6):
            raise ValueError("La refracción del vial necesita ángulos equiespaciados que cubran 360°")
        from .._backend import default_device

        self.device = torch.device(default_device(device))
        self.vial, self.n_a, self.n_t = vial, n_a, int(n_t)
        self.u_pitch = float(u_pitch_mm or voxel_pitch_mm)
        self.u = projector_u_grid(vial, self.u_pitch)                 # columnas del proyector (mm)
        self.n_u = len(self.u)
        half_u = (self.n_u - 1) / 2

        # Rejilla de la resina: t_j = (j − (n_t − 1)/2)·pitch ; el píxel que la ilumina está en u = n_r·t/n₀
        t = (np.arange(self.n_t) - (self.n_t - 1) / 2) * voxel_pitch_mm
        u_src = vial.n_resin * t / vial.n_outside
        _, delta, T, valid = vial.ray_map(u_src)
        T0 = vial.ray_map(np.array([0.0]))[2][0]
        ok = valid & (T >= min_transmission)
        q_f = u_src / self.u_pitch + half_u                           # columna fraccionaria del proyector
        ok &= (q_f >= 0) & (q_f <= self.n_u - 1)
        shift = np.degrees(delta) / d_ang                             # θ = φ + δ → índice k + shift
        wT = np.where(ok, T / T0, 0.0)

        # 4 entradas por fila (k, j): bilineal en (θ, u)
        k = np.arange(n_a)[:, None]                                   # [a, 1]
        q0 = np.floor(q_f).astype(np.int64); wq = q_f - q0
        q1 = np.minimum(q0 + 1, self.n_u - 1); q0 = np.clip(q0, 0, self.n_u - 1)
        s0 = np.floor(shift).astype(np.int64); ws = shift - s0
        rows, cols, vals = [], [], []
        row = (k * self.n_t + np.arange(self.n_t)[None, :])           # [a, t]
        for (dk, wk) in ((s0, 1 - ws), (s0 + 1, ws)):
            kk = (k + dk[None, :]) % n_a                              # [a, t] (periódico en θ)
            for (qq, wu) in ((q0, 1 - wq), (q1, wq)):
                rows.append(row)
                cols.append(kk * self.n_u + qq[None, :])
                vals.append(np.broadcast_to(wk * wu * wT, row.shape))
        rows = np.concatenate([r.reshape(-1) for r in rows])
        cols = np.concatenate([c.reshape(-1) for c in cols])
        vals = np.concatenate([v.reshape(-1) for v in vals]).astype(np.float32)
        keep = vals != 0
        idx = torch.as_tensor(np.stack([rows[keep], cols[keep]]), device=self.device)
        v = torch.as_tensor(vals[keep], device=self.device)
        shape = (n_a * self.n_t, n_a * self.n_u)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            Gc = torch.sparse_coo_tensor(idx, v, shape).coalesce()
            self.G = Gc.to_sparse_csr()
            self.Gt = Gc.t().coalesce().to_sparse_csr()
        self.active_columns = int(np.unique(np.round(q_f[ok]).astype(int)).size)
        self.info = {
            "vial": vial.summary(), "u_pitch_mm": self.u_pitch, "n_columns": int(self.n_u),
            "max_delta_deg": float(np.degrees(np.abs(delta[ok]).max())) if ok.any() else 0.0,
            "min_transmission": float(T[ok].min()) if ok.any() else 0.0,
            "resin_rays_reached": int(ok.sum()), "resin_rays_total": int(self.n_t),
            "volume_radius_mm": float((self.n_t - 1) / 2 * voxel_pitch_mm), "mode": "model",
        }

    # p [u, θ, z] → b [t, φ, z]
    def apply(self, p):
        n_z = p.shape[2]
        P = p.permute(1, 0, 2).reshape(self.n_a * self.n_u, n_z)
        B = torch.sparse.mm(self.G, P.contiguous())
        return B.reshape(self.n_a, self.n_t, n_z).permute(1, 0, 2).contiguous()

    # b [t, φ, z] → p [u, θ, z]
    def adjoint(self, b):
        n_z = b.shape[2]
        Bf = b.permute(1, 0, 2).reshape(self.n_a * self.n_t, n_z)
        P = torch.sparse.mm(self.Gt, Bf.contiguous())
        return P.reshape(self.n_a, self.n_u, n_z).permute(1, 0, 2).contiguous()


class VialProjector:
    """Proyector compuesto: su «sinograma» es la imagen del proyector p[u, θ, z].
    forward(x) = Gᵀ·A·x (para el gradiente) ; adjoint(p) = Aᵀ·G·p (dosis)."""

    def __init__(self, base, resampler: VialResampler):
        if resampler.n_t != base.sino_shape[0] or resampler.n_a != base.sino_shape[1]:
            raise ValueError("El remuestreo del vial no coincide con la rejilla del proyector")
        self.base, self.R = base, resampler
        self.device = base.device
        self.vol_shape = base.vol_shape
        self.sino_shape = (resampler.n_u, base.sino_shape[1], base.sino_shape[2])

    def forward(self, x):
        return self.R.adjoint(self.base.forward(x))

    def adjoint(self, p):
        p = torch.as_tensor(p, dtype=torch.float32, device=self.device)
        return self.base.adjoint(self.R.apply(p))

    backward = adjoint

    def close(self):
        if hasattr(self.base, "close"):
            self.base.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

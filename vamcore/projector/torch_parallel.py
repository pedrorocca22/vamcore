"""Prisma — proyector de haz paralelo propio de VAMCORE en PyTorch (docs/MIGRACION.md F6).

Geometría: el rayo del ángulo φ va en la dirección r = (sin φ, −cos φ, 0) y el eje del detector es d = (cos φ, sin φ, 0)
(convención «parallel3d» heredada de ASTRA; de ella depende el signo de la corrección del vial). Rayos paralelos y
perpendiculares al eje Z, así que cada capa z es una transformada de Radon 2D independiente.

Directo (A), método de Joseph: según el eje dominante del rayo, se recorre el volumen fila a fila (o columna a
columna); en cada fila el rayo corta en un punto con coordenada entera en ese eje, se interpola linealmente en el otro
y se pondera con 1/|cos| (o 1/|sin|), la longitud del rayo entre filas:
               b[u, φ, z] = w(φ) · Σ_k  x( p_k(u, φ), z )

``SparseParallel3D`` (el que se usa) lo construye como matriz dispersa; ``TorchParallel3D`` es la versión de referencia
con ``grid_sample`` y adjunto por autograd. Ambos aplican la máscara del círculo inscrito (P·x en el directo, P·Aᵀ en
el adjunto) y no modifican su entrada.
"""
from __future__ import annotations

import math

import numpy as np

from .._backend import default_device, torch
from ..ops import circle_mask


def _sample(img, g):
    """grid_sample bilineal sin cuDNN: la ruta nativa de CUDA basta y la DLL de cuDNN falla en algunos entornos Windows
    («Cannot load symbol cudnnCreateTensorDescriptor»), lo que mata el proceso."""
    with torch.backends.cudnn.flags(enabled=False):
        return torch.nn.functional.grid_sample(img.expand(g.shape[0], -1, -1, -1), g, mode="bilinear",
                                               padding_mode="zeros", align_corners=True)


class TorchParallel3D:
    def __init__(self, vol_shape, angles_deg, device=None, method: str = "joseph", step: float = 1.0,
                 chunk: int | None = None, y_sign: float = 1.0):
        if torch is None:
            raise ImportError("TorchParallel3D necesita PyTorch")
        n_x, n_y, n_z = (int(v) for v in vol_shape)
        if n_x != n_y:
            raise ValueError("El volumen debe ser cuadrado en XY (eje de giro en el centro)")
        self.device = torch.device(default_device(device))
        self.angles_deg = np.asarray(angles_deg, dtype=float)
        n_a = len(self.angles_deg)
        self.vol_shape = (n_x, n_y, n_z)
        self.sino_shape = (n_x, n_a, n_z)
        self.step = float(step)
        self._mask = circle_mask(n_x, n_y, device=self.device)[:, :, None]
        # bloques de ángulos: ~1,5 GB de salida intermedia por bloque
        per_angle = n_z * int(math.ceil(n_x / min(self.step, 1.0))) * n_x * 4
        self.chunk = chunk or max(1, min(n_a, int(1.5e9 // max(per_angle, 1))))
        # rejilla de muestreo [n_a, n_s, n_u, 2] en coordenadas normalizadas de grid_sample (align_corners=True)
        c = (n_x - 1) / 2
        phi = torch.as_tensor(np.radians(self.angles_deg), dtype=torch.float64, device=self.device)
        cp, sp = torch.cos(phi)[:, None, None], torch.sin(phi)[:, None, None]
        u = (torch.arange(n_x, device=self.device, dtype=torch.float64) - c)[None, None, :]
        if method == "joseph":
            k = (torch.arange(n_x, device=self.device, dtype=torch.float64) - c)[None, :, None]  # filas/columnas
            by_rows = (cp.abs() >= sp.abs())                       # rayo r = (sin φ, −cos φ): domina y
            # filas: Y = k, s = (u·sinφ − k)/cosφ, X = u·cosφ + s·sinφ ; peso 1/|cosφ|
            cps = torch.where(by_rows, cp, torch.ones_like(cp))
            s_r = (u * sp - k) / cps
            X_r, Y_r = u * cp + s_r * sp, k.expand_as(s_r)
            # columnas: X = k, s = (k − u·cosφ)/sinφ, Y = u·sinφ − s·cosφ ; peso 1/|sinφ|
            sps = torch.where(by_rows, torch.ones_like(sp), sp)
            s_c = (k - u * cp) / sps
            X_c, Y_c = k.expand_as(s_c), u * sp - s_c * cp
            X = torch.where(by_rows, X_r, X_c)
            Y = torch.where(by_rows, Y_r, Y_c)
            w = torch.where(by_rows[:, 0, 0], 1 / cp[:, 0, 0].abs(), 1 / sp[:, 0, 0].abs())
        elif method == "sample":
            n_s = int(math.ceil(n_x / self.step))
            sv = ((torch.arange(n_s, device=self.device, dtype=torch.float64) - (n_s - 1) / 2) * self.step)[None, :, None]
            X, Y = u * cp + sv * sp, u * sp - sv * cp
            w = torch.full((len(self.angles_deg),), self.step, dtype=torch.float64, device=self.device)
        else:
            raise ValueError(f"method debe ser 'joseph' o 'sample', no {method!r}")
        self.method = method
        self._w = w.to(torch.float32)[:, None, None]                # [n_a, 1, 1]
        gx, gy = X / c, y_sign * Y / c                            # ix = X + c → normalizado (ix/c − 1)
        self._grid = torch.stack([gx, gy], dim=-1).to(torch.float32)  # x → W = ix, y → H = iy

    # ── núcleo lineal sin máscara: volumen [x, y, z] → sinograma [u, φ, z] ────
    def _fp(self, x):
        img = x.permute(2, 1, 0)[None]  # [1, C=z, H=iy, W=ix]
        out = []
        for a0 in range(0, self.sino_shape[1], self.chunk):
            g = self._grid[a0:a0 + self.chunk]
            smp = _sample(img, g)  # [a, z, s, u]
            out.append(smp.sum(dim=2) * self._w[a0:a0 + g.shape[0]])  # [a, z, u]
        return torch.cat(out, dim=0).permute(2, 0, 1)  # [u, a, z]

    def _as(self, a, shape, what):
        t = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        if tuple(t.shape) != shape:
            raise ValueError(f"{what}: forma {tuple(t.shape)}, se esperaba {shape}")
        return t

    def forward(self, x):
        x = self._as(x, self.vol_shape, "volumen")
        with torch.no_grad():
            return self._fp(x * self._mask).contiguous()

    def adjoint(self, b):
        b = self._as(b, self.sino_shape, "sinograma").detach()
        with torch.enable_grad():
            x0 = torch.zeros(self.vol_shape, dtype=torch.float32, device=self.device, requires_grad=True)
            img = x0.permute(2, 1, 0)[None]
            for a0 in range(0, self.sino_shape[1], self.chunk):
                g = self._grid[a0:a0 + self.chunk]
                smp = _sample(img, g)
                yb = b[:, a0:a0 + g.shape[0], :].permute(1, 2, 0)  # [a, z, u]
                (smp.sum(dim=2) * self._w[a0:a0 + g.shape[0]] * yb).sum().backward()
            return (x0.grad * self._mask).detach()

    backward = adjoint

    def close(self):
        self._grid = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def path_in_resin(X, Y, sin_phi, cos_phi, r_resin):
    """Distancia L que ha recorrido la luz dentro de la resina hasta el punto (X, Y) (vóxeles, origen en el eje).

    La luz viaja en la dirección del rayo r = (sin φ, −cos φ) y entra en la resina por la circunferencia
    de radio R (pared interior del vial o del recipiente). Resolviendo |p + t·r| = R, la entrada está en
    t = −p·r − √(R² − |p|² + (p·r)²), así que L = p·r + √(R² − |p|² + (p·r)²). Fuera de la resina, L = 0."""
    pr = X * sin_phi - Y * cos_phi
    disc = r_resin ** 2 - (X * X + Y * Y) + pr * pr
    return torch.where(disc > 0, pr + torch.sqrt(disc.clamp(min=0)), torch.zeros_like(disc)).clamp(min=0)


class SparseParallel3D:
    """Mismo operador que ``TorchParallel3D(method="joseph")``, como matriz dispersa 2D compartida por todas las capas.

    Con rayos perpendiculares a Z, cada capa es la misma transformada de Radon 2D: la matriz del sistema A₂D
    (filas = rayos (φ, u), columnas = vóxeles (ix, iy)) se construye una vez y el proyector es un producto disperso ×
    denso: b = A₂D · X con X = volumen como [n², n_z]. El adjunto usa A₂Dᵀ, construida explícitamente: es la
    traspuesta EXACTA (sin sumas atómicas, determinista). Joseph da 2 vóxeles por fila del volumen y rayo, así que
    cada fila de A₂D tiene 2n entradas (las que caen fuera del volumen llevan peso 0).

    Memoria: 2 × (n_a · n · 2n) × 8 B (valor + índice) ≈ 0,9 GB a 512 (n = 281, 360 ángulos).

    Velocidad: cuSPARSE rinde mucho mejor con bloques de capas z de tamaño intermedio (medido en RTX 4060 Ti: a 512,
    225 ms sin partir frente a 115 ms con bloques de 64). El tamaño óptimo depende de la rejilla y de la GPU, así que
    se autoajusta al crear el proyector (``z_chunk=None``) cronometrando unos pocos candidatos (< 1 s).

    Retroproyección (``backprojector``):
    - ``"voxel"`` (por defecto): por vóxel, como ASTRA. Se proyecta el centro de cada vóxel sobre el detector y se
      interpola linealmente entre los dos píxeles vecinos (peso 1 por ángulo). No es la traspuesta exacta de Joseph
      (igual que en ASTRA), pero no tiene sus artefactos: la traspuesta exacta de Joseph («por rayos») concentra dosis
      en algunos vóxeles (AᵀA·t con un máximo un 11 % mayor a 512 con la misma media) y, como el CAL normaliza por el
      máximo, a 512 la optimización no arrancaba (VER estancado en 0,20).
    - ``"exact"``: la traspuesta exacta (⟨Ax, y⟩ = ⟨x, Aᵀy⟩ a precisión de máquina), para gradientes y pruebas.

    Absorción (docs/MIGRACION.md F6b, Beer–Lambert): con ``alpha`` > 0 (por vóxel) cada entrada se multiplica por
    exp(−α·L), con L = distancia recorrida dentro de la resina de radio ``r_resin`` (vóxeles) hasta el punto de
    muestreo (proyección) o el centro del vóxel (retroproyección). Modelo directo y adjunto llevan la misma física.
    """

    Z_CANDIDATES = (32, 48, 64, 96, 128)

    def __init__(self, vol_shape, angles_deg, device=None, y_sign: float = 1.0, z_chunk: int | None = None,
                 backprojector: str = "voxel", forward: bool = True, alpha: float = 0.0, r_resin: float | None = None):
        """forward=False construye solo la retroproyección (p. ej. la prueba de robustez, que no proyecta).
        alpha: coeficiente de absorción por vóxel (α·tamaño del vóxel); r_resin: radio de la resina en vóxeles."""
        if alpha < 0:
            raise ValueError("alpha debe ser ≥ 0")
        if alpha > 0 and not (r_resin and r_resin > 0):
            raise ValueError("con absorción hace falta r_resin > 0 (radio de la resina en vóxeles)")
        self.alpha, self.r_resin = float(alpha), (float(r_resin) if r_resin else None)
        if torch is None:
            raise ImportError("SparseParallel3D necesita PyTorch")
        n_x, n_y, n_z = (int(v) for v in vol_shape)
        if n_x != n_y:
            raise ValueError("El volumen debe ser cuadrado en XY (eje de giro en el centro)")
        n = n_x
        self.device = torch.device(default_device(device))
        self.angles_deg = np.asarray(angles_deg, dtype=float)
        n_a = len(self.angles_deg)
        self.vol_shape = (n, n, n_z)
        self.sino_shape = (n, n_a, n_z)
        self._mask = circle_mask(n, n, device=self.device)[:, :, None]
        if y_sign < 0:
            raise NotImplementedError("solo y_sign = +1 (convención parallel3d)")
        self._A = self._joseph_matrix(n, n_a) if (forward or backprojector == "exact") else None
        if backprojector == "exact":
            self._At = self._A.to_sparse_coo().t().coalesce().to_sparse_csr()
        elif backprojector == "voxel":
            self._At = self._voxel_backprojector(n, n_a)
        else:
            raise ValueError(f"backprojector debe ser 'voxel' o 'exact', no {backprojector!r}")
        self.backprojector = backprojector
        self.z_chunk = int(z_chunk) if z_chunk else (self._autotune() if self._A is not None else n_z)

    def _joseph_matrix(self, n, n_a):
        """Matriz A₂D [n_a·n, n²] del método de Joseph (filas = a·n + u, columnas = ix·n + iy)."""
        c = (n - 1) / 2
        dev = self.device
        f64 = torch.float64
        k = torch.arange(n, device=dev, dtype=f64) - c                  # fila/columna recorrida (coordenada)
        u = torch.arange(n, device=dev, dtype=f64) - c
        cols_all, vals_all = [], []
        for a0 in range(0, n_a, 32):                                    # por bloques de ángulos (acota la memoria)
            phi = torch.as_tensor(np.radians(self.angles_deg[a0:a0 + 32]), dtype=f64, device=dev)
            cp, sp = torch.cos(phi)[:, None, None], torch.sin(phi)[:, None, None]
            by_rows = cp.abs() >= sp.abs()                              # [a, 1, 1]
            U, K = u[None, :, None], k[None, None, :]                  # [1, u, 1], [1, 1, k]
            # filas (Y = K): X = (U − K·sinφ)/cosφ ; columnas (X = K): Y = (U − K·cosφ)/sinφ
            cps = torch.where(by_rows, cp, torch.ones_like(cp))
            sps = torch.where(by_rows, torch.ones_like(sp), sp)
            frac_coord = torch.where(by_rows, (U - K * sp) / cps, (U - K * cp) / sps)  # [a, u, k]
            w = torch.where(by_rows, 1 / cp.abs(), 1 / sp.abs())                       # [a, 1, 1]
            f = frac_coord + c                                          # índice fraccionario en el eje no recorrido
            i0 = torch.floor(f)
            t = f - i0
            i0 = i0.long()
            fixed = (K + c).round().long().expand_as(i0)                # índice entero del eje recorrido
            if self.alpha > 0:
                Xs = torch.where(by_rows, frac_coord, K.expand_as(frac_coord))
                Ys = torch.where(by_rows, K.expand_as(frac_coord), frac_coord)
                w = w * torch.exp(-self.alpha * path_in_resin(Xs, Ys, sp, cp, self.r_resin))
            entries = []
            for di, wt in ((0, 1 - t), (1, t)):
                ii = i0 + di
                valid = (ii >= 0) & (ii < n)
                iic = ii.clamp(0, n - 1)
                # volumen aplanado como ix·n + iy ; por filas: iy = fijo, ix = fraccionario ; por columnas al revés
                col = torch.where(by_rows, iic * n + fixed, fixed * n + iic)
                val = torch.where(valid, wt * w, torch.zeros_like(wt))
                entries.append((col, val))
            col = torch.stack([entries[0][0], entries[1][0]], dim=-1).reshape(phi.numel() * n, 2 * n)
            val = torch.stack([entries[0][1], entries[1][1]], dim=-1).reshape(phi.numel() * n, 2 * n)
            cols_all.append(col.to(torch.int32))
            vals_all.append(val.to(torch.float32))
        cols = torch.cat(cols_all)                                      # [n_a·n, 2n], fila = a·n + u
        vals = torch.cat(vals_all)
        n_rows = n_a * n
        crow = torch.arange(0, n_rows * 2 * n + 1, 2 * n, device=dev, dtype=torch.int32)
        # columnas ordenadas por fila (algunos algoritmos de cuSPARSE lo requieren)
        order = torch.argsort(cols, dim=1)
        cols = torch.gather(cols, 1, order)
        vals = torch.gather(vals, 1, order)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # «Sparse CSR tensor support is in beta»
            A = torch.sparse_csr_tensor(crow, cols.reshape(-1), vals.reshape(-1), size=(n_rows, n * n))
        del cols, vals, order
        return A

    def _voxel_backprojector(self, n, n_a):
        """Matriz [n², n_a·n]: para cada vóxel (fila ix·n + iy) y ángulo, interpolación lineal en el detector en
        u = X·cosφ + Y·sinφ (+ centro). Columnas = a·n + u, como las filas de A."""
        import warnings

        dev, f64 = self.device, torch.float64
        c = (n - 1) / 2
        X = (torch.arange(n, device=dev, dtype=f64) - c)[:, None].expand(n, n).reshape(-1)  # ix
        Y = (torch.arange(n, device=dev, dtype=f64) - c)[None, :].expand(n, n).reshape(-1)  # iy
        phi = torch.as_tensor(np.radians(self.angles_deg), dtype=f64, device=dev)
        cols, vals = [], []
        for a0 in range(0, n_a, 32):
            ph = phi[a0:a0 + 32]
            uf = X[:, None] * torch.cos(ph)[None] + Y[:, None] * torch.sin(ph)[None] + c   # [n², a]
            att = (torch.exp(-self.alpha * path_in_resin(X[:, None], Y[:, None], torch.sin(ph)[None],
                                                          torch.cos(ph)[None], self.r_resin))
                   if self.alpha > 0 else None)
            i0 = torch.floor(uf)
            t = uf - i0
            i0 = i0.long()
            base = (torch.arange(a0, a0 + ph.numel(), device=dev) * n)[None]              # [1, a]
            ent_c, ent_v = [], []
            for di, wt in ((0, 1 - t), (1, t)):
                ii = i0 + di
                valid = (ii >= 0) & (ii < n)
                ent_c.append(base + ii.clamp(0, n - 1))
                ent_v.append(torch.where(valid, wt if att is None else wt * att, torch.zeros_like(wt)))
            cols.append(torch.stack(ent_c, -1).reshape(n * n, -1).to(torch.int32))      # [n², a·2]
            vals.append(torch.stack(ent_v, -1).reshape(n * n, -1).to(torch.float32))
        cols = torch.cat(cols, dim=1)
        vals = torch.cat(vals, dim=1)
        order = torch.argsort(cols, dim=1)
        cols, vals = torch.gather(cols, 1, order), torch.gather(vals, 1, order)
        k = cols.shape[1]
        crow = torch.arange(0, n * n * k + 1, k, device=dev, dtype=torch.int32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return torch.sparse_csr_tensor(crow, cols.reshape(-1), vals.reshape(-1), size=(n * n, n_a * n))

    def _autotune(self) -> int:
        import time

        n, _, n_z = self.vol_shape
        cands = sorted({min(c, n_z) for c in self.Z_CANDIDATES} | {n_z})
        if len(cands) == 1 or self.device.type != "cuda":
            return n_z
        X = torch.rand(n * n, n_z, device=self.device)
        best, best_t = n_z, float("inf")
        for zc in cands:
            self._mm(self._A, X, zc)  # calentamiento
            torch.cuda.synchronize()
            t = time.perf_counter()
            self._mm(self._A, X, zc)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            if dt < best_t:
                best, best_t = zc, dt
        del X
        return best

    @staticmethod
    def _mm(M, X, zc):
        n_z = X.shape[1]
        if zc >= n_z:
            return torch.sparse.mm(M, X)
        return torch.cat([torch.sparse.mm(M, X[:, i:i + zc]) for i in range(0, n_z, zc)], dim=1)

    def _as(self, a, shape, what):
        t = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        if tuple(t.shape) != shape:
            raise ValueError(f"{what}: forma {tuple(t.shape)}, se esperaba {shape}")
        return t

    def forward(self, x):
        if self._A is None:
            raise RuntimeError("proyector creado con forward=False: solo retroproyecta")
        x = self._as(x, self.vol_shape, "volumen")
        n, _, n_z = self.vol_shape
        X = (x * self._mask).reshape(n * n, n_z)
        B = self._mm(self._A, X, self.z_chunk)                          # [n_a·n, n_z], fila = a·n + u
        return B.reshape(self.sino_shape[1], n, n_z).permute(1, 0, 2).contiguous()

    def adjoint(self, b):
        b = self._as(b, self.sino_shape, "sinograma")
        n, _, n_z = self.vol_shape
        Y = b.permute(1, 0, 2).reshape(-1, n_z)
        X = self._mm(self._At, Y, self.z_chunk)
        return X.reshape(n, n, n_z) * self._mask

    backward = adjoint

    def close(self):
        self._A = self._At = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

"""Criterios de parada del CAL: umbral de VER, margen de ventana de proceso, estancamiento y tiempo máximo.

Modo VER (``pw_margin is None``, por defecto): semántica idéntica a ``vamtoolbox/optimizer/CAL.py`` (versión de
VAM Studio). Solo se decide en las iteraciones en que se evalúa el VER; primero el umbral (sobre el último VER) y
después el estancamiento (sobre el mejor VER visto, ``patience`` evaluaciones atrás). La evaluación inicial
(iteración 0) no entra en el historial.

Modo margen (``pw_margin`` = δ > 0): para cuando PW ≤ −δ, es decir, cuando hay una ventana de proceso de al menos δ
entre la dosis mínima del gel y la máxima del vacío. Con la banda de guarda el umbral de VER se alcanza casi al
instante (y separa gel y vacío por muy poco); exigir un margen conserva la robustez frente a variaciones de dosis.
El estancamiento se mide sobre el mejor PW: si no baja al menos ``stall_pw_tol`` en ``patience`` evaluaciones, para.
"""
from __future__ import annotations

import math
import time

from ..config import StopCriteria


class StopRule:
    def __init__(self, crit: StopCriteria, n_iter: int, exit_param: float | None = None):
        self.crit = crit
        self.n_iter = n_iter
        self.threshold = crit.ver_threshold if exit_param is None else exit_param
        self.pw_mode = crit.pw_margin is not None
        if self.pw_mode and not crit.pw_margin > 0:
            raise ValueError("pw_margin debe ser > 0")
        self.best_hist: list[float] = []     # mejor VER visto
        self.best_pw_hist: list[float] = []  # mejor (menor) PW visto
        self.last_ver = float("inf")
        self.last_pw = float("nan")
        self.reason = "max_iter"
        self._t0 = time.perf_counter()

    def update(self, it: int, ver: float, pw: float | None = None) -> bool:
        """Registra la evaluación de la iteración ``it`` (≥ 1). Devuelve True si hay que parar."""
        c = self.crit
        self.last_ver = ver
        self.best_hist.append(min(self.best_hist[-1], ver) if self.best_hist else ver)
        p = c.stall_patience
        if self.pw_mode:
            if pw is None:
                raise ValueError("el modo pw_margin necesita el PW de cada evaluación")
            self.last_pw = pw
            self.best_pw_hist.append(min(self.best_pw_hist[-1], pw) if self.best_pw_hist else pw)
            if pw <= -c.pw_margin:
                self.reason = "pw_margin"
                return True
            if p and it >= c.stall_min_iter and len(self.best_pw_hist) > p:
                if self.best_pw_hist[-1 - p] - self.best_pw_hist[-1] < c.stall_pw_tol:
                    self.reason = "stall"
                    return True
        else:
            if self.threshold is not None and ver <= self.threshold:
                self.reason = "threshold"
                return True
            if p and it >= c.stall_min_iter and len(self.best_hist) > p:
                ref = self.best_hist[-1 - p]
                if ref <= 0 or (ref - self.best_hist[-1]) / ref < c.stall_rel_tol:
                    self.reason = "stall"
                    return True
        if c.max_time_s is not None and time.perf_counter() - self._t0 >= c.max_time_s:
            self.reason = "max_time"
            return True
        return False

    def info(self, iterations: int, ver0: float) -> dict:
        out = {
            "reason": self.reason,
            "iterations": int(iterations),
            "max_iterations": int(self.n_iter),
            "best_ver": float(self.best_hist[-1]) if self.best_hist else float(ver0),
            "last_ver": float(self.last_ver) if self.last_ver != float("inf") else float(ver0),
        }
        if self.pw_mode:
            out["pw_margin"] = float(self.crit.pw_margin)
            out["best_pw"] = float(self.best_pw_hist[-1]) if self.best_pw_hist else math.nan
            out["last_pw"] = float(self.last_pw)
        return out

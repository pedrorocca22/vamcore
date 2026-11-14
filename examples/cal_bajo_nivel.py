"""Ejemplo 2 — los módulos por separado: voxelizar, elegir d_h, optimizar con CAL y medir.

    python examples/cal_bajo_nivel.py
"""
import numpy as np
import trimesh

from vamcore.config import CALOptions
from vamcore.dose import suggest_d_h
from vamcore.metrics import all_metrics
from vamcore.ops import metric_masks
from vamcore.optim.cal import run_cal
from vamcore.voxelize import voxelize_mesh

# 1. Malla centrada en el eje de giro (Z): un cilindro hueco de 6 mm × 8 mm
outer = trimesh.creation.cylinder(radius=3.0, height=8.0, sections=96)
inner = trimesh.creation.cylinder(radius=1.5, height=8.2, sections=96)
mesh = outer.difference(inner) if trimesh.boolean.engines_available else outer

# 2. Ocupación parcial por vóxel (0…1), indexada [ix, iy, iz]
target, grid, stats = voxelize_mesh(mesh, pitch_mm=0.125)
print("rejilla", target.shape, f"· error de volumen {stats.get('volume_error_pct', 0):+.2f} %")

# 3. Contraste de dosis según el grosor medio de la pieza
dh = suggest_d_h(target)
print(f"d_h = {dh.d_h} ({dh.tag})")

# 4. CAL con el proyector Prisma (GPU si hay; si no, CPU)
angles = np.linspace(0, 360, 180, endpoint=False)
opts = CALOptions.for_resolution(target.shape[2]).with_(d_h=dh.d_h)
res = run_cal(target, angles, opts)
print(f"parada: {res.stop['reason']} tras {res.stop['iterations']} iteraciones")

# 5. Métricas con banda de guarda (excluye los vóxeles parciales)
gel, void, info = metric_masks(target)
print({k: round(v, 5) for k, v in all_metrics(res.reconstruction, gel, void).items()}, info["rule"])

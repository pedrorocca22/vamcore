"""Ejemplo 1 — el pipeline completo: de un STL a la pila de imágenes del proyector.

    python examples/pipeline_basico.py            # GPU si hay; añade --cpu para forzar CPU
"""
import sys

from vamcore.orient import find_orientation
from vamcore.optics.vial import RESIN_PRESETS, VIAL_MATERIALS, VialOptics
from vamcore.pipeline import BUILD_H_MM, run
from vamcore.resources import example_stl

import trimesh

stl = example_stl()
vial = dict(outer_diameter_mm=20, wall_thickness_mm=1,
            n_vial=VIAL_MATERIALS["borosilicate"], n_resin=RESIN_PRESETS["gelma"])

# Orientación automática dentro de la zona imprimible del vial
o = find_orientation(trimesh.load(stl, force="mesh").vertices, max_height_mm=BUILD_H_MM,
                     max_radius_mm=VialOptics(**vial).printable_radius)
print(f"orientación: Rx {o.rx:.1f}°, Ry {o.ry:.1f}° (ancho −{o.reduction_pct:.0f} %)")

cap = {}
r = run(stl, "salida_ejemplo", rx=o.rx, ry=o.ry, resolution=128, num_angles=360, vial=vial,
        cuda="--cpu" not in sys.argv, capture=cap)
if not r["success"]:
    sys.exit(r["error"])
print(f"Score Bio {r['score_bio']:.0f}/100 · {r['verdict']} · VER {r['metrics']['ver']:.5f}")
print(f"sinograma {cap['sinogram'].shape} · reconstrucción {cap['reconstruction'].shape}")
print(f"fotogramas y result.json en {r['output_dir']}")

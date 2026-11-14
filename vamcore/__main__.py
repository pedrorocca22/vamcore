"""Línea de comandos de vamcore: de un STL a la pila de imágenes del proyector.

    python -m vamcore modelo.stl salida/ --res 256
    python -m vamcore modelo.stl salida/ --res 512 --vial 20 1 --resin gelma --absorption 0.5 --auto-orient
    python -m vamcore --example salida/            # vasculatura de ejemplo incluida en el paquete

Sustituye a la antigua CAL_run.py (sobre VAMToolbox). Imprime un resumen y deja en la carpeta de salida los
fotogramas, result.json, metricas.csv y las imágenes de diagnóstico.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vamcore", description=__doc__.splitlines()[0])
    ap.add_argument("stl", nargs="?", help="modelo STL (o --example)")
    ap.add_argument("output", nargs="?", help="carpeta de salida (por defecto, junto al STL)")
    ap.add_argument("--example", action="store_true", help="usa la vasculatura de ejemplo incluida en el paquete")
    ap.add_argument("--res", type=int, default=128, help="capas en Z (64, 128, 256, 512…); por defecto 128")
    ap.add_argument("--angles", type=int, default=360, help="número de proyecciones por vuelta (360)")
    ap.add_argument("--vial", nargs=2, type=float, metavar=("DIAM_MM", "PARED_MM"),
                    help="vial sin baño: diámetro exterior y grosor de pared (mm)")
    ap.add_argument("--vial-material", default="borosilicate", help="borosilicate, fused_silica, soda_lime, pmma, polystyrene")
    ap.add_argument("--resin", default="gelma", help="gelma, water o acrylate (índice de la resina)")
    ap.add_argument("--vial-mode", choices=["model", "post"], default="model", help="refracción dentro del modelo o corrección posterior")
    ap.add_argument("--absorption", type=float, default=0.0, help="absorción de la resina α (cm⁻¹); 0 = sin absorción")
    ap.add_argument("--stop", choices=["ver", "margin"], default="ver", help="parada por umbral de VER o por margen de PW")
    ap.add_argument("--margin", type=float, default=0.03, help="margen de ventana de proceso para --stop margin")
    ap.add_argument("--auto-orient", action="store_true", help="orienta la pieza antes de optimizar (vamcore.orient)")
    ap.add_argument("--rot", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("RX", "RY", "RZ"),
                    help="giro manual en grados (sistema del STL, Z = eje de giro)")
    ap.add_argument("--cpu", action="store_true", help="calcula en CPU aunque haya GPU")
    ap.add_argument("--no-robustness", action="store_true", help="omite la prueba de robustez")
    a = ap.parse_args(argv)

    from .resources import example_stl

    if a.example:
        if a.stl and not a.output:  # con --example, el primer posicional es la salida
            a.output, a.stl = a.stl, None
        a.stl = str(example_stl())
    if not a.stl:
        ap.error("falta el STL (o usa --example)")
    stl = Path(a.stl)
    if not stl.exists():
        ap.error(f"no existe {stl}")

    from .optics.vial import RESIN_PRESETS, VIAL_MATERIALS

    vial = None
    if a.vial:
        if a.vial_material not in VIAL_MATERIALS or a.resin not in RESIN_PRESETS:
            ap.error(f"material o resina desconocidos (materiales: {list(VIAL_MATERIALS)}; resinas: {list(RESIN_PRESETS)})")
        vial = dict(outer_diameter_mm=a.vial[0], wall_thickness_mm=a.vial[1],
                    n_vial=VIAL_MATERIALS[a.vial_material], n_resin=RESIN_PRESETS[a.resin])

    rx, ry, rz = a.rot
    if a.auto_orient:
        import trimesh

        from .orient import find_orientation
        from .pipeline import BUILD_D_MM, BUILD_H_MM
        from .optics.vial import VialOptics

        max_r = VialOptics(**vial).printable_radius if vial else BUILD_D_MM / 2
        o = find_orientation(trimesh.load(stl, force="mesh").vertices, max_height_mm=BUILD_H_MM, max_radius_mm=max_r)
        rx, ry, rz = o.rx, o.ry, 0.0
        print(f"[orientación] Rx {rx:.1f}° · Ry {ry:.1f}° · ancho −{o.reduction_pct:.0f} %"
              + ("" if o.feasible else " · ⚠ no cabe en el volumen disponible"))

    from .pipeline import run

    r = run(stl, a.output, rx=rx, ry=ry, rz=rz, resolution=a.res, num_angles=a.angles, cuda=not a.cpu, vial=vial,
            vial_mode=a.vial_mode, absorption_cm=a.absorption,
            stop_ver_mask="guard_band" if a.stop == "margin" else "half",
            stop_pw_margin=a.margin if a.stop == "margin" else None, robustness=not a.no_robustness)
    if not r["success"]:
        print(f"ERROR: {r['error']}", file=sys.stderr)
        return 1
    m, rb = r["metrics"], r.get("robustness") or {}
    print(f"\nScore Bio {r['score_bio']:.0f}/100 · {r['verdict']} (simulado; no garantiza la impresión)")
    print(f"  VER {m['ver']:.5f} · PW {m['pw']:+.4f} · tolerancia de exposición ±{m.get('exposure_tolerance_pct', 0):.1f} %"
          + (f" · robustez {rb['n_pass']}/{rb['n_scenarios']}" if "n_pass" in rb else ""))
    print(f"  {m.get('iterations_run')} iteraciones ({m.get('stop_reason')}) · optimización {r['elapsed_time']:.1f} s · "
          f"total {r['total_time']:.1f} s")
    print(f"  salida: {r['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

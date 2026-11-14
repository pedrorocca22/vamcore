# Vídeo promocional VAMCORE · guion v3

*14-11-2026 · sigue `software-showcase-video/SKILL.md` (estilo APTUM). Sustituye al guion v2 de VAM Studio.*

## Decisiones

- **Marca:** «VAMCORE» en la portada y en el pie de cada capítulo; cierre con el clip APTUM completo
  (`software-showcase-video/assets/aptum/aptum_logo_animation.mp4`, 6 s).
- **Formato:** 1920×1080, 24 fps, H.264, sin audio, textos en inglés. Duración ≈ 54 s con el cierre.
- **Un solo trabajo real** guía todo el vídeo: la vasculatura de ejemplo, inclinada 35° como entrada, orientada con
  «Auto», a 512³ con 360 ángulos, vial de borosilicato Ø20 mm con pared de 1 mm, resina GelMA y vial dentro del
  modelo. Todas las cifras salen de esa ejecución (`promo/data/`), salvo la histórica del capítulo 02, que se
  rotula como tal.
- **Sin nombres de terceros** en pantalla.

## Capítulos

| # | Etiqueta | Titular | Lo que cambia en el visual | Cifra | Rótulo de procedencia |
|---|---|---|---|---|---|
| — | Portada | **VAMCORE** · *From STL to projection, physically correct* | Completa desde el primer fotograma | — | BETA |
| 01 | ORIENT | *Finds the narrowest pose* | La pieza inclinada gira hasta la orientación que calcula `find_orientation` | −% de ancho | input tilted 35° for the demo |
| 02 | VOXELIZE | *Layer by layer, with partial occupancy* | Un plano baja por la pieza; al lado, la sección real de esa capa con sus bordes parciales | tiempo de voxelización a 512³ | previous version: historical measurement, same PC |
| 03 | VIAL | *No index-matching bath* | Rayos paralelos que se doblan al cruzar el vial (`ray_map`), zona imprimible | Ø imprimible | computed optics · physical validation pending |
| 04 | PRISMA | *Optimizes the image the projector really emits* | Corte de dosis tras 0, 5, 15, 40, 80 iteraciones y el final, sincronizado con la curva de VER | tiempo de optimización | simulated dose · Prisma engine |
| 05 | VERIFY | *Checks the print before you print* | Histograma de dosis gel/vacío con la ventana; los 20 escenarios de robustez se marcan uno a uno | Score Bio | simulated · not a print test |
| 06 | STACK | *360 frames, ready for the projector* | La pila real de fotogramas pasa, con el ángulo | tiempo total STL → pila | — |
| 07 | PRINTER | *Plays back in sync with the rotation* | Prueba de luz real del prototipo | — | prototype light test · real footage · no part formed |
| — | Cierre | Clip APTUM completo | — | — | — |

## Qué no se dice

- Nada de resultados de impresión: el vídeo no muestra piezas impresas.
- Nada de «validated», «clinical», «biocompatible».
- El Score Bio es calidad de dosis simulada.

## Producción

```bash
python promo/compute.py            # GPU: datos reales en promo/data/ (≈ 5 min)
python promo/render.py --preview   # 4 fps + hoja de contactos en promo/out/
python promo/render.py             # final 24 fps → promo/out/vamcore_promo.mp4
```

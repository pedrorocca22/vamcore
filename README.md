<div align="center">
  <img src="assets/images/vam_studio_hero.PNG" alt="Interfaz de VAM Studio" width="800"/>
  <br/>
  <i>Software completo para impresión volumétrica tomográfica (VAM / CAL)</i>
</div>
<br/>

# VAM Studio

VAM Studio cubre todo el proceso de la **impresión volumétrica tomográfica** (*Volumetric Additive Manufacturing*,
*Computed Axial Lithography*). Empieza con un modelo STL, genera la pila de imágenes que proyecta la impresora mientras
gira el vial y termina con la proyección sincronizada con el motor.

Lo forman cuatro piezas:

| Pieza | Carpeta | Dónde corre |
|---|---|---|
| **vamcore**, el motor de cálculo | [`vamcore/`](vamcore/) | PC (GPU NVIDIA o CPU) |
| API local (FastAPI) | [`server.py`](server.py) | PC |
| Interfaz web (React + Three.js) | [`frontend/`](frontend/) | navegador del PC |
| Software del projector | [`device/`](device/) | controlador del projector (Raspberry Pi + Klipper) |

El motor es propio. Escrito en PyTorch, incluye su proyector tomográfico (**Prisma**) y modela el vial y la resina
dentro de la optimización. No depende de ASTRA Toolbox ni de VAMToolbox.

> **Estado: beta.** Los resultados del motor son simulados. El Score Bio mide la calidad de las imágenes generadas y
> **no garantiza** el éxito de la impresión. Las perturbaciones de la prueba de robustez son hipótesis por defecto
> hasta calibrarlas con la máquina real.

**Contenido:** [Funciones](#funciones) · [Instalación](#instalación) · [Uso de la aplicación](#uso-de-la-aplicación) ·
[Línea de comandos](#línea-de-comandos) · [API de Python](#api-de-python) · [Referencia del motor](#referencia-del-motor) ·
[API HTTP](#api-http-serverpy) · [Rendimiento medido](#rendimiento-medido) · [Pruebas](#pruebas) ·
[Estructura](#estructura-del-repositorio) · [Licencia](#licencia-y-atribución)

---

## Funciones

- **Interfaz web:** visor 3D del modelo y del volumen de trabajo, orientación manual o automática («Auto»), panel
  de vial y resina, parámetros de optimización, resultados con Score Bio, fiabilidad estimada y dosis simulada, y
  una biblioteca de trabajos con imágenes de diagnóstico.
- **Motor vamcore:**
  - Prisma, proyector de haz paralelo en GPU o CPU;
  - refracción del vial dentro del modelo (impresión sin baño de *index matching*);
  - absorción de la resina (Beer–Lambert);
  - voxelizador por secciones con ocupación parcial;
  - contraste de dosis automático según el grosor de la pieza;
  - CAL con parada por umbral de error o por margen de ventana de proceso.
- **Evaluación:** métricas de dosis con banda de guarda, tolerancia de exposición, prueba de robustez con 20
  escenarios y **Score Bio /100**.
- **Salida:** fotogramas a la escala física del proyector (uno por ángulo), `result.json`, CSV e imágenes de
  diagnóstico.
- **Envío al projector:** la API empaqueta la pila y la manda por HTTP. El projector la reproduce sincronizada con el
  giro del vial (motor paso a paso a través de Moonraker/Klipper).

## Instalación

Requisitos: Python ≥ 3.10, Node.js ≥ 20 (interfaz) y, opcionalmente, una GPU NVIDIA con CUDA. Sin GPU todo funciona
en CPU, más lento.

### Con pip

```bash
# 1. PyTorch: con CUDA si hay GPU NVIDIA (si no, basta con `pip install torch`)
pip install torch --index-url https://download.pytorch.org/whl/cu124
# 2. Motor + API
pip install -e ".[app]"
```

### Con conda

```bash
conda env create -f environment.yml
conda activate vamstudio
pip install -e ".[app]"
```

### Interfaz

```bash
cd frontend
npm ci
```

## Uso de la aplicación

```bash
# Terminal 1 — API local en http://127.0.0.1:8000
python server.py
# Terminal 2 — interfaz en http://localhost:5173
cd frontend && npm run dev
```

1. **Modelo:** arrastra un `.stl` (en mm). Oriéntalo a mano o pulsa **Auto**.
2. **Vial y resina:** diámetro, pared y material del vial, resina (índice de refracción) y absorción α.
3. **Optimización:** resolución (64–512 capas), número de ángulos y criterio de parada (umbral VER o margen PW).
   En *Avanzado* están el filtro FBP y la elección GPU/CPU.
4. **Procesar:** el panel derecho muestra el Score Bio, la fiabilidad estimada (tolerancia de exposición,
   escenarios de robustez, error de forma), la dosis simulada y los tiempos de cada etapa.
5. **Guardar en biblioteca**, revisar los diagnósticos y **enviar al projector**.

La API solo escucha en `127.0.0.1` y acepta peticiones de la interfaz en cualquier puerto local (CORS). Guarda los
trabajos en `jobs/` (biblioteca) y `tmp_jobs/` (temporales, se vacía al arrancar).

**Compilar la interfaz** para producción: `cd frontend && npm run build` (deja la versión estática en
`frontend/dist/`).

### Projector

El software del projector y su instalación están en [`device/README.md`](device/README.md). En resumen, con
`pip install -e ".[device]"` y un `config.json` copiado de [`config.example.json`](config.example.json), se ejecuta:

```bash
python upload_to_projector.py      # sube el software del projector por SSH
python start_projector_server.py   # arranca su API
```

## Línea de comandos

El motor funciona sin la interfaz:

```bash
# Vasculatura de ejemplo incluida, vial de 20 mm con pared de 1 mm, orientación automática
python -m vamcore --example salida/ --res 128 --vial 20 1 --auto-orient
```

```text
Score Bio 93/100 · EXCELENTE (simulado; no garantiza la impresión)
  VER 0.00000 · PW -0.0546 · tolerancia de exposición ±3.3 % · robustez 20/20
  250 iteraciones (max_iter) · optimización 1.8 s · total 12.2 s
  salida: salida/
```

(Medido en una RTX 4060 Ti; en CPU la optimización tarda bastante más.)

```text
python -m vamcore [STL] [SALIDA] [opciones]        (o el comando `vamcore` tras instalar)
```

| Opción | Por defecto | Qué hace |
|---|---|---|
| `STL` | — | Modelo a imprimir (mm; Z = eje de giro). |
| `SALIDA` | junto al STL | Carpeta de resultados. |
| `--example` | — | Usa la vasculatura de ejemplo incluida (`vamcore.resources.example_stl()`). |
| `--res N` | 128 | Capas en Z (64, 128, 256, 512…). Fija el tamaño de vóxel: alto / N. |
| `--angles N` | 360 | Proyecciones por vuelta. |
| `--vial DIAM PARED` | sin vial | Vial sin baño: diámetro exterior y grosor de pared (mm). |
| `--vial-material` | `borosilicate` | `borosilicate`, `fused_silica`, `soda_lime`, `pmma`, `polystyrene`. |
| `--resin` | `gelma` | `gelma`, `water`, `acrylate` (índice de refracción de la resina). |
| `--vial-mode` | `model` | `model`: refracción dentro de la optimización; `post`: corrección posterior. |
| `--absorption α` | 0 | Absorción de la resina en cm⁻¹ (Beer–Lambert). |
| `--stop` | `ver` | `ver`: umbral de error volumétrico; `margin`: margen de ventana de proceso (más rápido). |
| `--margin δ` | 0.03 | Margen de ventana de proceso para `--stop margin` (dosis normalizada a 1). |
| `--auto-orient` | — | Orienta la pieza para minimizar su ancho alrededor del eje, dentro de los límites. |
| `--rot RX RY RZ` | 0 0 0 | Giro manual en grados (sistema del STL). |
| `--cpu` | — | Fuerza CPU aunque haya GPU. |
| `--no-robustness` | — | Omite la prueba de robustez (más rápido; el Score Bio reparte su peso). |

## API de Python

### Pipeline completo: `vamcore.pipeline.run`

```python
from vamcore.pipeline import run
from vamcore.optics.vial import VIAL_MATERIALS, RESIN_PRESETS

r = run("pieza.stl", "salida/", resolution=256, num_angles=360,
        vial=dict(outer_diameter_mm=20, wall_thickness_mm=1,
                  n_vial=VIAL_MATERIALS["borosilicate"], n_resin=RESIN_PRESETS["gelma"]),
        absorption_cm=0.0)
if r["success"]:
    print(r["score_bio"], r["verdict"], r["metrics"]["ver"])
```

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `stl_path`, `output_dir` | —, junto al STL | Entrada y carpeta de salida. |
| `rx`, `ry`, `rz` | 0 | Giro en grados, R = Rz·Ry·Rx, en el sistema del STL (Z = eje de giro). |
| `resolution` | 128 | Capas en Z. Iteraciones y *learning rate* siguen `CALOptions.for_resolution`. |
| `num_angles` | 360 | Proyecciones por vuelta. |
| `filter_type` | `"cosine"` | Filtro de la inicialización FBP: `ram-lak`, `shepp-logan`, `cosine`, `hamming`, `hanning`. |
| `cuda` | `True` | `False` fuerza CPU; sin CUDA disponible se usa CPU de todos modos. |
| `iterations`, `learning_rate` | según resolución | Sobrescriben la escalera de `CALOptions.for_resolution`. |
| `exit_param` | 0.0005 | Umbral de VER para parar. |
| `stall_patience`, `stall_rel_tol` | 10, 0.01 | Parada por estancamiento (0 la desactiva). |
| `stop_ver_mask` | `"half"` | Máscaras del VER de parada: `half` (al 50 %) o `guard_band` (sin vóxeles parciales). |
| `stop_pw_margin` | `None` | δ > 0: para cuando la ventana de proceso PW ≤ −δ (usar con `guard_band`). |
| `vial` | `None` | `dict` o `VialOptics`: `outer_diameter_mm`, `wall_thickness_mm`, `n_vial`, `n_resin`, `n_outside`. |
| `vial_mode` | `"model"` | `model` (refracción dentro del modelo) o `post` (corrección posterior). |
| `absorption_cm` | 0 | α de la resina (cm⁻¹). La luz entra por la pared interior del vial. |
| `projector_pixel_um` | 14.3 | Tamaño del píxel del proyector en el plano del vial. |
| `voxel_pitch_um` | alto / `resolution` | Tamaño de vóxel explícito. |
| `robustness` | `True` | `True`, `False` o un `RobustnessSpec` con otras perturbaciones. |
| `capture` | `None` | `dict` que se rellena con los arrays intermedios (ver abajo). |

**Devuelve** un `dict` con estas claves:
- `success` y `error`;
- `score_bio` (0–100) y `verdict`: `EXCELENTE` ≥ 85, `BUENO` ≥ 70, `ACEPTABLE` ≥ 55, `POBRE` ≥ 40, `FALLIDO` por debajo;
- `score` (nota de dosis) y `penalties`;
- `metrics`: VER, PW, IPDR, CV, contraste, métricas por capa, tolerancia de exposición, iteraciones, razón de parada…;
- `robustness`, `stage_times` (segundos por etapa) y `ver_curve`;
- `elapsed_time` (optimización), `total_time` y `output_dir`.

`result.json` guarda lo mismo, más la geometría, la malla, la voxelización y el vial.

**`capture`** recibe:
- `target` (ocupación), `sinogram` (imagen del proyector `[u, ángulo, z]`) y `reconstruction` (dosis simulada);
- `ver` (curva) y `d_h`;
- `n_iter`, `learning_rate`, `filter`, `voxel_pitch_mm` y `angles`.

**Archivos de salida:**
- `frame_NNNN.png`: uno por ángulo, a escala física del proyector y recortados a la caja útil;
- `result.json`, `metricas.csv` y `model_oriented.stl`;
- `reconstruction_slices.png`, `layer_profile.png` y `dose_histogram.png`.

Las constantes de la máquina están en `vamcore.pipeline`:
- `PROJECTOR_PIXEL_UM = 14.3`;
- `BUILD_D_MM = 20` y `BUILD_H_MM = 25`, es decir, un volumen de trabajo de Ø20 × 25 mm;
- `NUM_ANGLES`, `RESOLUTION` y `EXIT_PARAM`.

### Módulos por separado

```python
import numpy as np
from vamcore.voxelize import voxelize_mesh
from vamcore.dose import suggest_d_h
from vamcore.config import CALOptions
from vamcore.optim.cal import run_cal
from vamcore.ops import metric_masks
from vamcore.metrics import all_metrics

target, grid, stats = voxelize_mesh(mesh, pitch_mm=0.125)      # ocupación 0…1, [ix, iy, iz]
opts = CALOptions.for_resolution(target.shape[2]).with_(d_h=suggest_d_h(target).d_h)
res = run_cal(target, np.linspace(0, 360, 360, endpoint=False), opts)
gel, void, info = metric_masks(target)                         # banda de guarda
print(res.stop, all_metrics(res.reconstruction, gel, void))
```

Ejemplos completos y ejecutables en [`examples/`](examples/): `pipeline_basico.py` y `cal_bajo_nivel.py`.

## Referencia del motor

### Flujo

```text
STL ─► orientar (orient) ─► voxelizar por secciones (voxelize) ─► d_h por grosor medio (dose)
    ─► CAL con Prisma (optim.cal + projector) ── vial dentro del modelo (optics) · absorción (projector)
    ─► métricas con banda de guarda (ops, metrics) ─► tolerancia de exposición y robustez (robustness)
    ─► Score Bio ─► fotogramas a escala física + result.json (pipeline)
```

### Módulos

| Módulo | Contenido principal |
|---|---|
| `vamcore.pipeline` | `run(...)`: el proceso completo; constantes de la máquina; exportación de fotogramas. |
| `vamcore.__main__` | Línea de comandos (`python -m vamcore`, comando `vamcore`). |
| `vamcore.config` | `CALOptions` (iteraciones, *learning rate*, momento, sigmoide, filtro, `d_h`, cada cuánto se evalúa el VER), `StopCriteria`; `ProjectorSpec`, `VoxelGrid`, `PrintJob`. Todo inmutable y sin estado global. |
| `vamcore.optim.cal` | `run_cal(target, angles_deg, opts, ...)` → `CALResult(sinogram, reconstruction, ver, pw, stop, times)`. |
| `vamcore.optim.stopping` | `StopRule`: parada por umbral de VER, por margen de ventana de proceso y por estancamiento. |
| `vamcore.projector` | `make_projector(vol_shape, angles_deg, backend="torch", device=None, **kw)`; protocolo `Projector` (`forward`, `adjoint`, `close`; usable con `with`). |
| `vamcore.projector.torch_parallel` | `SparseParallel3D` (**Prisma**): `alpha`/`r_resin` para la absorción, `z_chunk` autoajustado, `backprojector="voxel"` o `"exact"`. `TorchParallel3D`: variante con `grid_sample`, de referencia. |
| `vamcore.optics.vial` | `VialOptics` (geometría, `printable_radius`, `ray_map`), `VIAL_MATERIALS`, `RESIN_PRESETS`, `correct_sinogram` (modo `post`), `check_fit`. |
| `vamcore.optics.vial_operator` | `VialResampler` (imagen del proyector → rayos en la resina, con adjunto) y `VialProjector` (Prisma compuesto con el vial). |
| `vamcore.voxelize` | `voxelize_mesh(mesh, pitch_mm, ...)` → `(occupancy, grid, stats)`; `native_pitch_mm`, `mesh_report`. |
| `vamcore.dose` | `suggest_d_h(occupancy)` → `DhSuggestion(d_h, mean_thickness_vox, tag)` según el grosor medio (EDT). |
| `vamcore.metrics` | `ver`, `pw`, `ipdr`, `cv` (también por capa con `layerwise=True`), `all_metrics`. |
| `vamcore.ops` | `metric_masks` (banda de guarda), `get_inds`, `circle_mask`, `clip_to_circle`, `sigmoid`, `discretize`. |
| `vamcore.filters` | `filter_sinogram(sino, name)`, `fourier_filter`, `FILTERS`. |
| `vamcore.robustness` | `exposure_tolerance`, `RobustnessSpec`, `robustness_test`, `reliability_score`. |
| `vamcore.orient` | `find_orientation(vertices, max_height_mm, max_radius_mm, ...)` → `OrientResult`. |
| `vamcore.resources` | `example_stl()`: la vasculatura trifurcada de ejemplo. |

### Modelos físicos y numéricos

- **Prisma.**
  - Método de Joseph como matriz dispersa 2D, compartida por todas las capas Z porque los rayos son
    perpendiculares al eje.
  - Producto disperso × denso en cuSPARSE, con bloques de capas autoajustados.
  - Retroproyección por vóxel: la traspuesta exacta creaba vóxeles calientes y hacía divergir el CAL a 512.
- **CAL.** Parte de una inicialización FBP y desciende con momento sobre el sinograma. Usa la sigmoide de respuesta
  de la resina centrada en `d_h`, un contraste que se elige según el grosor medio de la pieza.
- **Absorción.** Beer–Lambert con la longitud recorrida en la resina desde la pared interior del vial:
  L = p·r + √(R² − |p|² + (p·r)²).
- **Vial dentro del modelo.**
  - El CAL optimiza directamente la imagen que emite el proyector, compuesta con la refracción y la transmisión de
    Fresnel en las dos interfaces del vial.
  - El paso de columnas es pitch·n_r/n₀, para que el adjunto no reparta el gradiente en peine.
  - Con dosis física da entre 2,5 y 6 veces más tolerancia de exposición que la corrección posterior.
- **Métricas con banda de guarda.**
  - Los vóxeles con ocupación entre 0,05 y 0,95 están medio dentro de la pieza: no cuentan ni como gel ni como vacío.
  - VER es la fracción de vóxeles mal clasificados.
  - PW = máx(dosis vacío) − mín(dosis gel). Si es negativo, hay ventana de proceso.
- **Tolerancia de exposición.** ε = (r − 1)/(r + 1), con r = mín(gel)/máx(vacío) y el umbral en la media armónica.
  Indica cuánto puede variar la dosis global sin que la pieza cambie.
- **Robustez.**
  - 20 escenarios perturbados: desplazamiento del eje, error de velocidad, fluctuación de ángulo, desenfoque,
    parpadeo del LED y error de dosis.
  - Se evalúan con el mismo proyector y se cuenta cuántos conservan la forma.
- **Score Bio** = Q · (0,6 + 0,2·T + 0,2·R), acotado a 0–100:
  - Q es la nota de dosis simulada;
  - T = mín(1, tolerancia/±5 %);
  - R es la fracción de escenarios correctos.
- **Parada.** Tres criterios:
  - umbral de VER, el criterio por defecto;
  - margen de ventana de proceso (`stop_pw_margin`), mucho más rápido a cambio de menos tolerancia;
  - estancamiento.
- **Auto-orientación.**
  - Minimiza el ancho medio proyectado en 72 direcciones (rejilla de 10° y Nelder-Mead).
  - Descarta las orientaciones que superan la altura o el radio imprimibles.
  - Es solo geométrica.

### Convenciones

- **Ejes:** volumen `[ix, iy, iz]` con el eje de giro en el centro de (ix, iy) y z creciente con iz; sinograma
  `[u, ángulo, z]`.
- **Unidades:** milímetros, salvo donde el nombre indica otra cosa (`_um`, `_cm`, `_deg`). Ángulos en grados.
- **Arrays:** los módulos aceptan NumPy y, con torch instalado, `torch.Tensor` (se opera en su dispositivo).
  Importar `vamcore` no carga torch.

## API HTTP (`server.py`)

| Método y ruta | Uso |
|---|---|
| `GET /health` | Estado de la API. |
| `GET /vial/presets` · `GET /vial/preview` | Materiales y resinas; zona imprimible de un vial. |
| `POST /orient` | Auto-orientación de un STL (con los límites del vial o del área de trabajo). |
| `POST /optimize` | Lanza un trabajo (STL + parámetros del formulario) → `job_id`. |
| `GET /status/{job_id}` | Progreso y resultado. |
| `GET /library` · `POST /save/{job_id}` · `DELETE /library/{job_id}` | Biblioteca de trabajos. |
| `GET /export/{job_id}.json` | Resultado completo. |
| `GET /job/{job_id}/frame/{i}` · `GET /job/{job_id}/artifact/{name}` · `GET /file/{job_id}/{path}` | Fotogramas e imágenes de diagnóstico (rutas validadas contra *path traversal*). |
| `GET /printer_status` · `POST /print/{job_id}` · `GET /print_send_status/{job_id}` | Estado del projector y envío de la pila. |

`POST /optimize` acepta, entre otros:
- `num_angles`, `resolution`, `filter`, `cuda` y `rx`/`ry`/`rz`;
- `stop_ver_mask` y `stop_pw_margin`;
- `absorption_cm`;
- `vial_outer_diameter_mm`, `vial_wall_thickness_mm`, `vial_material` o `n_vial`, `resin` o `n_resin`, y `vial_mode`.

## Rendimiento medido

En una RTX 4060 Ti 16 GB, con la vasculatura de ejemplo y 360 ángulos (solo la optimización):

| Resolución | Parada por VER | Parada por margen δ = 0,03 |
|---|---|---|
| 256 | 8,0 s | — |
| 512 | 37–40 s | 7,9 s |

Un cubo a 128³ con 360 ángulos, lanzado desde la interfaz, tarda 0,5 s en optimizarse. El proceso completo, con
robustez y 360 fotogramas, tarda 8,3 s.

El detalle de las mediciones y la comparación con el motor anterior están en
[`docs/MIGRACION.md`](docs/MIGRACION.md).

## Pruebas

```bash
pip install -e ".[app,test]"
python -m pytest
```

Hay 97 pruebas, del motor y de la API.

**Referencias doradas:**
- las de `tests/golden/*_64.npz`, generadas con el motor anterior, comprueban que el CAL reproduce sus resultados;
- las de 128 (`tests/golden/large/`, fuera de git) solo se usan si están presentes y hay GPU.

Las pruebas que necesitan torch se saltan solas si no está instalado. La CI ([`.github/workflows/tests.yml`](.github/workflows/tests.yml))
corre las pruebas sin GPU y compila la interfaz.

## Estructura del repositorio

```text
vamcore/            motor (paquete de Python)
server.py           API local que usa la interfaz
frontend/           interfaz web (Vite + React + Three.js)
device/             software del projector (proyección, motor, red) y configuración de Klipper
upload_to_projector.py · start_projector_server.py   instalación del software del projector por SSH
examples/           ejemplos de uso del motor
tests/              pruebas y referencias doradas
docs/MIGRACION.md   registro de diseño y validación del motor
assets/             imágenes del README
```

## Licencia y atribución

GPL-3.0-or-later ([`LICENSE`](LICENSE)).

Partes de `ops`, `metrics`, `filters`, `dose` y `optim.cal` están adaptadas de
[VAMToolbox](https://github.com/computed-axial-lithography/VAMToolbox) (GPL-3.0, Regents of the University of
California / Taylor lab). Cada módulo lo indica en su cabecera.

El proyecto se inspira en la investigación y el código abierto de
[Computed Axial Lithography (CAL)](https://github.com/computed-axial-lithography). La visión, la conceptualización y
el diseño del flujo de trabajo de VAM Studio son del autor. El código se implementó íntegramente con asistencia de IA.

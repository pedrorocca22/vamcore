> **Documento histórico.** Registro de diseño y validación de la migración de VAM Studio desde VAMToolbox y ASTRA
> hasta este motor propio (fases F0–F7 y limpieza final). Las rutas que menciona (`cal_core.py`, `server.py`,
> `AUDITORIA.md`, `promocional/…`, el voxelizador «legacy») pertenecen al repositorio de la aplicación VAM Studio o
> ya no existen; las cifras son las medidas en su momento. La referencia vigente del motor está en el `README.md`.

# Plan de migración: de VAMToolbox a un motor propio

*13-11-2026 · Documento de diseño. Todavía no se ha migrado nada: este plan se ejecuta por fases y cada una se valida
antes de pasar a la siguiente. Complementa a [AUDITORIA.md](AUDITORIA.md).*

## 1. Objetivo

Sustituir la dependencia de VAMToolbox por un paquete propio (nombre provisional: **`vamcore`**). El paquete estará
diseñado para lo que hace VAM Studio: generar una pila de imágenes físicamente correcta a partir de un STL, con
unidades físicas explícitas y todo el cálculo en la GPU. Esto nos da libertad para cambiar el modelo físico
(refracción, absorción, óptica) sin heredar decisiones de una librería genérica.

Hay dos niveles, y no hay que confundirlos:

| Nivel | Qué cambia | Esfuerzo | Riesgo |
|---|---|---|---|
| **1. Motor propio + ASTRA** | Contenedores, métricas, filtros, CAL y pipeline pasan a `vamcore`. ASTRA sigue calculando las proyecciones, detrás de una interfaz. | Bajo–medio | Bajo |
| **2. Proyector propio** | Las proyecciones las calcula nuestro propio código en PyTorch, con la refracción del vial y la absorción dentro del modelo. | Medio–alto | Medio (exactitud y velocidad) |

El nivel 1 es el que da la libertad de diseño. El nivel 2 es la gran mejora física del producto.

**Destino final (decidido 13-11-2026):** los dos niveles. Ni VAMToolbox ni ASTRA serán dependencias de ejecución:
VAM Studio será un sistema empaquetado propio, con control total de lo que pasa dentro (F7). ASTRA se conserva solo
como referencia para validar en los tests.

## 2. Inventario: qué usamos hoy de VAMToolbox

Ruta activa de `cal_core.run` (método CAL en la GPU):

```
TargetGeometry(target=occ)            geometry.py      contenedor + suggest_d_h (EDT) + getInds
ProjectionGeometry("parallel", CUDA)  geometry.py      contenedor de ángulos
optimize.Options / optimize.optimize  optimize.py      opciones + despacho
  └ minimizeCAL                       optimizer/CAL.py bucle (ya reescrito por nosotros: VRAM, parada)
      ├ projectorconstructor          projectorconstructor.py  selecciona el proyector
      │   └ Projector3DParallelCUDA   projector/       ASTRA FP3D/BP3D vía DLPack  ← el cálculo pesado
      ├ util.data.filterSinogram      util/data.py     filtro rampa (Hann/Hamming…)
      ├ util.data.sigmoid / discretize
      └ metrics.calcVER               metrics.py
metrics.calcVER / calcPW / calcIPDR   metrics.py       informe final
voxelize.voxelizeTarget               voxelize.py      solo con voxelizer="legacy" (bench «versión anterior»)
```

**Ya es nuestro:** `slicevox.py` (voxelizador), `vial.py` (corrección del vial), la mayor parte de `CAL.py`, y en
`cal_core.py` la exportación física, la orientación y el informe.

**No se usa en la ruta activa:** OSMO, FBP como método, `display*`, `imagesequence`, `response`, `medium`,
proyectores algebraicos, TIGRE, `astra3Dabsorption` y `pyTorchRayTrace`. De unas 10 800 líneas, la ruta activa toca
menos de 2 000, y de esas una parte importante ya la hemos escrito nosotros.

**Corrección a lo que dije en el chat:** VAMToolbox **sí** trae dos cosas relacionadas con el nivel 2, aunque no las
usamos:
- **Absorción:** una máscara aproximada, `exp(−α·z)` sobre la distancia radial. El propio código dice que *«this is
  not mathematically valid»* (`geometry.py`, ~l. 194).
- **Trazador de rayos curvos** (`pyTorchRayTrace.py`, GPL, de Meta): sirve para medios de índice variable. Es general
  y costoso. En el nivel 2 nos sirve como **referencia independiente** con la que comparar.

## 3. Licencia (D1 decidida 13-11-2026: **se sigue con GPL-3.0**)

> Decisión: `vamcore` se distribuye bajo GPL-3.0. Se puede reutilizar y adaptar código de VAMToolbox con atribución
> (cabecera de origen en cada módulo adaptado y mención en el README). Lo que sigue queda como contexto.

VAMToolbox es **GPL-3.0** (`setup.py`), y este repositorio es un fork, así que el conjunto se distribuye bajo GPL.

- **Si el producto sigue siendo GPL:** se puede migrar con libertad, reutilizando y adaptando código con atribución.
  Es lo más rápido.
- **Si se quiere un producto cerrado:** no basta con reescribir. Hay que reimplementar desde las publicaciones (CAL:
  Kelly et al., *Science* 2019; las métricas son definiciones publicadas), no a partir del código GPL. Habría que
  documentar el origen de cada módulo y **consultar con un especialista en propiedad intelectual** antes de
  distribuir. Los archivos escritos por nosotros desde cero (`vial.py`, `slicevox.py`) son nuestros, pero hoy viven
  dentro de un fork GPL.

Esto no es asesoramiento legal. La decisión condiciona cómo se escriben las fases 1–3, así que conviene tomarla antes
de empezar.

## 4. Arquitectura destino

```
vamcore/
  config.py        dataclasses: PrintJob, VoxelGrid, ProjectorSpec, VialOptics, CALOptions (sin estado global)
  mesh.py          carga, centrado, orientación, mesh_report
  voxelize.py      ← slicevox.py (ocupación parcial, unidades en mm)
  orient.py        auto-orientación geométrica
  optics/
    vial.py        ← vial.py (ray_map, u_max, transmisión de Fresnel)
    absorption.py  (nivel 2) Beer–Lambert a lo largo del rayo
  projector/
    base.py        Protocol Projector: forward(x) -> b, adjoint(b) -> x, ambos con torch.Tensor en el dispositivo
    astra.py       backend ASTRA (DLPack, sin copias), adaptado de Projector3DParallelCUDA
    torch_parallel.py    (nivel 2) haz paralelo propio
    torch_refractive.py  (nivel 2) rayos por píxel con la refracción del vial
  filters.py       filtros rampa con FFT de PyTorch
  optim/
    cal.py         bucle CAL solo en PyTorch
    stopping.py    criterios: umbral, estancamiento, tiempo máximo
  metrics.py       VER, PW, IPDR, CV en PyTorch (los mismos números que hoy)
  dose.py          d_h sugerido (EDT), histogramas, Score Bio
  export.py        fotogramas con escala física, recorte, PNG/bitplanes
  pipeline.py      STL → pila de imágenes; cal_core.run pasa a ser un envoltorio fino
```

**Principios:**
1. **Una sola convención de ejes**, documentada y probada: volumen `[ix, iy, iz]`, sinograma `[u, ángulo, z]`.
   Hoy hay permutaciones repartidas entre ASTRA, el CAL y la exportación.
2. **Unidades físicas explícitas** (mm, µm) en todas las interfaces. Nada de «resolución» sin unidades.
3. **Tensores de PyTorch en la GPU** de principio a fin. NumPy solo en la entrada y la salida.
4. **El proyector es intercambiable.** El CAL no sabe si debajo está ASTRA o nuestro código.
5. **Sin efectos colaterales:** nada de imprimir dentro del cálculo (se usa `logging`) ni de modificar las opciones
   desde dentro (hoy `options.stop_info` se escribe sobre el objeto que se recibe; pasará a devolverse en el
   resultado).

## 5. Fases

Cada fase termina con **un criterio de aceptación medible**. VAMToolbox se queda como **referencia** hasta la fase 5.

### F0 — Referencias «doradas» (4070)
- Ejecutar el pipeline actual con 4 modelos (trifurcatedvasculature, cubo, esfera, cilindro hueco) a 64 y 128.
  Guardar por cada uno: objetivo, sinograma antes del vial, reconstrucción, curva VER, métricas y `stop_info` en
  `tests/golden/*.npz`. Los de 64 se versionan; los mayores se dejan fuera de git.
- Añadir una **prueba del adjunto** del proyector: ⟨Ax, y⟩ = ⟨x, Aᵀy⟩ con tolerancia relativa de 1e-4.
- **Criterio:** referencias guardadas y la prueba del adjunto pasa con ASTRA.
- **✅ Hecho 13-11-2026 (RTX 4060 Ti):** `tests/golden/make_golden.py` → 4 × `*_64.npz` versionados (≈7,7 MB) y
  4 × `large/*_128.npz` fuera de git; `tests/test_golden_files.py` valida su estructura (corre en el CI).
  `tests/test_projector_adjoint.py` pasa; error relativo medido 7,5e-6 a 8,8e-5 (poco margen en volúmenes
  pequeños: FP y BP de ASTRA 3D no son un par exactamente emparejado).

### F1 — Esqueleto y módulos puros (se puede hacer en este PC)
- Crear `vamcore/` con `config.py`, `metrics.py`, `filters.py`, la sigmoide, la discretización y `dose.suggest_d_h`.
- Mover `slicevox.py` y `vial.py`, dejando *shims* de importación en `vamtoolbox/` para no romper nada.
- **Criterio:** con entradas aleatorias pequeñas, en CPU, los resultados coinciden con VAMToolbox a 1e-6 (métricas,
  filtros) o exactamente (índices, d_h). Los 34 tests actuales siguen pasando.
- **✅ Hecho 13-11-2026:** `vamcore/{config,ops,metrics,filters,dose}.py`; `vamtoolbox/slicevox.py` →
  `vamcore/voxelize.py` y `vamtoolbox/vial.py` → `vamcore/optics/vial.py`, con *shims* (`sys.modules`) en el sitio
  antiguo. `cal_core.py` y `server.py` importan ya de `vamcore`. Paridad: `tests/test_vamcore.py` (exacta en índices,
  d_h, discretización y círculo; ≤ 1e-6 en filtros y métricas). Las métricas por capa se calculan de verdad por capa
  (VAMToolbox fallaba o repetía el valor global). `import vamcore` no carga torch ni ASTRA.

### F2 — Interfaz de proyector y backend ASTRA (4070)
- `projector/base.py` y `projector/astra.py`, que absorbe `Projector3DParallelCUDA` y la ruta DLPack.
- **Criterio:** la salida coincide con el proyector antiguo a 1e-5, pasan la prueba del adjunto y
  `test_astra_convention.py`, y no hay pérdida de velocidad (±5 %).
- **✅ Hecho 13-11-2026:** `vamcore/projector/{base,astra}.py` (`make_projector`). Buffers y algoritmos de ASTRA
  creados una vez; no modifica la entrada. `tests/test_projector_astra.py`: diferencia 0 con el proyector antiguo,
  adjunto correcto, sin arrastre entre llamadas. Velocidad a 512: FP +0,1 %, adjunto +1,9 % (a 256: −8 % y 0 %).
  Aún no lo usa el CAL (eso es F3).

### F3 — CAL en `vamcore` (4070)
- `optim/cal.py` y `optim/stopping.py` sobre la interfaz `Projector`, solo en PyTorch. Se elimina la rama NumPy salvo
  un modo CPU para tests.
- **Criterio:** con las referencias de F0, el VER final queda dentro de ±0,5 % relativo, la parada cae en la misma
  evaluación ±1 y el tiempo es igual o mejor. Las retroproyecciones de ASTRA usan sumas atómicas y no son
  deterministas, así que no se puede exigir igualdad exacta.

- **✅ Hecho 13-11-2026:** `vamcore/optim/{cal,stopping}.py` (`run_cal`, `StopRule`), todo en GPU sobre
  `make_projector`. `tests/test_vamcore_cal.py`: con las 8 referencias doradas (64 y 128), misma razón de parada y
  mismas iteraciones; reconstrucción y sinograma con diferencia L2 < 5e-5; mejor VER idéntico salvo `sphere_64`
  (0,000589 frente a 0,000607, unos pocos vóxeles). Hallazgo: VAMToolbox recortaba el círculo con dos máscaras
  distintas (torch float32 en el bucle, NumPy float64 en la reconstrucción final) y discrepaban en los vóxeles justo
  sobre la circunferencia (hasta un 3 % L2 en la reconstrucción); `vamcore` usa una sola, la de NumPy.
  Velocidad (vasculatura, optimización): 256 → 12,5 s a 10,6 s; 512 → 44,2 s a 37,4 s (**−15 %**), con métricas
  idénticas (`promocional/bench_engine_*.json`).
- **Banda de guarda en la parada** (`StopCriteria.ver_mask="guard_band"`): con el umbral 0,0005 para demasiado pronto
  (512: 16 it., PW +0,031, Score Bio 91,1). Con umbral 0 para en cuanto gel y vacío se separan (512: 21 it., 5,4 s,
  Score Bio 100) pero con una ventana de proceso estrecha (PW −0,005 frente a −0,049 con la regla actual). Resuelto
  con una parada por **margen de ventana de proceso** (`StopCriteria.pw_margin` = δ): con banda de guarda, para cuando
  PW ≤ −δ; el estancamiento se mide sobre el PW (mejora < 0,001 en 10 evaluaciones). Medido (vasculatura, Score Bio
  100 en todos; `promocional/bench_pw_margin_*.json`):

  | δ | 256: it. · opt. · PW | 512: it. · opt. · PW |
  |---|---|---|
  | 0,02 | 16 · 1,0 s · −0,035 | 26 · 6,9 s · −0,022 |
  | 0,03 | 16 · 0,7 s · −0,035 | 31 · 7,9 s · −0,031 |
  | 0,04 | 21 · 0,8 s · −0,040 | 56 · 14,2 s · −0,041 |
  | 0,05 | 86 · 3,2 s · −0,050 | 196 · 48,7 s · −0,050 |
  | regla actual (VER al 50 %) | 291 · 10,6 s · −0,057 | 156 · 37,4 s · −0,049 |

  A 512, la regla actual equivale a δ ≈ 0,05; δ = 0,04 es 2,6× más rápido y δ = 0,03, 4,7×. En la interfaz:
  «vamcore + banda» con campo «Margen PW» (0,03 por defecto). Qué margen hace falta lo decide la resina real.

### F4 — Pipeline y conmutación (4070)
- `pipeline.py`. `cal_core.run(engine="vamtoolbox" | "vamcore")`; la API y la interfaz no cambian.
- **En marcha 13-11-2026:** `cal_core.run(engine="vamtoolbox"|"vamcore", stop_ver_mask="half"|"guard_band")`, mismos
  campos en `POST /optimize` y selector «Motor» en la interfaz (VAMToolbox / vamcore / vamcore + banda). Por defecto,
  VAMToolbox hasta que el usuario lo valide. `result.json` → `params.engine`, `params.stop_ver_mask`.
- **Criterio:** comparar los `result.json` de los dos motores con los 4 modelos a 256 (métricas ±0,5 %, mismas
  dimensiones de imagen y recorte). Después, `vamcore` pasa a ser el motor por defecto.

- **✅ Hecho 13-11-2026:** `promocional/bench_f4.py` con los 4 modelos a 256 (`promocional/bench_f4_256.json`):
  mismas iteraciones y razón de parada, métricas con diferencia máxima del 0,001 %, mismo recorte, mismo tamaño de
  imagen y mismo número de fotogramas. vamcore más rápido en todos (optimización: vasculatura 12,2 → 10,1 s; cubo
  5,5 → 1,4 s; esfera 36,1 → 34,1 s; cilindro hueco 4,3 → 2,6 s). **vamcore es el motor por defecto** en `cal_core`,
  en `POST /optimize` y en la interfaz («VAMToolbox (referencia)» sigue disponible).

### F5 — Retirar la dependencia
- `requirements` sin VAMToolbox. El código antiguo **se congela** en `legacy/` (o en un extra opcional), porque sigue
  haciendo falta como «versión anterior» en `bench_cal.py` y en el capítulo 09 del vídeo.
- **Criterio:** instalación limpia en la 4070 sin VAMToolbox; los tests y los benchmarks pasan.
- **✅ Hecho 13-11-2026:** `cal_core` ya no importa `vamtoolbox` al cargarse. El camino por defecto (vamcore +
  voxelizador por secciones) usa un contenedor propio del objetivo (`_Target`: recorte al círculo y máscaras
  idénticos), `vamcore.dose` para d_h y `vamcore.metrics` para las métricas. VAMToolbox se carga bajo demanda solo
  para `engine="vamtoolbox"` o `voxelizer="legacy"`, con un error claro si no está. Se optó por el **extra opcional**
  en lugar de mover el código a `legacy/` (lo usan las pruebas de paridad, las referencias y los benchmarks).
  `requirements.txt` nuevo: dependencias de ejecución sin VAMToolbox ni las suyas (pyvista, scikit-image, numba…).
  **Verificado** con `vamtoolbox` bloqueado (`sys.modules["vamtoolbox"] = None`): la ejecución a 256 funciona, no
  se carga ningún módulo de vamtoolbox y métricas, iteraciones, d_h y recorte son idénticos a los de antes del
  cambio. No se ha probado una instalación desde cero en un entorno nuevo.

### F6 — Proyector propio y física (nivel 2, 4070)

**6a · Haz paralelo propio (sustituye a ASTRA; D3).** Es obligatoria, no opcional: el objetivo es retirar ASTRA
(ver F7). Se hace en dos escalones:
1. **PyTorch puro:** rotación más integración por muestreo trilineal (`grid_sample`) o método de Joseph, con el
   adjunto exacto (el adjunto discreto real, no una retroproyección aproximada). Se procesa por bloques de ángulos y
   de capas z para acotar la VRAM.
2. **Kernel CUDA propio,** solo si el escalón 1 no alcanza la velocidad. Se escribe en CUDA C++ como extensión de
   PyTorch (`torch.utils.cpp_extension`), usando memoria de texturas para la interpolación, igual que ASTRA. Se
   distribuye **precompilado** dentro del paquete para que el usuario no necesite compilador. Triton es otra opción,
   pero su soporte en Windows es limitado.
- **Criterio de exactitud:** diferencia con ASTRA < 1 % (norma L2) y adjunto a 1e-4.
- **Criterio de velocidad para ser el motor por defecto:** ≤ 1,2× el tiempo de ASTRA en el bucle CAL completo a 256
  y 512 en la 4070. Mientras no lo cumpla, ASTRA sigue siendo el motor por defecto, pero el trabajo continúa hasta
  cumplirlo.
- Ventajas: se acaba la instalación de ASTRA en Windows, se pueden calcular gradientes automáticos, todo queda en un
  solo framework y 6b/6c-ii se implementan **dentro** del mismo kernel (absorción y rayos por píxel), algo que ASTRA
  no permite.

- **✅ 6a hecho 13-11-2026 (escalón 1, sin kernel CUDA propio):** `vamcore/projector/torch_parallel.py`,
  `make_projector(..., backend="torch")`.
  - **Método de Joseph** (el de ASTRA): se recorre el volumen fila a fila (o columna a columna) con interpolación
    lineal y peso 1/|cos| (o 1/|sin|). Como los rayos son perpendiculares a Z, la matriz 2D es la misma para todas
    las capas: se construye una vez como **matriz dispersa** y el proyector es un producto disperso × denso
    (cuSPARSE), por bloques de capas z autoajustados (a 512 el bloque de 64 baja de 225 a 115 ms).
  - **Retroproyección por vóxel** (la de ASTRA), no la traspuesta exacta de Joseph. Hallazgo: con la traspuesta exacta
    («por rayos») AᵀA·t tiene un máximo un 11 % mayor con la misma media (vóxeles calientes) y, como el CAL normaliza
    por el máximo, a 512 no arrancaba (VER estancado en 0,20). La traspuesta exacta sigue disponible
    (`backprojector="exact"`, adjunto a 2e-9) para gradientes y pruebas. `backend="torch_grid"` es la versión con
    `grid_sample` + autograd, de referencia (10–80× más lenta).
  - **Exactitud frente a ASTRA:** proyección 6e-5 y retroproyección 5e-5 (L2) a 512. CAL a 512: misma trayectoria de
    VER y PW, misma parada (it. 156), Score Bio 86,03 en ambos.
  - **Velocidad** (optimización, vasculatura; `promocional/bench_projector_*.json`): 256 → 10,6 s ASTRA / 8,0 s propio
    (**0,76×**); 512 → 37,1 s / 40,4 s (**1,09×**, criterio ≤ 1,2×). Pipeline completo a 512: 1,10×. La prueba de
    robustez es más lenta con el propio (7,2 s frente a 3,7 s a 512) porque reconstruye la retroproyección para cada
    escenario de ángulos perturbados.
  - En la API (`projector`) y en la interfaz (selector «Proyector»). **ASTRA sigue por defecto** hasta que el usuario lo
    valide.
  - Pendiente: cuSPARSE necesita ~5,7 GB de VRAM pico a 512 al construir las matrices. La DLL de cuDNN de este entorno
    está rota: `grid_sample` y las convoluciones se ejecutan sin cuDNN.

**6b · Absorción exacta (Beer–Lambert).** Atenuación acumulada a lo largo de cada rayo, `exp(−∫α ds)`, aplicada en el
proyector directo y en su adjunto. Sustituye a la máscara aproximada de VAMToolbox.
- **Parámetro:** α de la resina (cm⁻¹), editable en la interfaz como los del vial.
- **Criterio:** coincide con la solución analítica en un cilindro homogéneo y pasa la prueba del adjunto.

- **✅ 6b hecho 13-11-2026:** en el proyector propio (`alpha`, `r_resin` de `SparseParallel3D`). Cada entrada de la
  proyección (en el punto de muestreo de Joseph) y de la retroproyección (en el centro del vóxel) se multiplica por
  exp(−α·L), con L = p·r + √(R² − |p|² + (p·r)²) la distancia recorrida dentro de la resina desde la pared interior
  del vial (o el borde del área de trabajo). `cal_core.run(absorption_cm=…)`, campo `absorption_cm` en
  `POST /optimize` y «Absorción α (cm⁻¹)» en la interfaz; con α > 0 se usa el proyector propio (ASTRA no la modela) y
  la prueba de robustez la incluye. No disponible con el motor «vamtoolbox».
  - **Validación** (`tests/test_absorption.py`): proyección de un cilindro homogéneo frente a (1 − e^(−2αh))/α; dosis
    frente a la suma directa Σ e^(−αL) (1e-4); sentido de la luz con un solo ángulo (cociente e^(αR) exacto); adjunto
    exacto con absorción (`backprojector="exact"`); α = 0 idéntico a sin absorción.
  - **Medido** (vasculatura, 256, resina de radio 10 mm; `promocional/bench_absorption_256.json`): sin modelar la
    absorción la ventana de proceso se pierde a partir de α ≈ 1 cm⁻¹ (α = 1: PW +0,033, tolerancia 0 %; α = 2:
    VER 1,5 %). Modelándola, la tolerancia de exposición se mantiene en ±3,2–3,4 % para α = 0,5, 1 y 2.
  - **Pendiente de calibrar en la máquina:** el α real de la resina a 405 nm y el sentido de la luz (se asume que el
    proyector está en el lado −r del rayo de ASTRA; si fuera el opuesto, la compensación quedaría girada 180°).

**6c · Refracción del vial dentro del modelo directo.** Es la mejora clave. Dentro de la resina, que es homogénea,
**los rayos son rectos**: el vial solo cambia por dónde y con qué ángulo entra cada rayo. El píxel (θ, u) del
proyector produce dentro de la resina un rayo con dirección θ − δ(u), distancia t(u) al eje y peso T(u). Son
exactamente las funciones que ya calcula `vial.ray_map`. Hay dos variantes:
- **6c-i (barata):** `A_vial = R ∘ A_paralelo`, donde R es el remuestreo lineal que ya hace `correct_sinogram`,
  expresado como operador con su adjunto Rᵀ.
- **6c-ii (exacta):** un proyector con un rayo por píxel y dirección y desplazamiento propios, sin interpolación
  intermedia.

**Por qué es mejor que la corrección actual:** hoy CAL optimiza en el espacio «ideal» y después deformamos las
imágenes. Con 6c, CAL optimiza **directamente la imagen que emite el proyector**. Las restricciones de positividad,
de nivel máximo y de discretización a 8 bits se aplican sobre los píxeles reales, se acaba el recorte posterior por
la normalización con T(u), y las métricas de dosis ya incluyen el efecto del vial.
- **Validación:** se compara con el trazador `pyTorchRayTrace` de VAMToolbox (de referencia, no se distribuye) y con
  el trazador de Snell de `tests/test_vial.py`.
- **Criterio:** la dosis simulada coincide con la referencia en ±1 %; con un vial de n = 1 todo debe reducirse al
  caso paralelo.

- **✅ 6c-i hecho 13-11-2026:** `vamcore/optics/vial_operator.py` (`VialResampler` = G, `VialProjector` = A ∘ G).
  G (imagen del proyector → rayos en la resina): b(φ, t) = p(θ = φ + δ(u), u)·T(u)/T₀ con u = n_r·t/n₀, bilineal en
  (θ, u), igual para todas las capas; Gᵀ explícita. El CAL optimiza la imagen del proyector sin cambios en el bucle;
  arranca de la FBP ideal corregida y exporta la imagen optimizada sin corregir después. `cal_core.run(vial_mode=
  "model")`, `vial_mode` en `POST /optimize` y «Corrección: Posterior / En el modelo» en el panel del vial
  (por defecto, «Posterior»). Funciona con los dos proyectores y con la absorción (que entra por la pared interior).
  - **Hallazgo de muestreo:** el rayo de la resina en t viene de la columna u = n_r·t/n₀ (×1,35 con GelMA). Con
    columnas del proyector al paso del vóxel, la rejilla de la resina era 1,35× más gruesa y Gᵀ repartía el gradiente
    en peine: el CAL divergía a partir de la iteración 20. Con columnas a paso pitch·n_r/n₀ cada rayo cae en una
    columna y converge igual que sin vial. La exportación usa ese paso en horizontal y el del vóxel en vertical.
  - **Validación** (`tests/test_vial_operator.py`): G deshace `correct_sinogram` (< 2 %); identidad con índices iguales;
    cada rayo coincide con el trazador de Snell vectorial (1e-9) y el valor de G con f(θ*, u*)·T/T₀ (1 %); adjunto
    exacto; el proyector compuesto coincide con el paralelo con índices iguales.
  - **Medido con la dosis física** (imagen final cuantizada a 8 bits → Aᵀ·G·p; vasculatura; vial Ø20, pared 1,
    borosilicato, GelMA; `promocional/bench_vial_model_*.json`):

    | | Corrección posterior | Vial en el modelo |
    |---|---|---|
    | 256: PW · tolerancia | −0,023 · ±1,33 % | −0,056 · **±3,30 %** |
    | 512: PW · tolerancia | −0,008 · ±0,45 % | −0,047 · **±2,65 %** |
    | 512: optimización | 41,4 s | 45,0 s |

    La corrección posterior pierde casi todo el margen real (la dosis «ideal» con la que se puntuaba no la ve: decía
    ±3,3 %). Con el vial en el modelo el margen es 2,5× (256) y 6× (512) mayor, casi el de un caso sin vial.
  - Prueba en la interfaz (128): fotogramas completos (lienzo de 1399 px = 20 mm, recorte de 1025 px ≈ zona
    imprimible), tolerancia ±3,33 %, robustez 20/20. La prueba de robustez perturba la imagen del proyector antes de G.
  - Pendiente: 6c-ii (un rayo por píxel sin interpolación intermedia) solo si hiciera falta; 6d.

- **Valores por defecto (13-11-2026, a petición del usuario):** proyector propio (`projector="torch"`) y vial dentro
  del modelo (`vial_mode="model"`) en `cal_core`, `POST /optimize` y la interfaz. ASTRA y la corrección posterior
  siguen disponibles. El motor «vamtoolbox» pasa solo a ASTRA y corrección posterior; sin CUDA, el vial vuelve a la
  corrección posterior con un aviso. Con esto, el camino por defecto ya no usa ASTRA (F7: retirarlo de la instalación).

**6d · Después.** Óptica no telecéntrica (proyector con ángulo de campo), vial no centrado o inclinado, varios
materiales dentro del volumen.

### F7 — Retirar ASTRA y empaquetar (4070)

Objetivo (D3): **un sistema empaquetado propio, con control total de lo que pasa dentro**. Las únicas dependencias de
cálculo serán PyTorch, NumPy, SciPy y trimesh/pyvista para las mallas, todas instalables con `pip`.
- Proyector propio (6a) como motor por defecto; ASTRA sale de `requirements`.
- ASTRA se conserva **solo como referencia en los tests** (extra opcional `vamcore[validation]`). Las pruebas de
  equivalencia se saltan si no está instalado, como hoy `test_astra_convention.py`.
- **Paquete:** `pyproject.toml`, versión semántica y rueda (*wheel*) con el kernel precompilado para las versiones de
  CUDA y PyTorch soportadas. Si no hay GPU compatible, se usa la ruta PyTorch pura (también en CPU, lenta, útil para
  tests y CI).
- **Instalador de la aplicación:** el backend con `vamcore` más el frontend compilado, en un solo instalador para
  Windows (el flujo de `release.yml` se adaptará en su momento).
- **Criterio:** instalación limpia en una máquina sin ASTRA ni VAMToolbox; los tests, el benchmark y un trabajo
  completo con vial producen los mismos resultados que el motor de F4 (métricas ±0,5 %), sin pérdida de velocidad
  (≤ 1,2×).

- **✅ Hecho 13-11-2026 (salvo el instalador de Windows):**
  - El camino por defecto no usa ASTRA: proyector propio por defecto en `make_projector`, `run_cal`, la prueba de
    robustez, `cal_core` y la API. Dispositivo automático (`vamcore._backend.default_device`): CUDA si está, si no
    **CPU** (ruta PyTorch pura; antes sin CUDA se caía al motor de VAMToolbox).
  - **Paquete:** `pyproject.toml` en la raíz empaqueta solo `vamcore` 0.2.0 (dependencias: numpy, scipy, trimesh,
    torch; extras `app`, `test`, `validation`). El `setup.py` de vamtoolbox pasó a `legacy/setup_vamtoolbox.py` (la
    receta de conda heredada y `release.yml` lo esperaban en la raíz; no aplican a VAM Studio). `requirements.txt`
    sin ASTRA. README con instrucciones de instalación.
  - **Pruebas:** las de equivalencia con ASTRA se saltan si no está; las referencias doradas se comparan con el
    proyector propio (y con ASTRA si está). Tolerancia del mejor VER a 64 ampliada a 5e-5 absoluto (≈15 vóxeles): con
    el proyector propio `sphere_64` difiere en 8 vóxeles.
  - **Verificado en GPU** (`promocional/bench_f7.py`; trabajo completo con vial en el modelo y robustez; ASTRA y
    VAMToolbox bloqueados en el proceso, sin ningún módulo suyo cargado), frente al mismo trabajo con ASTRA:
    256 → métricas a ≤ 0,018 %, optimización **0,88×**, total 0,93×; 512 → ≤ 0,010 %, **1,09×**, total 1,10×. Mismas
    iteraciones, robustez, Score Bio y fotogramas.
  - **Instalación limpia** (entorno virtual nuevo sin conda: la *wheel* de vamcore + PyTorch 2.15 CPU + extras):
    sin ASTRA, VAMToolbox ni pyvista; CAL en CPU (con vial en el modelo y absorción) correcto; pruebas: 98 pasan y
    30 se saltan (exactamente las de ASTRA, VAMToolbox y pyvista).
  - **Pendiente:** instalador de Windows (backend + frontend compilado; faltan los `tsconfig*.json` para
    `npm run build`), y kernel CUDA propio solo si hiciera falta (no ha hecho falta: se cumple ≤ 1,2×).

### Limpieza final (13-11-2026): todo en `vamcore`, sin VAMToolbox ni ASTRA

A petición del usuario, el proyecto queda con un solo motor y sin código de referencia externo:
- **`cal_core.py` → `vamcore/pipeline.py`** (mismo `run`, sin motor ni proyector seleccionables; parámetro `capture`
  para observar los arrays intermedios). `server.py` importa `vamcore.pipeline` y ya no acepta `engine`, `projector`
  ni `voxelizer`.
- **CLI propia:** `python -m vamcore` (o el comando `vamcore` al instalar la *wheel*) sustituye a `CAL_run.py`, que
  queda archivado en `legacy/`.
- **Borrados:** la carpeta `vamtoolbox/` y su `egg-info`, `setup.py`, la receta de conda heredada, `release.yml`, el
  backend ASTRA de `vamcore.projector`, las pruebas de paridad con ASTRA/VAMToolbox y los benchmarks comparativos
  (sus resultados medidos siguen en `promocional/bench*` y en `AUDITORIA.md` §13). El modelo de ejemplo pasó a
  `vamcore/resources/` y va dentro de la *wheel*.
- **Interfaz:** sin selector de proyector (siempre Prisma) ni rastros del motor de referencia; la parada es un campo
  propio (`stop`: Umbral VER / Margen PW). Los trabajos antiguos de la biblioteca hechos con ASTRA se siguen rotulando
  como tales.
- **Empaquetado:** sin el extra `validation`; `package-data` incluye el STL; `requirements.txt` sin la sección
  opcional. `conda/one_channel_env.yaml` es el entorno `vamstudio`.
- **Verificado:** 97 pruebas pasan, ninguna se salta; *wheel* `vamcore-0.2.0` instalada fuera del repositorio
  ejecuta la CLI en CPU sin cargar ningún módulo de ASTRA ni de VAMToolbox; trabajo real lanzado desde la interfaz
  (cubo, 128³, 360 ángulos, GPU): Score Bio 89/100, robustez 20/20, 360 fotogramas, optimización 0,5 s, total 8,3 s.
- Las referencias doradas de `tests/golden/` se generaron con el motor anterior y se conservan como están.

## 6. Riesgos y cómo se mitigan

| Riesgo | Mitigación |
|---|---|
| Cambiar resultados sin darse cuenta | Referencias de F0, criterios numéricos por fase y conmutador `engine` hasta F4 |
| No determinismo de la GPU | Tolerancias relativas, nunca igualdad exacta; semilla fija en las partes estocásticas |
| Perder la «versión anterior» del vídeo y los benchmarks | Congelarla en `legacy/` (F5), sin borrarla |
| Proyector propio lento | Interfaz intercambiable: ASTRA sigue siendo el motor por defecto hasta que el nuestro cumpla ≤ 1,2×; si PyTorch puro no basta, kernel CUDA propio (6a, escalón 2) |
| Distribuir un kernel CUDA compilado | Ruedas precompiladas por versión de CUDA/PyTorch y ruta PyTorch pura como respaldo |
| Que el proyecto crezca sin control | Un PR por fase; las fases 6b–6d son opcionales e independientes |
| Licencia | Resuelto: GPL-3.0, con atribución en los módulos adaptados |

## 7. Decisiones pendientes

- ~~**D1 · Licencia**~~ → **GPL-3.0** (decidido 13-11-2026, §3).
- **D2 · Nombre del paquete:** ¿`vamcore`, `aptum_vam`…?
- ~~**D3 · ASTRA a largo plazo**~~ → **se sustituye por un proyector propio** y queda solo como referencia en los
  tests (decidido 13-11-2026; 6a y F7).
- **D4 · Absorción:** ¿hay valores medidos de α para la GelMA y el fotoiniciador que se usan? Sin ellos, 6b queda
  como un parámetro con valor por defecto 0.

## 8. Dónde se hace cada cosa

| En este PC (ligero) | En la PC con la RTX 4070 |
|---|---|
| F1 completa (CPU, tamaños pequeños) | F0, F2, F3, F4, F5 |
| Diseño e interfaces de F6, prototipo 6c-i en CPU a 32³ | Validación y medición de velocidad de F6 |
| Tests unitarios y CI | Benchmarks y referencias doradas |

Orden recomendado: **F0 → F1 → F2 → F3 → F4 → F5 → 6c-i → 6a → 6b → 6c-ii → F7**.
- La 6c-i va primero porque funciona sobre el proyector paralelo de ASTRA y es la que más mejora el resultado físico.
- Después, la 6a sustituye a ASTRA. Con el proyector propio ya existe, la absorción (6b) y los rayos por píxel (6c-ii)
  van dentro del mismo kernel.
- F7 cierra: se retira ASTRA y se empaqueta.

La interfaz `Projector` de F2 es la pieza que hace posible todo esto sin reescribir el CAL ni el pipeline.

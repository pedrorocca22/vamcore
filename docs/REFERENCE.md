# VAMCORE — technical reference

Reference for the `vamcore` engine and the local API. For installation and use of the application, see the
[README](../README.md); for how the engine works, see the [white paper](WHITEPAPER.md).

**Contents:** [Command line](#command-line) · [Python API](#python-api) · [Engine reference](#engine-reference) ·
[Conventions](#conventions) · [HTTP API](#http-api-serverpy)

## Command line

The engine runs without the interface:

```bash
# Bundled example vasculature, 20 mm vial with a 1 mm wall, automatic orientation
python -m vamcore --example output/ --res 128 --vial 20 1 --auto-orient
```

```text
Score Bio 93/100 · EXCELENTE (simulado; no garantiza la impresión)
  VER 0.00000 · PW -0.0546 · tolerancia de exposición ±3.3 % · robustez 20/20
  250 iteraciones (max_iter) · optimización 1.8 s · total 12.2 s
  salida: output/
```

(Measured on an RTX 4060 Ti; on a CPU the optimization takes much longer. The console messages are currently in
Spanish.)

```text
python -m vamcore [STL] [OUTPUT] [options]        (or the `vamcore` command once installed)
```

| Option | Default | What it does |
|---|---|---|
| `STL` | — | Model to print (mm; Z = rotation axis). |
| `OUTPUT` | next to the STL | Results folder. |
| `--example` | — | Uses the bundled example vasculature (`vamcore.resources.example_stl()`). |
| `--res N` | 128 | Layers in Z (64, 128, 256, 512…). Sets the voxel size: height / N. |
| `--angles N` | 360 | Projections per turn. |
| `--vial DIAM WALL` | no vial | Vial without a bath: outer diameter and wall thickness (mm). |
| `--vial-material` | `borosilicate` | `borosilicate`, `fused_silica`, `soda_lime`, `pmma`, `polystyrene`. |
| `--resin` | `gelma` | `gelma`, `water`, `acrylate` (refractive index of the resin). |
| `--vial-mode` | `model` | `model`: refraction inside the optimization; `post`: correction afterwards. |
| `--absorption α` | 0 | Resin absorption in cm⁻¹ (Beer–Lambert). |
| `--stop` | `ver` | `ver`: volumetric error threshold; `margin`: process-window margin (faster). |
| `--margin δ` | 0.03 | Process-window margin for `--stop margin` (dose normalized to 1). |
| `--auto-orient` | — | Orients the part to minimize its width around the axis, within the limits. |
| `--rot RX RY RZ` | 0 0 0 | Manual rotation in degrees (STL frame). |
| `--cpu` | — | Forces the CPU even if a GPU is available. |
| `--no-robustness` | — | Skips the robustness test (faster; the Score Bio redistributes its weight). |

## Python API

### Full pipeline: `vamcore.pipeline.run`

```python
from vamcore.pipeline import run
from vamcore.optics.vial import VIAL_MATERIALS, RESIN_PRESETS

r = run("part.stl", "output/", resolution=256, num_angles=360,
        vial=dict(outer_diameter_mm=20, wall_thickness_mm=1,
                  n_vial=VIAL_MATERIALS["borosilicate"], n_resin=RESIN_PRESETS["gelma"]),
        absorption_cm=0.0)
if r["success"]:
    print(r["score_bio"], r["verdict"], r["metrics"]["ver"])
```

| Parameter | Default | Description |
|---|---|---|
| `stl_path`, `output_dir` | —, next to the STL | Input and output folder. |
| `rx`, `ry`, `rz` | 0 | Rotation in degrees, R = Rz·Ry·Rx, in the STL frame (Z = rotation axis). |
| `resolution` | 128 | Layers in Z. Iterations and learning rate follow `CALOptions.for_resolution`. |
| `num_angles` | 360 | Projections per turn. |
| `filter_type` | `"cosine"` | FBP initialization filter: `ram-lak`, `shepp-logan`, `cosine`, `hamming`, `hanning`. |
| `cuda` | `True` | `False` forces the CPU; without CUDA the CPU is used anyway. |
| `iterations`, `learning_rate` | by resolution | Override the `CALOptions.for_resolution` schedule. |
| `exit_param` | 0.0005 | VER threshold for stopping. |
| `stall_patience`, `stall_rel_tol` | 10, 0.01 | Stall stopping (0 disables it). |
| `stop_ver_mask` | `"half"` | Masks for the stopping VER: `half` (at 50 %) or `guard_band` (partial voxels excluded). |
| `stop_pw_margin` | `None` | δ > 0: stop when the process window PW ≤ −δ (use with `guard_band`). |
| `vial` | `None` | `dict` or `VialOptics`: `outer_diameter_mm`, `wall_thickness_mm`, `n_vial`, `n_resin`, `n_outside`. |
| `vial_mode` | `"model"` | `model` (refraction inside the model) or `post` (correction afterwards). |
| `absorption_cm` | 0 | Resin α (cm⁻¹). Light enters through the inner vial wall. |
| `projector_pixel_um` | 14.3 | Projector pixel size in the vial plane. |
| `voxel_pitch_um` | height / `resolution` | Explicit voxel size. |
| `robustness` | `True` | `True`, `False` or a `RobustnessSpec` with other perturbations. |
| `capture` | `None` | `dict` filled with the intermediate arrays (see below). |

**Returns** a `dict` with these keys:
- `success` and `error`;
- `score_bio` (0–100) and `verdict`: `EXCELENTE` ≥ 85, `BUENO` ≥ 70, `ACEPTABLE` ≥ 55, `POBRE` ≥ 40, `FALLIDO` below (excellent, good, acceptable, poor, failed);
- `score` (dose score) and `penalties`;
- `metrics`: VER, PW, IPDR, CV, contrast, per-layer metrics, exposure tolerance, iterations, stop reason…;
- `robustness`, `stage_times` (seconds per stage) and `ver_curve`;
- `elapsed_time` (optimization), `total_time` and `output_dir`.

`result.json` stores the same, plus the geometry, mesh, voxelization and vial data.

**`capture`** receives:
- `target` (occupancy), `sinogram` (projector image `[u, angle, z]`) and `reconstruction` (simulated dose);
- `ver` (curve) and `d_h`;
- `n_iter`, `learning_rate`, `filter`, `voxel_pitch_mm` and `angles`.

**Output files:**
- `frame_NNNN.png`: one per angle, at the projector's physical scale and cropped to the useful area;
- `result.json`, `metricas.csv` and `model_oriented.stl`;
- `reconstruction_slices.png`, `layer_profile.png` and `dose_histogram.png`.

The machine constants live in `vamcore.pipeline`:
- `PROJECTOR_PIXEL_UM = 14.3`;
- `BUILD_D_MM = 20` and `BUILD_H_MM = 25`, a build volume of Ø20 × 25 mm;
- `NUM_ANGLES`, `RESOLUTION` and `EXIT_PARAM`.

### Individual modules

```python
import numpy as np
from vamcore.voxelize import voxelize_mesh
from vamcore.dose import suggest_d_h
from vamcore.config import CALOptions
from vamcore.optim.cal import run_cal
from vamcore.ops import metric_masks
from vamcore.metrics import all_metrics

target, grid, stats = voxelize_mesh(mesh, pitch_mm=0.125)      # occupancy 0…1, [ix, iy, iz]
opts = CALOptions.for_resolution(target.shape[2]).with_(d_h=suggest_d_h(target).d_h)
res = run_cal(target, np.linspace(0, 360, 360, endpoint=False), opts)
gel, void, info = metric_masks(target)                         # guard band
print(res.stop, all_metrics(res.reconstruction, gel, void))
```

Complete, runnable examples in [`examples/`](../examples/): `pipeline_basico.py` and `cal_bajo_nivel.py`.

## Engine reference

### Flow

```text
STL ─► orient (orient) ─► slice voxelization (voxelize) ─► d_h from mean thickness (dose)
    ─► CAL with Prisma (optim.cal + projector) ── vial inside the model (optics) · absorption (projector)
    ─► guard-band metrics (ops, metrics) ─► exposure tolerance and robustness (robustness)
    ─► Score Bio ─► frames at physical scale + result.json (pipeline)
```

### Modules

| Module | Main contents |
|---|---|
| `vamcore.pipeline` | `run(...)`: the full process; machine constants; frame export. |
| `vamcore.__main__` | Command line (`python -m vamcore`, `vamcore` command). |
| `vamcore.config` | `CALOptions` (iterations, learning rate, momentum, sigmoid, filter, `d_h`, how often the VER is evaluated), `StopCriteria`; `ProjectorSpec`, `VoxelGrid`, `PrintJob`. All immutable, no global state. |
| `vamcore.optim.cal` | `run_cal(target, angles_deg, opts, ...)` → `CALResult(sinogram, reconstruction, ver, pw, stop, times)`. |
| `vamcore.optim.stopping` | `StopRule`: stopping on a VER threshold, a process-window margin or a stall. |
| `vamcore.projector` | `make_projector(vol_shape, angles_deg, backend="torch", device=None, **kw)`; `Projector` protocol (`forward`, `adjoint`, `close`; usable with `with`). |
| `vamcore.projector.torch_parallel` | `SparseParallel3D` (**Prisma**): `alpha`/`r_resin` for absorption, auto-tuned `z_chunk`, `backprojector="voxel"` or `"exact"`. `TorchParallel3D`: `grid_sample` variant, kept as a reference. |
| `vamcore.optics.vial` | `VialOptics` (geometry, `printable_radius`, `ray_map`), `VIAL_MATERIALS`, `RESIN_PRESETS`, `correct_sinogram` (`post` mode), `check_fit`. |
| `vamcore.optics.vial_operator` | `VialResampler` (projector image → resin rays, with adjoint) and `VialProjector` (Prisma composed with the vial). |
| `vamcore.voxelize` | `voxelize_mesh(mesh, pitch_mm, ...)` → `(occupancy, grid, stats)`; `native_pitch_mm`, `mesh_report`. |
| `vamcore.dose` | `suggest_d_h(occupancy)` → `DhSuggestion(d_h, mean_thickness_vox, tag)` from the mean thickness (EDT). |
| `vamcore.metrics` | `ver`, `pw`, `ipdr`, `cv` (also per layer with `layerwise=True`), `all_metrics`. |
| `vamcore.ops` | `metric_masks` (guard band), `get_inds`, `circle_mask`, `clip_to_circle`, `sigmoid`, `discretize`. |
| `vamcore.filters` | `filter_sinogram(sino, name)`, `fourier_filter`, `FILTERS`. |
| `vamcore.robustness` | `exposure_tolerance`, `RobustnessSpec`, `robustness_test`, `reliability_score`. |
| `vamcore.orient` | `find_orientation(vertices, max_height_mm, max_radius_mm, ...)` → `OrientResult`. |
| `vamcore.resources` | `example_stl()`: the example trifurcated vasculature. |

### Physical and numerical models

The [white paper](WHITEPAPER.md) explains each model in detail, with the measurements behind it.

- **Prisma.**
  - Joseph's method as a sparse 2D matrix, shared by every Z layer because the rays are perpendicular to the axis.
  - Sparse × dense product in cuSPARSE, with auto-tuned layer blocks.
  - Voxel-driven backprojection: the exact transpose created hot voxels and made CAL diverge at 512.
- **CAL.** Starts from an FBP initialization and descends with momentum on the sinogram, using the resin's sigmoid
  response centred on `d_h`, a contrast chosen from the part's mean thickness.
- **Absorption.** Beer–Lambert with the path length in the resin from the inner vial wall:
  L = p·r + √(R² − |p|² + (p·r)²).
- **Vial inside the model.**
  - CAL optimizes the image the projector emits directly, composed with the refraction and Fresnel transmission at
    the two vial interfaces.
  - The column spacing is pitch·n_r/n₀, so the adjoint does not spread the gradient in a comb pattern.
  - With the physical dose it gives 2.5 to 6 times more exposure tolerance than post-correction.
- **Guard-band metrics.**
  - Voxels with occupancy between 0.05 and 0.95 are half inside the part: they count neither as gel nor as void.
  - VER is the fraction of misclassified voxels.
  - PW = max(void dose) − min(gel dose). A negative value means there is a process window.
- **Exposure tolerance.** ε = (r − 1)/(r + 1), with r = min(gel)/max(void) and the threshold at the harmonic mean.
  It says how much the global dose can vary before the part changes.
- **Robustness.**
  - 20 perturbed scenarios: axis offset, speed error, angle jitter, blur, LED flicker and dose error.
  - Each is evaluated with the same projector, counting how many keep the shape.
- **Score Bio** = Q · (0.6 + 0.2·T + 0.2·R), bounded to 0–100:
  - Q is the simulated dose score;
  - T = min(1, tolerance/±5 %);
  - R is the fraction of passing scenarios.
- **Stopping.** Three criteria:
  - VER threshold, the default;
  - process-window margin (`stop_pw_margin`), much faster at the cost of less tolerance;
  - stall.
- **Auto-orientation.**
  - Minimizes the mean projected width over 72 directions (10° grid and Nelder-Mead).
  - Discards orientations that exceed the printable height or radius.
  - It is purely geometric.

## Conventions

- **Axes:** volume `[ix, iy, iz]` with the rotation axis at the centre of (ix, iy) and z increasing with iz; sinogram
  `[u, angle, z]`.
- **Units:** millimetres, unless the name says otherwise (`_um`, `_cm`, `_deg`). Angles in degrees.
- **Arrays:** modules accept NumPy and, with torch installed, `torch.Tensor` (computed on its device).
  Importing `vamcore` does not load torch.

## HTTP API (`server.py`)

| Method and route | Use |
|---|---|
| `GET /health` | API status. |
| `GET /vial/presets` · `GET /vial/preview` | Materials and resins; printable zone of a vial. |
| `POST /orient` | Auto-orientation of an STL (within the vial or build-area limits). |
| `POST /optimize` | Starts a job (STL + form parameters) → `job_id`. |
| `GET /status/{job_id}` | Progress and result. |
| `GET /library` · `POST /save/{job_id}` · `DELETE /library/{job_id}` | Job library. |
| `GET /export/{job_id}.json` | Full result. |
| `GET /job/{job_id}/frame/{i}` · `GET /job/{job_id}/artifact/{name}` · `GET /file/{job_id}/{path}` | Frames and diagnostic images (paths validated against path traversal). |
| `GET /printer_status` · `POST /print/{job_id}` · `GET /print_send_status/{job_id}` | Projector status and sending the image stack. |

`POST /optimize` accepts, among others:
- `num_angles`, `resolution`, `filter`, `cuda` and `rx`/`ry`/`rz`;
- `stop_ver_mask` and `stop_pw_margin`;
- `absorption_cm`;
- `vial_outer_diameter_mm`, `vial_wall_thickness_mm`, `vial_material` or `n_vial`, `resin` or `n_resin`, and `vial_mode`.

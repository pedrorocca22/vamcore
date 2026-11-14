<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-optional-76B900?logo=nvidia&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![Three.js](https://img.shields.io/badge/Three.js-000000?logo=threedotjs&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?logo=vite&logoColor=white)
![Klipper](https://img.shields.io/badge/Klipper-Moonraker-B12F36)
![License](https://img.shields.io/badge/license-GPL--3.0-blue)

# VAMCORE

**Tomographic volumetric printing, from 3D model to projection**

<img src="assets/images/vamcore_interfaz.png" alt="VAMCORE interface" width="900"/>

</div>

## Overview

VAMCORE is complete software for **tomographic volumetric printing** (*volumetric additive manufacturing*, *computed axial lithography*). Starting from an STL file, it computes the sequence of images the projector emits while the vial rotates, evaluates the quality of the simulated print, and sends the sequence to the printer in sync with the rotation motor.

At its core is **Prisma**, an in-house tomographic engine written in PyTorch and accelerated on the GPU. It models the refraction of the vial and the absorption of the resin inside the optimization itself, which makes it possible to print without an index-matching bath. Every result comes with a **Score Bio /100** that combines the simulated dose, the exposure tolerance and a robustness test against machine errors.

> **Beta.** Results are simulated: the Score Bio measures the quality of the generated images and does not guarantee a successful print. No print tests have been carried out yet.

## Key features

- **Prisma engine:** sparse tomographic projector on the GPU (or CPU), CAL optimization, voxelization with partial occupancy and automatic part orientation.
- **Vial and resin physics:** refraction and losses at the vial walls, Beer–Lambert absorption.
- **Evaluation:** dose metrics, exposure tolerance, 20 robustness scenarios and Score Bio.
- **Web interface:** 3D viewer, vial, resin and optimization settings, results and a job library.
- **Printer:** the sequence is sent over HTTP and played back in sync with the rotation (Klipper).

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU; without a GPU: pip install torch
pip install -e ".[app]"
python server.py                     # local API at http://127.0.0.1:8000
cd frontend && npm ci && npm run dev # interface at http://localhost:5173
```

It also runs without the interface:

```bash
python -m vamcore --example output/ --res 128 --vial 20 1 --auto-orient
```

## Performance

Measured on an RTX 4060 Ti 16 GB, with the example vasculature, 360 angles and the vial inside the model:

| Resolution | Optimization | Full pipeline | Score Bio |
|---|---|---|---|
| 128³ | 1.8 s | 12.2 s | 93/100 |
| 256³ | 14.4 s | 26.5 s | 93/100 |

The full pipeline includes voxelization, the robustness test and the export of 360 frames.

## Documentation

| Document | Contents |
|---|---|
| [White paper](docs/WHITEPAPER.md) | How the Prisma engine works: projector, physics, metrics, measured results and limitations. |
| [Technical reference](docs/REFERENCE.md) | Command line, Python API, engine modules, physical models and HTTP API. |
| [Projector software](device/README.md) | Projection, rotation motor, networking and installation on the printer. |
| [Examples](examples/) | Using the engine from Python. |

## Structure

```text
vamcore/     Prisma engine (Python package)          frontend/   web interface
server.py    local API                               device/     projector software
tests/       97 tests                                docs/       technical documentation
```

## License

[GPL-3.0-or-later](LICENSE). Modules derived from earlier work keep their attribution in the file header.

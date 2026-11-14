# device/ — projector software

This code runs on the projector's controller (a Raspberry Pi with the DLPC1438 and the motor board running Klipper),
not on the PC. VAMCORE sends it the stack of frames over HTTP, and the projector plays it back in sync with the
rotation of the vial.

| File | What it does |
|---|---|
| `ejemplo_servidor_subproceso.py` | Projector HTTP API (FastAPI, port 8000): receives the stack, starts the projection, status, Wi-Fi. |
| `proyectar_volumetrico.py` | Projects the sequence through the DMD and advances the motor 1° per frame. |
| `volumetric_controller.py` | Fast DLPC1438 driver over SPI/GPIO (`FastDMD`). |
| `motor_rotacion.py` | Rotation motor (NEMA 17 + TMC2209, 5:1 reduction) through the Moonraker/Klipper API. |
| `wifi_manager.py` | Network management with `nmcli`. |
| `klipper/printer.cfg` | Klipper configuration used on the prototype (the Z axis is the rotation motor). |

**Installation** from the PC. It needs `pip install -e ".[device]"` and a `config.json` in the repository root,
copied from `config.example.json`, with the projector's SSH user and password:

```bash
python upload_to_projector.py      # uploads these files to /home/pi and installs fastapi/uvicorn
python start_projector_server.py   # starts the projector API in the background
```

The projector also needs `RPi.GPIO`, `smbus`, `spidev`, `numpy` and `pillow`, with Moonraker/Klipper running.

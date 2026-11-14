# device/ — software del projector

Se ejecuta en el controlador del projector (Raspberry Pi con el DLPC1438 y la placa del motor con Klipper), no en el
PC. VAMCORE le envía la pila de fotogramas por HTTP y el projector la reproduce sincronizada con el giro del vial.

| Archivo | Qué hace |
|---|---|
| `ejemplo_servidor_subproceso.py` | API HTTP del projector (FastAPI, puerto 8000): recibe la pila, lanza la proyección, estado, Wi-Fi. |
| `proyectar_volumetrico.py` | Proyecta la secuencia por el DMD y avanza el motor 1° por fotograma. |
| `volumetric_controller.py` | Controlador rápido del DLPC1438 por SPI/GPIO (`FastDMD`). |
| `motor_rotacion.py` | Motor de rotación (NEMA 17 + TMC2209, reducción 5:1) a través de la API de Moonraker/Klipper. |
| `wifi_manager.py` | Gestión de la red con `nmcli`. |
| `klipper/printer.cfg` | Configuración de Klipper usada en el prototipo (el eje Z es el motor de rotación). |

**Instalación** desde el PC (necesita `pip install -e ".[device]"` y un `config.json` en la raíz, copiado de
`config.example.json`, con el usuario y la contraseña SSH del projector):

```bash
python upload_to_projector.py      # sube estos archivos a /home/pi e instala fastapi/uvicorn
python start_projector_server.py   # arranca la API del projector en segundo plano
```

En el projector hacen falta además `RPi.GPIO`, `smbus`, `spidev`, `numpy` y `pillow`, y Moonraker/Klipper en marcha.

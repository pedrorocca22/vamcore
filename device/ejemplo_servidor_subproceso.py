import asyncio
import json
import os
import subprocess
import zipfile
import io
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta al ramdisk o archivo temporal donde el proyector volcará su progreso
ESTADO_FILE = "/tmp/estado_dmd.json"

@app.on_event("startup")
async def startup_event():
    # Inicializar estado. Usamos sudo porque el archivo pudo ser creado por root en el print anterior.
    try:
        if os.path.exists(ESTADO_FILE):
            subprocess.run(["sudo", "chmod", "666", ESTADO_FILE], check=False)
        with open(ESTADO_FILE, "w") as f:
            json.dump({"status": "IDLE"}, f)
    except Exception as e:
        print(f"[CM4 Server] No se pudo inicializar estado: {e}")
        
    # Pre-encender la placa base del proyector para "quemar" el pantallazo de arranque
    # (el logo de Elegoo) en el boot del servidor, antes de que haya resina.
    try:
        import RPi.GPIO as GPIO
        import time
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(5, GPIO.OUT)
        GPIO.output(5, GPIO.HIGH)
        print("[CM4 Server] Proyector pre-encendido para evitar flash de inicio.")
    except Exception as e:
        print("[CM4 Server] No se pudo pre-encender el proyector:", e)

# ─────────────────────────────────────────────────────────────────────────────
# WiFi Management Endpoints
# ─────────────────────────────────────────────────────────────────────────────

class WifiConnectRequest(BaseModel):
    ssid: str
    password: str = ""

@app.get("/wifi_status")
async def wifi_status():
    """Estado actual de la conexión WiFi."""
    try:
        from wifi_manager import get_current_connection
        return get_current_connection()
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.get("/scan_networks")
async def scan_networks():
    """Escanear redes WiFi disponibles."""
    try:
        from wifi_manager import scan_networks as _scan
        networks = _scan()
        return {"networks": networks}
    except Exception as e:
        raise HTTPException(500, f"Error escaneando redes: {str(e)}")

@app.post("/connect_wifi")
async def connect_wifi(req: WifiConnectRequest):
    """Conectarse a una red WiFi."""
    try:
        from wifi_manager import connect_to_network
        result = connect_to_network(req.ssid, req.password)
        if not result["success"]:
            raise HTTPException(400, result["message"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error conectando: {str(e)}")

@app.post("/disconnect_wifi")
async def disconnect_wifi():
    """Desconectar del WiFi actual."""
    try:
        from wifi_manager import disconnect
        return disconnect()
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.delete("/forget_network/{ssid}")
async def forget_network(ssid: str):
    """Olvidar una red guardada."""
    try:
        from wifi_manager import forget_network as _forget
        return _forget(ssid)
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.post("/iniciar_impresion")
async def iniciar_impresion(
    file: UploadFile = File(None),
    directorio: str = Form("/home/pi/imagenes"), 
    led_pwm: int = Form(50), 
    motor: bool = Form(True),
    loops: int = Form(1)
):
    """
    Endpoint para desempaquetar las imagenes de un ZIP e iniciar impresión en un subproceso estricto.
    No bloquea a FastAPI.
    """
    import time
    t_rx = time.perf_counter()
    
    # 0. Extraer ZIP si llegó por red
    import shutil
    if file:
        t_zip = time.perf_counter()
        directorio = "/tmp/vam_frames"
        if os.path.exists(directorio):
            shutil.rmtree(directorio)
        os.makedirs(directorio)
        
        content = await file.read()
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zip_ref:
                for member in zip_ref.infolist():
                    # Prevencion de Path Traversal (Zip Slip): extraer solo archivos, sin rutas
                    if member.is_dir(): continue
                    member_name = os.path.basename(member.filename)
                    if not member_name: continue
                    target_path = os.path.join(directorio, member_name)
                    with open(target_path, "wb") as f_out, zip_ref.open(member) as f_in:
                        f_out.write(f_in.read())
            t_zip_end = time.perf_counter()
            print(f"[CM4] ZIP extraido de forma segura en {t_zip_end - t_zip:.3f}s")
        except Exception as e:
            raise HTTPException(400, f"Error extrayendo ZIP: {e}")

    # 1. Limpiar estado viejo si existe (con cuidado de permisos)
    if os.path.exists(ESTADO_FILE):
        try:
            os.remove(ESTADO_FILE)
        except:
            pass
        
    # 2. Construir el comando del subproceso aislando E/S y GIL de Python
    cmd = [
        "sudo", "python3", "proyectar_volumetrico.py",
        "--directorio", directorio,
        "--estado", ESTADO_FILE,
        "--led_pwm", str(led_pwm),
        "--loops", str(loops),
        "--auto"  # Para no bloquear esperando ENTER
    ]
    
    if motor:
        cmd.append("--motor")
        
    print(f"Lanzando subproceso: {' '.join(cmd)}")
    
    # 3. Lanzar subprocess (Aislamiento Físico)
    # shell=False es más rápido y seguro.
    subprocess.Popen(cmd)
    
    t_total_api = time.perf_counter() - t_rx
    
    return {
        "status": "ok", 
        "message": "Proyección iniciada", 
        "metrics": {
            "extract_time_s": round(t_zip_end - t_zip, 3) if 't_zip_end' in locals() else 0,
            "api_overhead_s": round(t_total_api, 3)
        }
    }


@app.get("/estado_impresion")
async def estado_impresion():
    """
    Lee el mini-archivo JSON que el subproceso escribe constantemente.
    Carga de RAM casi "Zero I/O"
    """
    if not os.path.exists(ESTADO_FILE):
        return {"status": "IDLE", "progress_percent": 0}
        
    try:
        with open(ESTADO_FILE, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        # Pasa si hay colisión de lectura/escritura (muy raro pero posible)
        return {"status": "LEYENDO", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # uvicorn ejemplo_servidor_subproceso:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
VAM Studio — Backend API
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
"""

import re
import sys
import uuid
import json
import asyncio
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# El pipeline imprime caracteres no ASCII (→, µ, Ø, ·). Sin una consola UTF-8 (servicio, lanzador, IDE) Windows usa
# cp1252 y el print lanza UnicodeEncodeError, que hacía fallar la optimización entera.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Importamos el núcleo de cálculo
from vamcore import pipeline  # motor: vamcore.pipeline (antes cal_core.py)
from vamcore.optics.vial import VialOptics, VIAL_MATERIALS, RESIN_PRESETS

# Configuración de rutas
BASE_DIR   = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
JOBS_DIR   = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)
TMP_JOBS_DIR = BASE_DIR / "tmp_jobs"
if TMP_JOBS_DIR.exists(): shutil.rmtree(TMP_JOBS_DIR)  # Limpiar al arranque
TMP_JOBS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="VAM Studio API")

# Habilitar CORS para React (restringido a frontend local por seguridad). Cualquier puerto local: Vite se mueve a
# 5174, 5175… si el 5173 está ocupado por otra aplicación, y el backend solo escucha en 127.0.0.1.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pool para ejecutar tareas pesadas de CPU/GPU sin bloquear la API
executor = ThreadPoolExecutor(max_workers=1)

# Diccionario de estado de trabajos en memoria
jobs = {}

# ─── Validación de identificadores y rutas ──────────────────────────────────
# Los job_id se generan como uuid4()[:8]; se admite cualquier nombre seguro para trabajos antiguos,
# pero nunca separadores ni "..", que permitirían salir de JOBS_DIR / TMP_JOBS_DIR.
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _check_job_id(job_id: str) -> str:
    if not _JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(400, "Identificador de trabajo no válido")
    return job_id


def _job_file(job_id: str, rel_path: str) -> Optional[Path]:
    """Devuelve el archivo rel_path del trabajo (guardado o temporal) si existe y está DENTRO de su carpeta."""
    _check_job_id(job_id)
    for base in (JOBS_DIR, TMP_JOBS_DIR):
        root = (base / job_id).resolve()
        candidate = (root / rel_path).resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            return candidate
    return None


class JobState(BaseModel):
    job_id: str
    status: str  # "running", "done", "error"
    progress: str
    summary: Optional[dict] = None
    error: Optional[str] = None

# ─── CM4 Discovery ───────────────────────────────────────────────────────────

CM4_USB_IP    = "10.55.0.2"          # IP fija por USB Gadget (g_ether) — canal primario
CM4_HOSTNAME  = "dlp3.local"         # mDNS hostname — fallback WiFi
CM4_PORT      = 8000
CM4_CACHE_FILE = BASE_DIR / "cm4_ip_cache.json"  # Caché persistente de IP (fallback)
_cm4_ip_cache: dict = {}             # caché en memoria { "ip": "...", "ts": ... }

def _load_ip_cache():
    """Cargar IP cacheada del disco (persiste entre reinicios)."""
    global _cm4_ip_cache
    if CM4_CACHE_FILE.exists():
        try:
            with open(CM4_CACHE_FILE) as f:
                _cm4_ip_cache = json.load(f)
        except Exception:
            _cm4_ip_cache = {}

def _save_ip_cache(ip: str):
    """Guardar IP al disco."""
    global _cm4_ip_cache
    import time
    _cm4_ip_cache = {"ip": ip, "ts": time.time()}
    try:
        with open(CM4_CACHE_FILE, "w") as f:
            json.dump(_cm4_ip_cache, f)
    except Exception:
        pass

def resolve_cm4_ip() -> str:
    """
    Resuelve la IP de la CM4 con prioridad clara:
      1. USB Gadget directo (10.55.0.2) — instantáneo, más rápido
      2. mDNS WiFi (dlp3.local)         — fallback si no hay cable USB
      3. IP cacheada / fallback conocido — último recurso (TTL 24h, BUG-03 fix)
    Hace un probe TCP rápido (0.3s) para verificar que el puerto está abierto.
    """
    import socket, time

    # Construir lista de candidatos en orden de preferencia
    candidates = [CM4_USB_IP]

    # Añadir IP resuelta por mDNS
    try:
        info = socket.getaddrinfo(CM4_HOSTNAME, None, socket.AF_INET)
        wifi_ip = info[0][4][0]
        if wifi_ip and wifi_ip != "127.0.0.1" and wifi_ip not in candidates:
            candidates.append(wifi_ip)
    except Exception:
        pass

    # BUG-03 fix: caché como último fallback SOLO si tiene menos de 24 horas
    cached_ip = _cm4_ip_cache.get("ip")
    cached_ts = _cm4_ip_cache.get("ts", 0)
    cache_age_h = (time.time() - cached_ts) / 3600
    if cached_ip and cache_age_h < 24 and cached_ip not in candidates:
        candidates.append(cached_ip)

    # Probe activo: devolver la primera que tenga el puerto abierto
    for ip in candidates:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.3)
            if s.connect_ex((ip, CM4_PORT)) == 0:
                s.close()
                if ip != cached_ip:
                    _save_ip_cache(ip)
                return ip
            s.close()
        except Exception:
            pass

    # Si ninguna responde, lanzar error para que el frontend sepa que está desconectada
    raise HTTPException(status_code=503, detail="No se encontró ninguna placa CM4 activa.")

# ─── Caché de IP resuelta de corta duración (BUG-06 fix) ─────────────────────
# Evita un probe TCP por cada llamada a /printer_status (1/seg durante impresión)
_resolved_ip_cache: dict = {}   # {"ip": str, "ts": float}
RESOLVED_IP_TTL_S = 20          # Re-probar cada 20 segundos

def get_cm4_url(path: str = "") -> str:
    """Construir URL completa del endpoint de la CM4. Usa caché de corta duración."""
    import time
    cached = _resolved_ip_cache.get("ip")
    cached_ts = _resolved_ip_cache.get("ts", 0)
    if cached and (time.time() - cached_ts) < RESOLVED_IP_TTL_S:
        return f"http://{cached}:{CM4_PORT}{path}"
    # Caché expirado o vacío — resolver y actualizar
    ip = resolve_cm4_ip()
    _resolved_ip_cache["ip"] = ip
    _resolved_ip_cache["ts"] = time.time()
    return f"http://{ip}:{CM4_PORT}{path}"

def invalidate_resolved_ip_cache():
    """Forzar re-probe en la próxima petición (usar cuando una petición falla)."""
    _resolved_ip_cache.clear()

# Inicializar caché al arrancar
_load_ip_cache()

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.1.0"}


# --- VIAL (corrección de refracción sin baño) ---

def _build_vial(outer_d, wall, material, n_vial, resin, n_resin, n_outside) -> Optional[VialOptics]:
    """Construye VialOptics desde campos de formulario. Sin diámetro → sin corrección."""
    if outer_d is None or outer_d <= 0:
        return None
    if n_vial is None:
        if material not in VIAL_MATERIALS:
            raise HTTPException(400, f"Material de vial desconocido: {material}. Usa uno de {list(VIAL_MATERIALS)} o indica n_vial.")
        n_vial = VIAL_MATERIALS[material]
    if n_resin is None:
        if resin not in RESIN_PRESETS:
            raise HTTPException(400, f"Resina desconocida: {resin}. Usa una de {list(RESIN_PRESETS)} o indica n_resin.")
        n_resin = RESIN_PRESETS[resin]
    vial = VialOptics(float(outer_d), float(wall or 0.0), float(n_vial), float(n_resin), float(n_outside or 1.0))
    try:
        vial.validate()
    except ValueError as e:
        raise HTTPException(400, str(e))
    return vial


@app.get("/vial/presets")
def vial_presets():
    """Índices aproximados a 405 nm para poblar los selectores de la interfaz."""
    return {"vial_materials": VIAL_MATERIALS, "resins": RESIN_PRESETS}


@app.get("/vial/preview")
def vial_preview(
    outer_diameter_mm: float, wall_thickness_mm: float = 1.0, material: str = "borosilicate",
    n_vial: Optional[float] = None, resin: str = "gelma", n_resin: Optional[float] = None,
    n_outside: float = 1.0,
):
    """Cálculo instantáneo (sin GPU) de la zona imprimible para mostrarlo mientras se editan los campos."""
    vial = _build_vial(outer_diameter_mm, wall_thickness_mm, material, n_vial, resin, n_resin, n_outside)
    return vial.summary()


@app.post("/orient")
async def orient(
    file: UploadFile = File(...),
    vial_outer_diameter_mm: Optional[float] = Form(None),
    vial_wall_thickness_mm: float = Form(1.0),
    vial_material: str = Form("borosilicate"),
    n_vial: Optional[float] = Form(None),
    resin: str = Form("gelma"),
    n_resin: Optional[float] = Form(None),
    n_outside: float = Form(1.0),
):
    """Auto-orientación geométrica (vamcore.orient): devuelve el giro en el sistema del STL (convención del pipeline)
    que hace la pieza lo más estrecha posible alrededor del eje de giro, sin superar la altura de impresión ni el
    radio disponible (zona imprimible del vial si se indica; si no, el área de trabajo). Tarda ~1 s; sin GPU."""
    import io
    import trimesh
    from vamcore.orient import find_orientation

    if Path(file.filename or "").suffix.lower() != ".stl":
        raise HTTPException(400, "Solo se admiten archivos STL")
    vial = _build_vial(vial_outer_diameter_mm, vial_wall_thickness_mm, vial_material, n_vial, resin, n_resin, n_outside)
    try:
        mesh = trimesh.load(io.BytesIO(await file.read()), file_type="stl", force="mesh")
    except Exception as e:
        raise HTTPException(400, f"No se pudo leer el STL: {e}")
    build_d = float(getattr(pipeline, "BUILD_D_MM", 20.0))
    build_h = float(getattr(pipeline, "BUILD_H_MM", 25.0))
    max_r = float(vial.printable_radius) if vial is not None else build_d / 2
    # Pool por defecto (no `executor`, que tiene un solo hilo y está ocupado mientras optimiza)
    res = await asyncio.get_running_loop().run_in_executor(
        None, lambda: find_orientation(mesh.vertices, max_height_mm=build_h, max_radius_mm=max_r))
    out = res.__dict__.copy()
    out["radius_limit_source"] = "vial" if vial is not None else "build_area"
    return out


# --- OPTIMIZATION ---



@app.post("/optimize")
async def optimize(
    file: UploadFile = File(...),
    cuda: bool       = Form(True),
    rx: float        = Form(0.0),
    ry: float        = Form(0.0),
    rz: float        = Form(0.0),
    num_angles: int  = Form(360),
    resolution: int  = Form(128),
    filter: str      = Form("cosine"),
    method: str      = Form("CAL"),
    # Vial (opcional): sin diámetro no se aplica corrección (equivale a usar baño de index matching)
    vial_outer_diameter_mm: Optional[float] = Form(None),
    vial_wall_thickness_mm: float = Form(1.0),
    vial_material: str = Form("borosilicate"),
    n_vial: Optional[float] = Form(None),
    resin: str = Form("gelma"),
    n_resin: Optional[float] = Form(None),
    n_outside: float = Form(1.0),
    projector_pixel_um: Optional[float] = Form(None),
    voxel_pitch_um: Optional[float] = Form(None),
    # Parada: máscaras del VER («half» o «guard_band») y margen de ventana de proceso
    stop_ver_mask: str = Form("half"),
    stop_pw_margin: Optional[float] = Form(None),
    absorption_cm: float = Form(0.0),
    vial_mode: str = Form("model"),
):
    method = method.upper()
    if method != "CAL":
        raise HTTPException(400, "Método no soportado. Por ahora solo CAL.")
    vial = _build_vial(vial_outer_diameter_mm, vial_wall_thickness_mm, vial_material, n_vial,
                       resin, n_resin, n_outside)
    if projector_pixel_um is not None and projector_pixel_um <= 0:
        raise HTTPException(400, "projector_pixel_um debe ser > 0")
    if voxel_pitch_um is not None and not (1.0 <= voxel_pitch_um <= 1000.0):
        raise HTTPException(400, "voxel_pitch_um debe estar entre 1 y 1000 µm")
    if stop_ver_mask not in ("half", "guard_band"):
        raise HTTPException(400, "stop_ver_mask debe ser 'half' o 'guard_band'")
    if vial_mode not in ("post", "model"):
        raise HTTPException(400, "vial_mode debe ser 'post' o 'model'")
    if not (0.0 <= absorption_cm <= 100.0):
        raise HTTPException(400, "absorption_cm debe estar entre 0 y 100 cm⁻¹")
    if stop_pw_margin is not None and not (0.0 < stop_pw_margin < 0.5):
        raise HTTPException(400, "stop_pw_margin debe estar entre 0 y 0,5")
    job_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix
    if ext.lower() not in [".stl"]:
        raise HTTPException(400, "Solo se admiten archivos STL")
    
    job_dir = TMP_JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    stl_path = job_dir / f"model{ext}"
    with open(stl_path, "wb") as f:
        f.write(await file.read())

    jobs[job_id] = {
        "status": "running",
        "progress": "Iniciando optimización...",
        "summary": None,
        "error": None
    }

    executor.submit(
        _run_job, 
        job_id, 
        str(stl_path), 
        str(job_dir), 
        cuda, rx, ry, rz,
        num_angles, resolution, filter, method,
        vial, projector_pixel_um, voxel_pitch_um, stop_ver_mask, stop_pw_margin, absorption_cm, vial_mode,
    )

    return {"job_id": job_id}

def _run_job(job_id, stl_path, job_dir, cuda, rx, ry, rz, num_angles, resolution, filter_type, method_type,
             vial=None, projector_pixel_um=None, voxel_pitch_um=None, stop_ver_mask="half", stop_pw_margin=None,
             absorption_cm=0.0, vial_mode="model"):
    try:
        jobs[job_id]["progress"] = "Calculando transformaciones y optimización..."
        res = pipeline.run(
            stl_path, 
            output_dir=job_dir,
            rx=rx, ry=ry, rz=rz,
            num_angles=num_angles,
            resolution=resolution,
            filter_type=filter_type,
            method=method_type,
            cuda=cuda,
            vial=vial,
            projector_pixel_um=projector_pixel_um,
            voxel_pitch_um=voxel_pitch_um,
            stop_ver_mask=stop_ver_mask,
            stop_pw_margin=stop_pw_margin,
            absorption_cm=absorption_cm,
            vial_mode=vial_mode,
        )

        if res["success"]:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["progress"] = "Finalizado con éxito"
            jobs[job_id]["summary"] = res
        else:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = res.get("error", "Error desconocido en el motor")
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    _check_job_id(job_id)
    if job_id in jobs:
        return jobs[job_id]
    
    # Si no está en memoria, buscar en disco (ya sea en tmp o final)
    # Primero en tmp (por si el servidor se reinició pero sigue ahí)
    job_dir = TMP_JOBS_DIR / job_id
    if not job_dir.exists():
        job_dir = JOBS_DIR / job_id
    
    res_path = job_dir / "result.json"
    if res_path.exists():
        try:
            with open(res_path) as f:
                raw = f.read()
                summary = json.loads(raw.replace('NaN','null').replace('Infinity','null'))
            
            frames_8bit = sorted(job_dir.glob("frame_*.png"))
            frames_1bit = sorted((job_dir / "bitplanes").glob("bit_*.png")) if (job_dir / "bitplanes").exists() else []
            
            return {
                "status": "done", 
                "progress": "Cargado de disco", 
                "summary": summary,
                "all_frames": [f"/file/{job_id}/{f.name}" for f in frames_8bit],
                "all_bitplanes": [f"/file/{job_id}/bitplanes/{f.name}" for f in frames_1bit]
            }
        except: pass
        
    raise HTTPException(404, "Trabajo no encontrado")

# --- LIBRARY & PERSISTENCE ---

@app.get("/library")
async def get_library():
    items = []
    for job_dir in sorted(JOBS_DIR.iterdir(), reverse=True):
        if not job_dir.is_dir(): continue
        
        # Intentar cargar result.json
        res_path = job_dir / "result.json"
        if not res_path.exists(): continue
        
        try:
            # VAM Studio espera un "summary" con métricas para mostrar en la lista
            with open(res_path, 'r', encoding='utf-8') as f:
                raw = f.read()
                # Limpiar NaNs/Infs que Python json no aguanta bien en modo estándar
                data = json.loads(raw.replace('NaN','null').replace('Infinity','null'))
                
            # Extraer miniatura si existe
            frames_8bit  = sorted(job_dir.glob("frame_*.png"))
            frames_1bit  = sorted((job_dir / "bitplanes").glob("bit_*.png")) \
                           if (job_dir / "bitplanes").exists() else []

            sample = f"/file/{job_dir.name}/{frames_8bit[len(frames_8bit)//2].name}" if frames_8bit else None

            # Formatear ítem para el frontend
            # App.tsx: interface LibraryItem
            item = {
                "job_id":       job_dir.name,
                "timestamp":    data.get("timestamp", ""),
                "model_name":   data.get("model", "Sin nombre"),
                "score":        data.get("score", 0),
                "score_bio":    data.get("score_bio", 0),
                "verdict":      data.get("verdict", "UNKNOWN"),
                "total_time":   data.get("total_time", 0),
                "n_frames":     len(frames_8bit),
                "n_bitplanes":  len(frames_1bit),
                "has_bitplanes": len(frames_1bit) > 0,
                "bitplane_params": data.get("bitplane_params", {}),
                "frame_sample": sample,
                "params":       data.get("params", {}),
                "orient":       data.get("orient", {}),
                "metrics":      data.get("metrics", {}),
                "penalties":    data.get("penalties", []),
                "robustness":   data.get("robustness"),
                "score_breakdown": data.get("score_breakdown"),
                "images": {
                    "reconstruction_slices": f"/job/{job_dir.name}/artifact/reconstruction_slices.png",
                    "layer_profile":         f"/job/{job_dir.name}/artifact/layer_profile.png",
                    "dose_histogram":        f"/job/{job_dir.name}/artifact/dose_histogram.png",
                },
                "all_frames":    [f"/file/{job_dir.name}/{fr.name}" for fr in frames_8bit],
                "all_bitplanes": [f"/file/{job_dir.name}/bitplanes/{fr.name}" for fr in frames_1bit],
                "frames_base":   f"/file/{job_dir.name}/",
                "bitplanes_base": f"/file/{job_dir.name}/bitplanes/",
                "summary":      data # Datos completos para la vista de detalle
            }
            items.append(item)
        except:
            continue
            
    return {"items": items}

@app.delete("/library/{job_id}")
async def delete_library_item(job_id: str):
    _check_job_id(job_id)
    job_dir = JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
        return {"status": "ok"}
    raise HTTPException(404, "No existe")

@app.post("/save/{job_id}")
async def save_job(job_id: str):
    _check_job_id(job_id)
    tmp_path = TMP_JOBS_DIR / job_id
    final_path = JOBS_DIR / job_id
    if tmp_path.exists():
        if final_path.exists(): shutil.rmtree(final_path)
        shutil.move(str(tmp_path), str(final_path))
        return {"status": "ok", "saved_path": str(final_path)}
    if final_path.exists():
        return {"status": "ok", "message": "Ya estaba guardado"}
    raise HTTPException(404, "No se encontró el trabajo en la zona temporal")

# --- ASSETS ---

@app.get("/export/{job_id}.json")
async def export_job(job_id: str):
    _check_job_id(job_id)
    path = JOBS_DIR / job_id / "result.json"
    if not path.exists():
        raise HTTPException(404, "JSON no encontrado")
    return FileResponse(path, filename=f"vam_export_{job_id}.json")

@app.get("/job/{job_id}/frame/{frame_idx}")
async def get_frame(job_id: str, frame_idx: int):
    # el pipeline numera desde frame_0000.png: frame_idx es 0-based
    path = _job_file(job_id, f"frame_{frame_idx:04d}.png")
    if path is None:
        raise HTTPException(404, "Imagen no encontrada")
    return FileResponse(path)

@app.get("/job/{job_id}/artifact/{name}")
async def get_artifact(job_id: str, name: str):
    path = _job_file(job_id, name)
    if path is None:
        raise HTTPException(404, "Artefacto no encontrado")
    return FileResponse(path)

@app.get("/file/{job_id}/{path:path}")
async def get_any_file(job_id: str, path: str):
    # Endpoint genérico para imágenes (incluye bitplanes/<archivo>). No permite salir de la carpeta del trabajo.
    full_path = _job_file(job_id, path)
    if full_path is None:
        raise HTTPException(404, "Archivo no encontrado")
    return FileResponse(full_path)

# ─── CM4 Proxy / Discovery ───────────────────────────────────────────────────

@app.get("/cm4/discover")
async def discover_cm4():
    """
    Intentar localizar la CM4 en la red local.
    Devuelve IP resuelta, hostname mDNS, y estado de conexión.
    Fuerza re-probe ignorando el caché de corta duración.
    """
    # BUG-06: forzar re-probe al descubrir explícitamente (BUG-07: import muerto eliminado)
    invalidate_resolved_ip_cache()
    try:
        ip = resolve_cm4_ip()
        _resolved_ip_cache["ip"] = ip
        import time; _resolved_ip_cache["ts"] = time.time()
        return {
            "found": True,
            "ip": ip,
            "hostname": CM4_HOSTNAME,
            "cached": _cm4_ip_cache.get("ip") == ip,
        }
    except Exception as e:
        cached_ip = _cm4_ip_cache.get("ip")
        err_msg = e.detail if isinstance(e, HTTPException) else str(e)
        return {
            "found": False,
            "ip": cached_ip,
            "hostname": CM4_HOSTNAME,
            "cached": bool(cached_ip),
            "error": err_msg,
        }

class SetIpBody(BaseModel):
    ip: str

@app.post("/cm4/set_ip")
async def set_cm4_ip(body: SetIpBody):
    """
    Forzar manualmente una IP para la CM4 si el descubrimiento falla.
    """
    _save_ip_cache(body.ip)
    return {"status": "ok", "ip": body.ip}


@app.get("/printer_status")
async def get_printer_status():
    import requests
    try:
        url = get_cm4_url("/estado_impresion")
        response = requests.get(url, timeout=1.5)
        if response.ok:
            return response.json()
        raise HTTPException(500, "Error del CM4")
    except HTTPException:
        raise
    except Exception:
        # BUG-06: invalidar caché de IP si la petición falla (fuerza re-probe)
        invalidate_resolved_ip_cache()
        raise HTTPException(503, "No se pudo alcanzar el CM4")



# Estado global del envío a impresora (BUG-01 fix)
_print_jobs: dict = {}   # job_id -> {"status": "sending"|"done"|"error", "error": str|None}

def _do_send_to_printer(job_id: str, job_dir: Path, power: float = 2.5, loops: int = 1):
    """Ejecutado en hilo separado para no bloquear el servidor (BUG-01 fix)."""
    import time, zipfile, io, requests
    t_start = time.perf_counter()
    _print_jobs[job_id] = {"status": "sending", "error": None}
    try:
        # 1. Empacar frames en ZIP
        t_zip = time.perf_counter()
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            bitplanes_dir = job_dir / "bitplanes"
            if bitplanes_dir.exists():
                for f in bitplanes_dir.glob("*.png"):
                    zf.write(f, f.name)
            else:
                for f in job_dir.glob("frame_*.png"):
                    zf.write(f, f.name)
            res_path = job_dir / "result.json"
            if res_path.exists():
                zf.write(res_path, res_path.name)
        zip_buffer.seek(0)
        zip_size_mb = len(zip_buffer.getvalue()) / (1024 * 1024)
        t_zip_end = time.perf_counter()
        print(f"[PC] ZIP creado: {zip_size_mb:.2f} MB en {t_zip_end - t_zip:.3f}s")

        # 2. Enviar a CM4 (timeout generoso pero no bloquea el servidor)
        PRINTER_URL = get_cm4_url("/iniciar_impresion")
        t_net = time.perf_counter()
        files = {'file': (f'{job_id}.zip', zip_buffer, 'application/zip')}
        
        # Mapeo temporal de mW/cm2 a PWM (0-1023). El usuario calibrará esto.
        # Asumimos temporalmente 1 mW/cm2 = 20 PWM
        pwm_val = int(power * 20)
        pwm_val = max(0, min(1023, pwm_val))

        post_data = {
            'motor': 'true',
            'led_pwm': str(pwm_val),
            'loops': str(loops)
        }
        
        response = requests.post(PRINTER_URL, data=post_data, files=files, timeout=300)
        t_net_end = time.perf_counter()

        if not response.ok:
            _print_jobs[job_id] = {"status": "error", "error": f"CM4 respondió con error: {response.text}"}
            return

        r_data = response.json()
        print(f"[PC] Envío completado en {t_net_end - t_net:.3f}s — {r_data.get('metrics', {})}")
        _print_jobs[job_id] = {
            "status": "done",
            "error": None,
            "metrics": {
                "zip_size_mb": round(zip_size_mb, 2),
                "zip_time_s": round(t_zip_end - t_zip, 3),
                "net_time_s": round(t_net_end - t_net, 3),
                "total_pc_s": round(time.perf_counter() - t_start, 3),
            }
        }
    except requests.exceptions.Timeout:
        invalidate_resolved_ip_cache()
        _print_jobs[job_id] = {"status": "error", "error": "Tiempo de espera agotado al conectar con CM4."}
    except requests.exceptions.ConnectionError:
        invalidate_resolved_ip_cache()
        _print_jobs[job_id] = {"status": "error", "error": "No se pudo alcanzar la CM4. Verifica el cable USB."}
    except Exception as e:
        import traceback; traceback.print_exc()
        _print_jobs[job_id] = {"status": "error", "error": str(e)}

@app.post("/print/{job_id}")
async def send_to_printer(job_id: str, power: float = 2.5, loops: int = 1):
    """
    Empaqueta el trabajo en un ZIP y lo envía al endpoint de la RPi CM4.
    BUG-01 fix: el envío corre en un hilo separado, el endpoint retorna inmediatamente.
    """
    _check_job_id(job_id)
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        job_dir = TMP_JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, "Trabajo no encontrado")

    # Lanzar en hilo sin bloquear FastAPI
    executor.submit(_do_send_to_printer, job_id, job_dir, power, loops)
    return {"status": "sending", "job_id": job_id}

@app.get("/print_send_status/{job_id}")
async def get_print_send_status(job_id: str):
    """Consultar el estado del envío del ZIP a la CM4."""
    state = _print_jobs.get(job_id)
    if not state:
        return {"status": "unknown"}
    return state

if __name__ == "__main__":
    import uvicorn
    # Bind a 127.0.0.1 en vez de 0.0.0.0 para no exponer la API a la red local
    uvicorn.run(app, host="127.0.0.1", port=8000)

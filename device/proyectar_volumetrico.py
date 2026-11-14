"""
proyectar_volumetrico.py — Proyectar secuencia de imagenes volumetricas
========================================================================
Ejecutar en la RPi CM4: sudo python3 proyectar_volumetrico.py

Carga las 360 imagenes del modelo, las proyecta en secuencia por el DMD
y sincroniza cada frame con un paso de 1° del motor de rotacion via Moonraker.

Pipeline por frame (paralelo):
  1. fast_swap()        — activa imagen en DMD (instantaneo)
  2. fast_expose_ms()   — inicia UV
  3. motor.paso(1°)     — envia comando al motor (async, no bloquea)
  4. fast_load(next)    — carga siguiente imagen via SPI (~8ms)
  5. wait_done()        — espera fin de exposicion UV
  → El motor ha girado 1° durante los pasos 3-5

ATENCION: Este script ENCIENDE el LED UV si led_pwm > 0.
  - Para visualizacion segura en pared/papel: led_pwm=50 (5%)
  - Para test sin UV real: led_pwm=0, MOTOR_HABILITADO=False
"""

import sys
import os
import time
import glob
import argparse
import numpy as np
from PIL import Image
import json

sys.path.insert(0, "/home/pi")

import RPi.GPIO as GPIO
import smbus
import spidev

GPIO.setmode(GPIO.BCM)

# Se usarán argumentos de línea de comandos en lugar de variables globales estáticas
# pero mantenemos variables por defecto por si no se especifica
DEFAULT_IMAGES_DIR = "/home/pi/imagenes"   # Carpeta con las 360 proyecciones
DMD_WIDTH = 2560
DMD_HEIGHT = 1440

TARGET_FPS = 6.0  # Apuntamos a 6 FPS estables para evitar stuttering


PIXEL_SIZE_MM = 0.020833   # ~20.8 um por pixel
TAM_IMAGEN_MM = 20.0       

PROJ_SIZE_PX = int(TAM_IMAGEN_MM / PIXEL_SIZE_MM)  # 960px

DEFAULT_LED_PWM = 50          
DEFAULT_EXPOSURE_MS = 80      
PAUSE_MS = 0          
NUM_LOOPS = 1         
ROTATION = 2          

GRADOS_POR_FRAME = 1.0     
VELOCIDAD_GRADOS_SEG = 3.0

# ============================================================

def load_single_image(f, target_size, rotation, crops):
    """Carga y procesa una sola imagen (para hilos)"""
    img = Image.open(f).convert('L')
    
    # Solo redimensionar si es estrictamente necesario
    w, h = img.size
    if target_size and (w != target_size or h != target_size):
        img = img.resize((target_size, target_size), Image.NEAREST)
    
    arr = np.array(img)
    
    # Rotar segun orientacion fisica del DMD
    if rotation > 0:
        arr = np.rot90(arr, rotation)
        
    # Aplicar recortes si el area activa es menor al frame
    c_y1, c_y2, c_x1, c_x2 = crops
    if c_y1 != 0 or c_x1 != 0 or c_y2 != arr.shape[0] or c_x2 != arr.shape[1]:
        arr = arr[c_y1:c_y2, c_x1:c_x2]
        
    # Transponer para el formato de bus del DMD (Column-major)
    return arr.T

def load_images(directory, target_size, rotation=0, status_file=None):
    """Cargar imagenes en paralelo para aprovechar los 4 cores de la CM4"""
    import concurrent.futures
    files = sorted(glob.glob(os.path.join(directory, "*.png")))
    if not files:
        print(f"ERROR: No se encontraron imagenes en {directory}")
        sys.exit(1)
        
    # Determinar parametros de la primera para todos
    test_img = Image.open(files[0])
    orig_w, orig_h = test_img.size
    
    # Si no se especifica target_size, usamos el original (Pass-through)
    if not target_size:
        target_size = orig_w

    x_off = (DMD_WIDTH - target_size) // 2
    y_off = (DMD_HEIGHT - target_size) // 2
    
    crops = (
        max(0, -y_off), 
        target_size - max(0, y_off + target_size - DMD_HEIGHT),
        max(0, -x_off),
        target_size - max(0, x_off + target_size - DMD_WIDTH)
    )
    
    final_x_off = max(0, x_off)
    final_y_off = max(0, y_off)
    
    print(f"Cargando {len(files)} imagenes (Paralelo 4 cores)...")
    print(f"  Target: {target_size}x{target_size} | Offset: ({final_x_off}, {final_y_off})")
    
    images = [None] * len(files)
    t_load = time.perf_counter()
    loaded_count = 0
    n_total = len(files)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {executor.submit(load_single_image, f, target_size, rotation, crops): i for i, f in enumerate(files)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            images[idx] = future.result()
            loaded_count += 1
            if loaded_count % 20 == 0 or loaded_count == n_total:
                print(f"  procesado... {loaded_count}/{n_total}")
                if status_file:
                    update_status_file(
                        status_file, "CARGANDO_MODELO",
                        loaded_count, n_total,
                        time.perf_counter() - t_load
                    )
            
    t_load_end = time.perf_counter()
    print(f"  {len(images)} imagenes listas en {t_load_end - t_load:.3f}s")
    return images, final_x_off, final_y_off

def update_status_file(file_path, status, progress, total_frames, elapsed, frame_time=0):
    if not file_path:
        return
    try:
        data = {
            "status": status,
            "progress_percent": int((progress / total_frames) * 100) if total_frames > 0 else 0,
            "current_frame": progress,
            "total_frames": total_frames,
            "elapsed_s": round(elapsed, 1),
            "frame_time_ms": round(frame_time, 1)
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass

# ============================================================

def auto_args():
    parser = argparse.ArgumentParser(description="Proyector Volumétrico FastDMD")
    parser.add_argument("--directorio", type=str, default=DEFAULT_IMAGES_DIR, help="Ruta a las imagenes PNG")
    parser.add_argument("--estado", type=str, default=None, help="Archivo JSON temporal para actualizar el estado (/tmp/estado.json)")
    parser.add_argument("--led_pwm", type=int, default=DEFAULT_LED_PWM, help="Potencia UV (0-1023)")
    parser.add_argument("--exposure_ms", type=int, default=DEFAULT_EXPOSURE_MS, help="Exposicion por frame en ms")
    parser.add_argument("--loops", type=int, default=NUM_LOOPS, help="Cantidad de vueltas completas de la impresión")
    parser.add_argument("--motor", action="store_true", help="Habilitar el motor de rotacion real")
    parser.add_argument("--auto", action="store_true", help="No esperar confirmacion de usuario, iniciar automaticamente")
    return parser.parse_args()

try:
    args = auto_args()
    
    # Sobrescribir variable global si se especificó
    NUM_LOOPS = args.loops

    update_status_file(args.estado, "INICIANDO", 0, 0, 0)
    
    from volumetric_controller import FastDMD
    
    if args.motor:
        try:
            from utilidades.motor_rotacion import MotorRotacion
        except ImportError:
            from motor_rotacion import MotorRotacion
    
    # Inicializar hardware
    i2c = smbus.SMBus(1)
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 125_000_000
    spi.mode = 3
    
    # Iniciar controlador DMD
    dmd = FastDMD(i2c, spi)
    dmd.init_volumetric_mode(led_pwm=args.led_pwm)

    # Iniciar motor de rotacion (via Moonraker/Klipper)
    motor = None
    if args.motor:
        motor = MotorRotacion()
        if not motor.test_conexion():
            print("  ATENCION: Moonraker no responde. Continuando SIN motor.")
            motor = None
        else:
            motor.habilitar()
            motor.set_modo_relativo()
            motor.home()

    # Contar frames disponibles para informar al UI antes de cargarlos
    _preview_files = sorted(glob.glob(os.path.join(args.directorio, "*.png")))
    n_preview = len(_preview_files)
    # Actualizar estado para telemetría (con total real)
    update_status_file(args.estado, "CARGANDO_MODELO", 0, max(n_preview, 1), 0, 0)

    # Cargar imagenes (con escalado y rotacion) — reporta progreso al archivo de estado
    images, X_OFFSET, Y_OFFSET = load_images(args.directorio, PROJ_SIZE_PX, ROTATION, status_file=args.estado)
    n_frames = len(images)
    
    print(f"\n{'='*60}")
    print(f"  PROYECCION VOLUMETRICA")
    print(f"{'='*60}")
    print(f"  Frames:      {n_frames}")
    print(f"  Imagen orig: {PROJ_SIZE_PX}x{PROJ_SIZE_PX}px = {TAM_IMAGEN_MM:.0f}x{TAM_IMAGEN_MM:.0f}mm")
    print(f"  Offset:      ({X_OFFSET}, {Y_OFFSET})")
    print(f"  LED PWM:     {args.led_pwm}/1023 ({args.led_pwm/1023*100:.1f}%)")
    print(f"  Exposicion:  {args.exposure_ms}ms/frame")
    print(f"  Rotacion:    {ROTATION * 90} grados")
    print(f"  Loops:       {NUM_LOOPS}")
    print(f"  Motor:       {'ACTIVADO (' + str(VELOCIDAD_GRADOS_SEG) + 'deg/s)' if motor else 'DESACTIVADO (solo proyeccion)'}")
    if args.led_pwm > 0:
        ms_por_frame = args.exposure_ms + 35  # exposicion + SPI overhead
        total_est = n_frames * NUM_LOOPS * ms_por_frame / 1000
        print(f"  Tiempo est:  {total_est:.1f}s ({total_est/60:.1f} min)")
        print(f"\n  ** LED UV ACTIVO — NO MIRAR DIRECTAMENTE **")
    print(f"{'='*60}")
    
    if not args.auto:
        input("\nPresiona ENTER para iniciar la proyeccion...")
    
    # ── Sincronización motor ──────────────────────────────────────────────────
    # La proyección corre a 6 FPS exactos (TARGET_FRAME_TIME = 1/6 s = 166.6 ms).
    # El bucle tarda: n_frames × (1/6) + t_preload (fast_load imagen 0, ~35 ms).
    # El motor debe cubrir total_grados en exactamente ese tiempo.
    #
    # Cálculo:
    #   total_grados   = n_frames × GRADOS_POR_FRAME         (= 360°)
    #   total_loop_s   = n_frames × (1/6) + PRELOAD_S        (≈ 60.035 s)
    #   vel_deg_s      = total_grados / total_loop_s          (≈ 5.997 °/s)
    #   fr (deg/min)   = vel_deg_s × 60                      (≈ 359.8 → 359)
    #
    # Se registra t_motor_start para poder esperar al motor tras el bucle.
    TARGET_FRAME_TIME = 1.0 / 6.0          # 166.6 ms — NO cambiar
    PRELOAD_S         = 0.035              # SPI fast_load imagen 0 antes del bucle

    total_grados   = n_frames * NUM_LOOPS * GRADOS_POR_FRAME
    total_loop_s   = n_frames * NUM_LOOPS * TARGET_FRAME_TIME + PRELOAD_S
    vel_deg_s      = total_grados / total_loop_s
    fr             = int(vel_deg_s * 60)

    t_motor_start = time.time()
    if motor:
        print(f"  Motor: {total_grados}° a {vel_deg_s:.4f}°/s (F{fr}) → duracion={total_loop_s:.3f}s")
        motor.paso(total_grados, feed_rate=fr)

    # Pre-cargar primera imagen (este tiempo ya está incluido en PRELOAD_S)
    dmd.fast_load(images[0], X_OFFSET, Y_OFFSET)
    
    t_start_all = time.time()
    for loop in range(NUM_LOOPS):
        t_loop_start = time.time()
        
        if NUM_LOOPS > 1:
            print(f"\n--- Loop {loop+1}/{NUM_LOOPS} ---")
        
        for i in range(n_frames):
            global_frame = loop * n_frames + i

            # 1. Activar imagen actual en el DMD
            dmd.fast_swap()

            # 2. Iniciar exposicion UV
            if args.exposure_ms > 0 and args.led_pwm > 0:
                dmd.fast_expose_ms(args.exposure_ms, dark_frames=0)

            # 3. El motor ya está girando de forma continua (fluida) en background.
            # No enviamos comandos G1 a trompicones aquí.

            # 4. Cargar siguiente imagen via SPI (paralelo con exposicion y motor)
            next_idx = (i + 1) % n_frames
            if i + 1 < n_frames or loop + 1 < NUM_LOOPS:
                dmd.fast_load(images[next_idx], X_OFFSET, Y_OFFSET)

            # 5. Esperar fin de exposicion UV
            if args.exposure_ms > 0 and args.led_pwm > 0:
                dmd.wait_exposure_done()

            # Calcular tiempo de espera usando tiempo ABSOLUTO desde el inicio global.
            # Garantiza sincronización milimétrica entre múltiples vueltas sin stuttering.
            target_frame_end = t_start_all + PRELOAD_S + (global_frame + 1) * TARGET_FRAME_TIME
            now = time.time()
            wait_time = target_frame_end - now
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Progreso
            if (i + 1) % 10 == 0 or (i + 1) == n_frames:
                dt_actual = time.time() - (target_frame_end - TARGET_FRAME_TIME) # Frame time aprox
                elapsed_total = time.time() - t_start_all
                
                # Actualizar log en pantalla menos frecuente
                if (i + 1) % 60 == 0 or (i + 1) == n_frames:
                    print(f"  Frame {i+1:3d}/{n_frames} | "
                          f"{dt_actual*1000:.1f}ms/frame (wait: {max(0, wait_time)*1000:.1f}ms) | "
                          f"{elapsed_total:.1f}s elapsed")
                          
                # Actualizar state file internamente (telemetría UI)
                update_status_file(args.estado, "PROYECTANDO", global_frame + 1, n_frames * NUM_LOOPS, elapsed_total, dt_actual * 1000)
        
        loop_time = time.time() - t_loop_start
        print(f"\n  Loop {loop+1} completado en {loop_time:.1f}s "
              f"({loop_time/n_frames*1000:.1f}ms/frame, "
              f"{n_frames/loop_time:.1f} FPS)")
    
    # Apagar LED
    dmd.fast_expose_stop()
    dmd.shutdown()

    if motor:
        # Esperar a que el motor complete los 360° antes de deshabilitarlo.
        # Si el bucle terminó antes de que el motor complete su recorrido,
        # dormimos el tiempo restante + 0.5 s de margen.
        elapsed_since_motor = time.time() - t_motor_start
        remaining = total_loop_s - elapsed_since_motor
        if remaining > 0:
            print(f"  Esperando al motor: {remaining:.2f}s restantes...")
            time.sleep(remaining + 0.5)
        motor.deshabilitar()

    update_status_file(args.estado, "FINALIZADO", n_frames * NUM_LOOPS, n_frames * NUM_LOOPS, time.time() - t_start_all, 0)
    print(f"\nProyeccion finalizada exitosamente.")
    print(f"Telemetria: {n_frames * NUM_LOOPS} frames en {time.time() - t_start_all:.1f}s")

except KeyboardInterrupt:
    print("\n\nInterrumpido por usuario. Apagando LED y motor...")
    try:
        dmd.fast_expose_stop()
        dmd.shutdown()
        if motor:
            motor.deshabilitar()
    except:
        pass

except Exception as e:
    print(f"\nERROR: {e}")
    if 'args' in locals():
        update_status_file(args.estado, "ERROR", 0, 0, 0)
    import traceback
    traceback.print_exc()
    try:
        dmd.fast_expose_stop()
        dmd.shutdown()
        if motor:
            motor.deshabilitar()
    except:
        pass

finally:
    pass  # NO hacemos GPIO.cleanup() para mantener PIN_PROJ_ON encendido y evitar el pantallazo de arranque en el siguiente print

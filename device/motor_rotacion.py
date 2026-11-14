"""
motor_rotacion.py — Control del motor de rotación via Moonraker/Klipper
========================================================================
Controla el motor NEMA 17 + TMC2209 de la Fly-Puppet a través de la
API HTTP de Moonraker. No requiere GPIO: toda la comunicación es REST.

SISTEMA DE TRANSMISIÓN:
  Polea motor:  20 dientes
  Polea vial:  100 dientes
  Reducción:    100/20 = 5:1 (5 vueltas del motor = 1 vuelta del vial)

CONFIGURACIÓN EN printer.cfg de Klipper:
-----------------------------------------
Añadir un eje de rotación configurado como extruder de rotación pura.
Con la reducción 5:1, rotation_distance = 360.0 / 5 = 72.0, lo que
hace que cada 1.0 "mm" de movimiento equivalga exactamente a 1.0° de
rotación del VIAL (Klipper compensa la reducción internamente).

  [extruder_stepper rotation_motor]       # NO usar [extruder] para no
  microsteps: 16                          # interferir con el extrusor real
  rotation_distance: 72.0                # 360/5 — 1 "mm" = 1° del vial
  step_pin: <pin_step_de_tu_placa>
  dir_pin: <pin_dir_de_tu_placa>
  enable_pin: !<pin_en_de_tu_placa>

  [tmc2209 extruder_stepper rotation_motor]
  uart_pin: <pin_uart_de_tu_placa>
  run_current: 0.6                        # NEMA17 típico: 0.4-0.8A
  stealthchop_threshold: 999999          # StealthChop siempre activo

NOTA: Si usas un eje lineal (X, Y, Z) en lugar de extruder_stepper,
      cambia AXIS_NAME y el G-code de movimiento correspondientemente.

RELACIÓN DE PASOS:
  NEMA17 estándar:      200 pasos/rev (1.8°/paso)
  TMC2209 microstepping: 16x por defecto en Klipper
  Pasos físicos/rev:    200 × 16 = 3200 pasos del motor por rev del motor
  Con reducción 5:1:    3200 × 5 = 16000 micropasos por rev del vial
  Resolución angular:   360° / 16000 = 0.0225°/micropaso del vial
  Con Klipper interpolation (256x): transparente al usuario
  → rotation_distance = 72.0 abstrae todo en grados del vial
"""

import requests
import time
import json

# ============================================================
# CONFIGURACION — Ajustar según la instalación
# ============================================================

MOONRAKER_URL = "http://localhost:7125"   # API Moonraker (local en RPi)
AXIS_NAME = "stepper_z"             # Nombre del stepper Z en printer.cfg

# Transmisión mecánica
POLEA_MOTOR  = 20     # Dientes de la polea del motor
POLEA_VIAL   = 100    # Dientes de la polea del vial
GEAR_RATIO   = POLEA_VIAL / POLEA_MOTOR  # 5.0 — reducción
# Con rotation_distance=72.0 en Klipper, G1 E1 = 1° del vial.
# La reducción da 16000 micropasos/rev del vial → resolución de 0.0225°/paso.

# Velocidad de rotación durante impresión (grados del VIAL por segundo)
# Con rotation_distance=72.0, 1 "mm" = 1° del vial.
# → En G-code: F = grados_por_segundo × 60 = mm/min
VELOCIDAD_GRADOS_SEG = 10   # 3°/s → un giro completo en 120s
                              # Ajustar: más rápido = menos tiempo, más inercia
VELOCIDAD_FEED_RATE = int(VELOCIDAD_GRADOS_SEG * 60)  # = F180 en G-code

# Timeout HTTP (segundos)
HTTP_TIMEOUT = 5.0

# ============================================================


class MotorRotacion:
    """
    Controlador de rotación para impresión volumétrica.
    Comunica con Klipper vía API Moonraker (HTTP REST).

    Uso típico en el ciclo de impresión:
        motor = MotorRotacion()
        motor.habilitar()
        motor.set_modo_relativo()

        for angulo in range(360):
            dmd.fast_swap()
            dmd.fast_expose_ms(80)
            motor.paso(1.0)          # Mover 1° mientras expone
            dmd.wait_exposure_done()

        motor.deshabilitar()
    """

    def __init__(self, url=MOONRAKER_URL, axis=AXIS_NAME):
        self.url = url
        self.axis = axis
        self._modo_relativo = False
        self._posicion_actual = 0.0  # grados acumulados

    # ----------------------------------------------------------
    # API interna: enviar G-code a Klipper via Moonraker
    # ----------------------------------------------------------

    def _gcode(self, script: str, timeout=HTTP_TIMEOUT) -> dict:
        """Enviar G-code a Klipper. Espera confirmación de ejecución."""
        try:
            r = requests.post(
                f"{self.url}/printer/gcode/script",
                json={"script": script},
                timeout=timeout
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"No se puede conectar a Moonraker en {self.url}. "
                f"¿Está Klipper activo? Verifica: curl {self.url}/printer/info"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Timeout al enviar G-code '{script}' a Moonraker")

    def _gcode_async(self, script: str) -> None:
        """Enviar G-code SIN esperar respuesta (fire-and-forget, más rápido)."""
        try:
            requests.post(
                f"{self.url}/printer/gcode/script",
                json={"script": script},
                timeout=HTTP_TIMEOUT
            )
        except Exception:
            pass  # En el bucle crítico, ignorar errores de red transitorios

    # ----------------------------------------------------------
    # Estado del sistema
    # ----------------------------------------------------------

    def estado_klipper(self) -> dict:
        """Consultar estado actual de Klipper."""
        try:
            r = requests.get(f"{self.url}/printer/info", timeout=HTTP_TIMEOUT)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def posicion_actual(self) -> float:
        """Consultar posición actual del motor en grados."""
        try:
            r = requests.get(
                f"{self.url}/printer/objects/query?motion_report",
                timeout=HTTP_TIMEOUT
            )
            data = r.json()
            # live_position puede estar disponible según configuración
            return self._posicion_actual
        except Exception:
            return self._posicion_actual

    # ----------------------------------------------------------
    # Control de movimiento
    # ----------------------------------------------------------

    def habilitar(self) -> None:
        """Activar el motor (SET_STEPPER_ENABLE)."""
        print(f"[Motor] Habilitando {self.axis}...")
        self._gcode(f"SET_STEPPER_ENABLE STEPPER={self.axis} ENABLE=1")
        time.sleep(0.2)  # Dar tiempo al driver para activarse

    def deshabilitar(self) -> None:
        """Desactivar el motor para que no consuma corriente en reposo."""
        print(f"[Motor] Deshabilitando {self.axis}...")
        self._gcode(f"SET_STEPPER_ENABLE STEPPER={self.axis} ENABLE=0")

    def set_modo_relativo(self) -> None:
        """Activar movimientos relativos (M83 para extruder, G91 para ejes)."""
        self._gcode("G91")  # Ejes en modo relativo (Z)
        self._modo_relativo = True

    def set_modo_absoluto(self) -> None:
        """Activar movimientos absolutos."""
        self._gcode("G90")  # Ejes en modo absoluto (Z)
        self._modo_relativo = False

    def home(self) -> None:
        """Enviar motor a posición 0 (requiere endstop o sea posición virtual)."""
        # Para un eje de rotación continua, el 'home' es simplemente
        # resetear la posición lógica a 0
        self._gcode("SET_KINEMATIC_POSITION Z=0")
        self._gcode("G92 Z0")  # Resetear posición del eje Z a 0
        self._posicion_actual = 0.0
        print(f"[Motor] Posición reseteada a 0° (Z homed virtualmente)")

    def paso(self, grados: float = 1.0, feed_rate: int = None) -> None:
        """
        Mover el motor exactamente 'grados' grados.

        NOTA: Ambas llamadas son fire-and-forget (_gcode_async) para no
        bloquear el hot loop de proyeccion. Klipper encola los comandos
        internamente y los ejecuta en orden, por lo que la secuencia
        G92→G1 se respeta aunque Python no espere respuesta.

        Args:
            grados:    Grados a mover (positivo = sentido horario segun config)
            feed_rate: Velocidad en mm/min (None = usar VELOCIDAD_FEED_RATE)
        """
        fr = feed_rate or VELOCIDAD_FEED_RATE
        # G92 Z0: resetear coordenada logica sin frenar el motor.
        # Async para no bloquear el ciclo de proyeccion (era la causa del jitter).
        self._gcode_async("G92 Z0")
        self._gcode_async(f"G1 Z{grados:.4f} F{fr}")
        self._posicion_actual += grados

    def paso_sync(self, grados: float = 1.0, feed_rate: int = None) -> None:
        """
        Mover el motor y ESPERAR a que complete el movimiento.
        Más seguro pero más lento. Útil para verificación.
        """
        fr = feed_rate or VELOCIDAD_FEED_RATE
        self._gcode("G92 Z0")  # Resetear coordenada sin frenar
        self._gcode(f"G1 Z{grados:.4f} F{fr}")
        # M400: esperar a que el buffer de movimientos esté vacío
        self._gcode("M400")
        self._posicion_actual += grados

    def girar_completo(self, angulos_total: int = 360,
                       velocidad: float = None) -> None:
        """
        Girar angulos_total grados a velocidad constante (modo continuo).
        Útil para verificación y calibración mecánica.
        """
        vel = velocidad or VELOCIDAD_GRADOS_SEG
        fr = int(vel * 60)
        print(f"[Motor] Girando {angulos_total}° a {vel}°/s (~{angulos_total/vel:.1f}s)...")
        self.set_modo_relativo()
        self._gcode("G92 Z0")  # Resetear coordenada sin frenar
        self._gcode(f"G1 Z{angulos_total:.1f} F{fr}")
        self._gcode("M400")  # Esperar fin
        self._posicion_actual += angulos_total
        print(f"[Motor] Giro completado. Posición: {self._posicion_actual:.1f}°")

    # ----------------------------------------------------------
    # Calibración de pasos
    # ----------------------------------------------------------

    def calibrar_pasos(self, grados_nominales: float, grados_reales: float,
                       rotation_distance_actual: float = 72.0) -> float:
        """
        Calcular el rotation_distance correcto basándose en medición real.

        El rotation_distance nominal con reducción 5:1 es 72.0
        (= 360° / gear_ratio). Ajustar si el vial no gira exactamente
        los grados esperados.

        Args:
            grados_nominales:       Los grados del vial que pediste mover
            grados_reales:          Los grados del vial que realmente se movió (medido)
            rotation_distance_actual: El rotation_distance configurado ahora (72.0)

        Returns:
            El nuevo rotation_distance a poner en printer.cfg

        Ejemplo:
            Si pides 90° del vial y gira 87.3°:
            nuevo_rd = calibrar_pasos(90, 87.3, 72.0) → 69.84
            Poner rotation_distance: 69.84 en printer.cfg
        """
        factor = grados_reales / grados_nominales
        nuevo_rd = rotation_distance_actual * factor
        print(f"\n[Calibración de pasos]")
        print(f"  Transmisión: {POLEA_MOTOR}T:{POLEA_VIAL}T (reducción {GEAR_RATIO:.1f}:1)")
        print(f"  Pedido (vial):   {grados_nominales}°")
        print(f"  Real (vial):     {grados_reales}°")
        print(f"  Factor:          {factor:.6f}")
        print(f"  rotation_distance actual:  {rotation_distance_actual:.4f}")
        print(f"  rotation_distance NUEVO:   {nuevo_rd:.4f}")
        print(f"\n  → En printer.cfg, cambiar:")
        print(f"    rotation_distance: {rotation_distance_actual:.4f}")
        print(f"  por:")
        print(f"    rotation_distance: {nuevo_rd:.4f}")
        return nuevo_rd

    # ----------------------------------------------------------
    # Test de conectividad
    # ----------------------------------------------------------

    def test_conexion(self) -> bool:
        """Verificar conexión con Moonraker y estado de Klipper."""
        print(f"[Motor] Verificando conexión con Moonraker en {self.url}...")
        estado = self.estado_klipper()
        if "error" in estado:
            print(f"  ❌ Error: {estado['error']}")
            return False
        estado_klipper = estado.get("result", {}).get("state", "desconocido")
        print(f"  ✅ Moonraker OK — Klipper: {estado_klipper}")
        return True


# ============================================================
# TEST STANDALONE — Ejecutar directamente en la RPi para verificar
# ============================================================
if __name__ == "__main__":
    import sys

    motor = MotorRotacion()

    print("=" * 55)
    print("  TEST DE MOTOR DE ROTACIÓN — Volumetric DLP")
    print("=" * 55)

    # 1. Verificar conexión
    if not motor.test_conexion():
        print("\n  ¿Está Klipper activo? Prueba: systemctl status klipper")
        sys.exit(1)

    # 2. Preparar motor
    motor.habilitar()
    motor.set_modo_relativo()
    motor.home()

    print("\n  Selecciona test:")
    print("  1) Giro lento completo (360°, ~120s)")
    print("  2) Giro rápido completo (360°, ~36s)")
    print("  3) Test 4 cuartos (4 × 90°)")
    print("  4) Test paso a paso manual (1° × N)")
    opcion = input("\n  Opción [1-4]: ").strip()

    if opcion == "1":
        motor.girar_completo(360, velocidad=3.0)

    elif opcion == "2":
        motor.girar_completo(360, velocidad=10.0)

    elif opcion == "3":
        for i in range(4):
            print(f"\n  Cuarto {i+1}/4 — girando 90°...")
            motor.paso_sync(90.0)
            time.sleep(1.0)

    elif opcion == "4":
        n = int(input("  ¿Cuántos pasos de 1°? "))
        print(f"  Ejecutando {n} pasos de 1°...")
        motor.set_modo_relativo()
        for i in range(n):
            motor.paso_sync(1.0)
            print(f"    Paso {i+1}/{n} — pos: {motor._posicion_actual:.1f}°")
            time.sleep(0.1)
    else:
        print("  Opción no válida")

    motor.deshabilitar()
    print(f"\n  Test completado. Posición final: {motor._posicion_actual:.1f}°")

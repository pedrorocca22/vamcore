"""
wifi_manager.py — Gestión de red WiFi para la RPi CM4
=======================================================
Usa nmcli (NetworkManager) para:
  - Escanear redes disponibles
  - Conectarse a una red (con contraseña)
  - Reportar el estado de la conexión actual
"""

import subprocess
import json
import re
import socket


def _run(cmd: list[str], timeout: int = 15) -> tuple[str, str, int]:
    """Ejecutar un comando y devolver (stdout, stderr, returncode)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout", 1
    except Exception as e:
        return "", str(e), 1


def scan_networks() -> list[dict]:
    """
    Escanear redes WiFi disponibles con nmcli.
    Devuelve lista de dicts ordenada por señal descendente.
    """
    # Forzar re-scan
    _run(["sudo", "nmcli", "dev", "wifi", "rescan"])

    # Listar redes
    stdout, stderr, rc = _run([
        "nmcli", "-f", "SSID,SIGNAL,SECURITY,IN-USE",
        "dev", "wifi", "list"
    ])

    networks = []
    seen_ssids = set()

    for line in stdout.splitlines()[1:]:  # Saltar cabecera
        # Parsear columnas (alineadas por espacios variables)
        # Formato: SSID                SIGNAL  SECURITY    IN-USE
        parts = line.split()
        if len(parts) < 2:
            continue

        in_use = parts[-1] == "*"
        # Si hay "*", las columnas están desplazadas
        if in_use:
            parts = parts[:-1]

        # El SIGNAL siempre es numérico — buscamos desde la derecha
        # SECURITY puede ser "--", "WPA2", "WPA1 WPA2", etc.
        # SSID puede tener espacios
        try:
            signal = int(parts[-2])
            security = parts[-1]
            ssid = " ".join(parts[:-2]).strip()
        except (ValueError, IndexError):
            continue

        if not ssid or ssid == "--":
            continue
        if ssid in seen_ssids:
            continue
        seen_ssids.add(ssid)

        networks.append({
            "ssid": ssid,
            "signal": signal,
            "security": security if security != "--" else "Open",
            "in_use": in_use,
        })

    # Ordenar por señal descendente, red activa al frente
    networks.sort(key=lambda n: (not n["in_use"], -n["signal"]))
    return networks


def get_current_connection() -> dict:
    """
    Estado de la conexión WiFi actual.
    Devuelve info de IP, SSID, señal.
    """
    # SSID activo
    stdout, _, rc = _run(["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"])
    current_ssid = None
    for line in stdout.splitlines():
        if line.startswith("yes:"):
            current_ssid = line.split(":", 1)[1]
            break

    # IP del interfaz wlan0
    ip_addr = None
    stdout, _, _ = _run(["ip", "-4", "addr", "show", "wlan0"])
    match = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", stdout)
    if match:
        ip_addr = match.group(1)

    # Hostname
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"

    # Estado del dispositivo
    stdout, _, _ = _run([
        "nmcli", "-t", "-f", "DEVICE,STATE,CONNECTION",
        "dev", "status"
    ])
    wifi_state = "disconnected"
    for line in stdout.splitlines():
        if "wlan0" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                wifi_state = parts[1]

    return {
        "connected": current_ssid is not None and ip_addr is not None,
        "ssid": current_ssid,
        "ip": ip_addr,
        "hostname": hostname,
        "mdns": f"{hostname}.local",
        "state": wifi_state,
    }


def connect_to_network(ssid: str, password: str = "") -> dict:
    """
    Conectarse a una red WiFi.
    Si la red ya existe en el keyring de NM, se reconecta sin password.
    Devuelve dict con {success, message}.
    """
    # Primero intentar conectar a una conexión guardada
    stdout, _, _ = _run(["nmcli", "-t", "-f", "NAME", "connection", "show"])
    saved = [line.strip() for line in stdout.splitlines()]

    if ssid in saved:
        # Reconectar sin contraseña
        _, stderr, rc = _run([
            "sudo", "nmcli", "connection", "up", ssid
        ])
        if rc == 0:
            status = get_current_connection()
            return {"success": True, "message": f"Reconectado a {ssid}", "status": status}
        # Si falla, probablemente la contraseña cambió — continuar con el flow normal

    # Conectar con contraseña (nueva red)
    if password:
        cmd = [
            "sudo", "nmcli", "dev", "wifi", "connect", ssid,
            "password", password
        ]
    else:
        cmd = ["sudo", "nmcli", "dev", "wifi", "connect", ssid]

    stdout, stderr, rc = _run(cmd, timeout=30)

    if rc == 0:
        status = get_current_connection()
        return {"success": True, "message": f"Conectado a {ssid}", "status": status}
    else:
        # Limpiar mensaje de error de nmcli
        msg = stderr or stdout
        msg = msg.replace("Error: ", "").strip()
        return {"success": False, "message": msg, "status": get_current_connection()}


def disconnect() -> dict:
    """Desconectar del WiFi actual sin apagar la interfaz."""
    # Encontrar la conexión activa en wlan0
    stdout, _, _ = _run(["nmcli", "-t", "-f", "NAME,DEVICE", "connection", "show", "--active"])
    for line in stdout.splitlines():
        if ":wlan0" in line:
            conn_name = line.split(":")[0]
            _, stderr, rc = _run(["sudo", "nmcli", "connection", "down", conn_name])
            return {"success": rc == 0, "message": f"Desconectado de {conn_name}" if rc == 0 else stderr}
    return {"success": False, "message": "No hay conexión activa"}


def forget_network(ssid: str) -> dict:
    """Eliminar una red guardada del keyring."""
    _, _, rc = _run(["sudo", "nmcli", "connection", "delete", ssid])
    return {"success": rc == 0}

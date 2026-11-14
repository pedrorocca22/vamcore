"""
upload_to_projector.py
================
Sube los archivos del servidor a la CM4 por SSH y opcionalmente
configura el modo Ethernet-over-USB (g_ether) para una conexión
directa por cable USB-C, sin depender del WiFi.

Uso:
    python upload_to_projector.py              # solo sube archivos
    python upload_to_projector.py --usb-setup  # configura USB Gadget + reinicia CM4
"""

import paramiko
import os
import sys
import time
import socket

# ─── Conexión ────────────────────────────────────────────────────────────────

USB_IP      = "10.55.0.2"       # IP fija tras configurar USB Gadget
HOSTNAME    = "dlp3.local"      # mDNS fallback
FALLBACK_IP = "192.168.1.140"   # IP WiFi fallback

import json

config_path = os.path.join(os.path.dirname(__file__), "config.json")
try:
    with open(config_path, "r") as f:
        cfg = json.load(f)
except Exception:
    cfg = {"cm4_username": "pi", "cm4_password": "pi"}

USERNAME = cfg.get("cm4_username", "pi")
PASSWORD = cfg.get("cm4_password", "pi")

# ─── Archivos a subir ─────────────────────────────────────────────────────────

LOCAL_DIR  = os.path.join(os.path.dirname(__file__), "device")
REMOTE_DIR = "/home/pi"

FILES_TO_UPLOAD = [
    "ejemplo_servidor_subproceso.py",
    "proyectar_volumetrico.py",
    "volumetric_controller.py",
    "wifi_manager.py",
    "motor_rotacion.py",
]

# ─── Resolución de host ───────────────────────────────────────────────────────

def resolve_host() -> str:
    """
    Prueba activamente qué IP tiene la CM4 con el puerto 22 (SSH) abierto.
    Orden: USB directo → mDNS → fallback WiFi
    """
    candidates = []

    # 1. USB directo (si ya está configurado)
    candidates.append(USB_IP)

    # 2. mDNS (WiFi actual)
    try:
        info = socket.getaddrinfo(HOSTNAME, None, socket.AF_INET)
        ip = info[0][4][0]
        if ip not in candidates:
            candidates.append(ip)
    except Exception:
        pass

    # 3. Fallback WiFi conocido
    if FALLBACK_IP not in candidates:
        candidates.append(FALLBACK_IP)

    print("Buscando CM4 en la red...")
    for ip in candidates:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            result = s.connect_ex((ip, 22))
            s.close()
            if result == 0:
                print(f"  [OK] CM4 encontrada en {ip}")
                return ip
        except Exception:
            pass

    print(f"  [ERR] No se encontró la CM4. Usando fallback: {FALLBACK_IP}")
    return FALLBACK_IP


def ssh_connect(ip: str) -> paramiko.SSHClient:
    """Conectar por SSH a la CM4."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, username=USERNAME, password=PASSWORD, timeout=15)
    return ssh


def run_cmd(ssh: paramiko.SSHClient, cmd: str, desc: str = "", timeout: int = 30) -> str:
    """Ejecutar comando SSH e imprimir resultado."""
    if desc:
        print(f"  [{desc}]", end=" ", flush=True)
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    result = out or err
    if desc:
        print("OK" if stdout.channel.recv_exit_status() == 0 else "WARN")
    return result


# ─── Configuración USB Gadget ─────────────────────────────────────────────────

def configure_usb_gadget(ssh: paramiko.SSHClient):
    """
    Configura el CM4 para exponer una interfaz de red virtual por el
    cable USB-C (modo g_ether). Tras esto, la CM4 tendrá siempre la
    IP 10.55.0.2 cuando esté conectada al PC por USB, independientemente
    del WiFi.

    Cambios que aplica:
      1. /boot/config.txt   → activa dwc2 en modo peripheral
      2. /boot/cmdline.txt  → carga módulos dwc2 y g_ether al arranque
      3. systemd-networkd   → IP estática 10.55.0.2 en usb0
      4. dnsmasq            → DHCP: asigna 10.55.0.1 al PC automáticamente
    """
    print("\n=== Configurando USB Ethernet Gadget (g_ether) ===")

    # ── 1. Detectar ruta del boot partition ──────────────────────────────────
    boot_dir = run_cmd(ssh,
        'ls /boot/firmware/config.txt > /dev/null 2>&1 && echo /boot/firmware || echo /boot',
        desc="Detectando ruta de boot"
    ).strip()
    print(f"     → Boot partition: {boot_dir}")

    config_path  = f"{boot_dir}/config.txt"
    cmdline_path = f"{boot_dir}/cmdline.txt"

    # ── 2. Parchear config.txt ────────────────────────────────────────────────
    # Primero eliminar cualquier línea dwc2 existente para no duplicar
    run_cmd(ssh,
        f"sudo sed -i '/dtoverlay=dwc2/d' {config_path}",
        desc="Limpiando dwc2 antiguo en config.txt"
    )
    # Añadir la directiva correcta al final
    run_cmd(ssh,
        f"echo 'dtoverlay=dwc2,dr_mode=peripheral' | sudo tee -a {config_path}",
        desc="Activando dwc2 modo peripheral en config.txt"
    )

    # ── 3. Parchear cmdline.txt ───────────────────────────────────────────────
    # Leer el contenido actual
    current_cmdline = run_cmd(ssh, f"cat {cmdline_path}").strip()

    if "modules-load=dwc2,g_ether" in current_cmdline:
        print("  [cmdline.txt] Ya configurado, sin cambios.")
    else:
        # Insertar ANTES de "rootwait" para garantizar orden de carga
        new_cmdline = current_cmdline.replace(
            "rootwait",
            "modules-load=dwc2,g_ether rootwait"
        )
        # Escribir de una sola vez (debe ser una línea)
        escaped = new_cmdline.replace("'", "'\"'\"'")
        run_cmd(ssh,
            f"echo '{escaped}' | sudo tee {cmdline_path}",
            desc="Añadiendo g_ether a cmdline.txt"
        )

    # ── 4. Configurar IP estática en usb0 via systemd-networkd ───────────────
    network_conf = (
        "[Match]\\n"
        "Name=usb0\\n"
        "\\n"
        "[Network]\\n"
        "Address=10.55.0.2/24\\n"
        "DHCPServer=yes\\n"    # <-- La CM4 actúa como DHCP server para el PC
        "\\n"
        "[DHCPServer]\\n"
        "PoolOffset=100\\n"    # Asigna 10.55.0.100+ al PC
        "PoolSize=1\\n"
        "EmitDNS=no\\n"
    )
    run_cmd(ssh,
        f'printf "{network_conf}" | sudo tee /etc/systemd/network/10-usb0.network',
        desc="Creando config de red usb0"
    )

    # ── 5. Habilitar systemd-networkd ─────────────────────────────────────────
    run_cmd(ssh,
        "sudo systemctl enable systemd-networkd",
        desc="Habilitando systemd-networkd"
    )

    # ── 6. Verificar que NetworkManager no pisa usb0 ─────────────────────────
    # Decirle a NetworkManager que ignore usb0
    nm_conf = (
        "[keyfile]\\n"
        "unmanaged-devices=interface-name:usb0\\n"
    )
    run_cmd(ssh,
        f'printf "{nm_conf}" | sudo tee /etc/NetworkManager/conf.d/99-usb0-unmanaged.conf',
        desc="Excluyendo usb0 de NetworkManager"
    )

    print("\n  [OK] Configuración USB Gadget aplicada correctamente.")
    print("  [INFO] La CM4 necesita reiniciarse para activar los cambios.")

    respuesta = input("\n¿Reiniciar la CM4 ahora? [S/n]: ").strip().lower()
    if respuesta in ("", "s", "si", "y", "yes"):
        print("  Reiniciando CM4...")
        try:
            ssh.exec_command("sudo reboot")
        except Exception:
            pass  # Normal que la conexión se corte al reiniciar
        print("  CM4 reiniciando. Espera ~30 segundos y luego conecta el cable USB-C.")
    else:
        print("  Reinicio cancelado. Recuerda reiniciar manualmente para activar g_ether.")


# ─── Subida de archivos ───────────────────────────────────────────────────────

def upload_files(ssh: paramiko.SSHClient):
    """Sube los archivos Python del servidor a la CM4."""
    print("\n=== Subiendo archivos del servidor ===")
    sftp = ssh.open_sftp()

    for filename in FILES_TO_UPLOAD:
        local_path  = os.path.join(LOCAL_DIR, filename)
        remote_path = f"{REMOTE_DIR}/{filename}"

        if os.path.exists(local_path):
            sftp.put(local_path, remote_path)
            print(f"  [OK] {filename}")
        else:
            print(f"  [ERR] No encontrado localmente: {local_path}")

    sftp.close()


def restart_server(ssh: paramiko.SSHClient):
    """Mata el servidor viejo y lanza uno nuevo en background."""
    print("\n=== Reiniciando servidor API en la CM4 ===")

    print("  Matando servidor anterior...")
    ssh.exec_command("pkill -f 'uvicorn ejemplo_servidor_subproceso'")
    time.sleep(1.5)

    print("  Instalando dependencias...")
    _, out, _ = ssh.exec_command(
        "sudo python3 -m pip install -q fastapi uvicorn python-multipart",
        timeout=60
    )
    out.read()  # esperar

    cmd = (
        f"cd {REMOTE_DIR} && "
        "nohup python3 -m uvicorn ejemplo_servidor_subproceso:app "
        "--host 0.0.0.0 --port 8000 > api_log.txt 2>&1 &"
    )
    ssh.exec_command(cmd)
    time.sleep(2)

    print("  [OK] Servidor API arrancado (puerto 8000)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    setup_usb = "--usb-setup" in sys.argv

    # Conectar
    target_ip = resolve_host()
    print(f"Conectando por SSH a {target_ip}...")
    try:
        ssh = ssh_connect(target_ip)
    except Exception as e:
        print(f"✗ Error de conexión: {e}")
        sys.exit(1)
    print("Conexión SSH establecida.\n")

    try:
        if setup_usb:
            configure_usb_gadget(ssh)
            # Tras el setup el SSH se cierra por el reboot, no hacemos más
        else:
            upload_files(ssh)
            restart_server(ssh)
            print("\n[OK] Todo subido y servidor corriendo.")
            print(f"  API disponible en: http://{target_ip}:8000")
    finally:
        try:
            ssh.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

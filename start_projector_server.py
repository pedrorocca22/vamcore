"""
Arranca el servidor en la CM4 via SSH en background y verifica que quede corriendo.
"""
import paramiko, time, json, os

config_path = os.path.join(os.path.dirname(__file__), "config.json")
try:
    with open(config_path, "r") as f:
        cfg = json.load(f)
except Exception:
    cfg = {"cm4_username": "pi", "cm4_password": "pi"}

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("10.55.0.2", username=cfg.get("cm4_username", "pi"), password=cfg.get("cm4_password", "pi"), timeout=5)
print("SSH OK - arrancando servidor...")

# Matar cualquier proceso previo en puerto 8000
client.exec_command("fuser -k 8000/tcp 2>/dev/null; sleep 1")
time.sleep(1.5)

# Arrancar uvicorn en background (nohup para que no muera al cerrar SSH)
_, stdout, stderr = client.exec_command(
    "cd /home/pi && nohup python -m uvicorn ejemplo_servidor_subproceso:app "
    "--host 0.0.0.0 --port 8000 > /tmp/vam_server.log 2>&1 &"
)
time.sleep(3)

# Verificar que arrancó
_, stdout, _ = client.exec_command("ss -tlnp | grep 8000")
out = stdout.read().decode()
if "8000" in out:
    print("[OK] Servidor arrancado correctamente en puerto 8000")
else:
    print("[ERROR] Servidor NO arranco. Log:")
    _, stdout, _ = client.exec_command("tail -30 /tmp/vam_server.log")
    print(stdout.read().decode())

client.close()

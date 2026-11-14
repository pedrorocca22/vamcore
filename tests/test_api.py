"""Pruebas de la API (server.py) sin GPU: rutas endurecidas y endpoints del vial.

server.py se importa desde una COPIA en una carpeta temporal: al arrancar borra tmp_jobs/ junto a sí mismo,
y no debe tocar los trabajos reales del repositorio. vamcore.pipeline se sustituye por un módulo vacío.

Ejecutar desde la raíz:  python -m pytest tests/test_api.py -q
"""
import importlib.util
import shutil
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    shutil.copy(ROOT / "server.py", tmp_path / "server.py")
    monkeypatch.setitem(sys.modules, "vamcore.pipeline", types.ModuleType("vamcore.pipeline"))
    spec = importlib.util.spec_from_file_location("server_under_test", tmp_path / "server.py")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    job = server.JOBS_DIR / "abc12345"
    (job / "bitplanes").mkdir(parents=True)
    (job / "frame_0000.png").write_bytes(b"frame0")
    (job / "bitplanes" / "bit_0000.png").write_bytes(b"bit0")
    (tmp_path / "secreto.txt").write_text("NO DEBE LEERSE")
    return TestClient(server.app), server, tmp_path


def test_serves_job_files(client):
    c, _, _ = client
    assert c.get("/file/abc12345/frame_0000.png").content == b"frame0"
    assert c.get("/file/abc12345/bitplanes/bit_0000.png").content == b"bit0"
    assert c.get("/job/abc12345/frame/0").content == b"frame0"         # índice 0-based


@pytest.mark.parametrize("url", [
    "/file/abc12345/..%2F..%2Fsecreto.txt",
    "/file/abc12345/%2E%2E/%2E%2E/secreto.txt",
    "/job/abc12345/artifact/..%2F..%2Fsecreto.txt",
])
def test_path_traversal_is_blocked(client, url):
    c, _, _ = client
    r = c.get(url)
    assert r.status_code in (400, 404) and b"NO DEBE" not in r.content


def test_job_id_validation(client):
    c, server, tmp = client
    assert c.delete("/library/%2E%2E").status_code == 400
    assert c.get("/status/a.b").status_code == 400
    assert (tmp / "server.py").exists() and (tmp / "secreto.txt").exists()


def test_vial_preview_and_presets(client):
    c, _, _ = client
    assert "borosilicate" in c.get("/vial/presets").json()["vial_materials"]
    r = c.get("/vial/preview", params=dict(outer_diameter_mm=20, wall_thickness_mm=1,
                                           material="borosilicate", resin="gelma")).json()
    assert r["printable_diameter_mm"] == pytest.approx(2 * 10 / 1.35, rel=1e-6)
    assert c.get("/vial/preview", params=dict(outer_diameter_mm=20, wall_thickness_mm=12)).status_code == 400
    assert c.get("/vial/preview", params=dict(outer_diameter_mm=20, material="oro")).status_code == 400


def test_optimize_rejects_bad_parameters_before_running(client):
    c, _, _ = client
    f = {"file": ("m.stl", b"solid x")}
    assert c.post("/optimize", files=f, data={"vial_outer_diameter_mm": "20", "vial_material": "oro"}).status_code == 400
    assert c.post("/optimize", files=f, data={"stop_ver_mask": "magia"}).status_code == 400
    assert c.post("/optimize", files=f, data={"vial_mode": "magia"}).status_code == 400
    assert c.post("/optimize", files=f, data={"absorption_cm": "-1"}).status_code == 400
    assert c.post("/optimize", files=f, data={"voxel_pitch_um": "0.1"}).status_code == 400
    assert c.post("/optimize", files=f, data={"method": "BCLP"}).status_code == 400


def test_orient_endpoint(client):
    trimesh = pytest.importorskip("trimesh")
    import numpy as np

    c, _server, _ = client
    cyl = trimesh.creation.cylinder(radius=1.0, height=12.0, sections=48)
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.radians(35), [1, 0, 0]))
    stl = cyl.export(file_type="stl")
    r = c.post("/orient", files={"file": ("c.stl", stl, "application/octet-stream")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["feasible"] and abs(abs(d["rx"]) - 35) < 0.5 and abs(d["ry"]) < 0.5
    assert d["radius_limit_source"] == "build_area" and d["limits"]["max_height_mm"] == 25.0
    # con vial: el límite de radio pasa a ser la zona imprimible
    r = c.post("/orient", files={"file": ("c.stl", stl, "application/octet-stream")},
               data={"vial_outer_diameter_mm": "20", "vial_wall_thickness_mm": "1"})
    assert r.status_code == 200 and r.json()["radius_limit_source"] == "vial"
    assert c.post("/orient", files={"file": ("c.txt", b"x", "text/plain")}).status_code == 400

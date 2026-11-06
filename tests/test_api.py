"""Tests de l'API du Studio sans CARLA (FastAPI TestClient). Nécessite fastapi et httpx."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from fastapi.testclient import TestClient

import app.main as m


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "RUNS", tmp_path)
    monkeypatch.setattr(m, "procs", {})
    return TestClient(m.app)


def fake_aeb_run(root, rid="r1"):
    d = root / rid
    d.mkdir()
    (d / "status.json").write_text(json.dumps({"state": "finished", "error": None}))
    (d / "config.json").write_text(json.dumps({"kind": "aeb", "scenario": "stopped"}))
    (d / "summary.json").write_text(json.dumps({"collided": False, "min_gap_m": 9.1, "max_state": "PARTIAL_BRAKING"}))
    (d / "log.csv").write_text("t,ego_v,lead_v,gap_true,n_det,n_tracks,meas_range,ttc,a_req,state,brake\n"
                               "0.05,13.9,0,60,1,0,,,0.0,NORMAL,0.0\n0.1,13.8,0,59,1,1,59,4.2,inf,WARNING,0.0\n")


def test_liste_et_log_d_un_run_aeb(client, tmp_path):
    fake_aeb_run(tmp_path)
    r = client.get("/api/runs").json()
    assert r[0]["kind"] == "aeb" and r[0]["collided"] is False and r[0]["min_gap_m"] == 9.1
    log = client.get("/api/runs/r1/log").json()
    assert log["state"] == ["NORMAL", "WARNING"] and log["a_req"] == [0.0, None]


def test_lancement_refuse_sans_carla(client, monkeypatch):
    monkeypatch.setattr(m, "carla_status", lambda: {"reachable": False})
    assert client.post("/api/runs", json={"kind": "aeb"}).status_code == 503


def test_parametres_invalides(client):
    assert client.post("/api/runs", json={"kind": "aeb", "scenario": "x"}).status_code == 422
    assert client.post("/api/runs", json={"kind": "aeb", "v0": 500}).status_code == 422


def test_commande_du_scenario_aeb(client, monkeypatch):
    monkeypatch.setattr(m, "carla_status", lambda: {"reachable": True})
    cap = {}

    class P:
        def poll(self):
            return None

    monkeypatch.setattr(m.subprocess, "Popen", lambda cmd, **k: (cap.setdefault("cmd", cmd), P())[1])
    r = client.post("/api/runs", json={"kind": "aeb", "scenario": "braking", "gap": 30, "aeb": False})
    assert r.status_code == 200
    cmd = cap["cmd"]
    assert cmd[1].endswith("11_aeb_closed_loop.py") and "--no-aeb" in cmd
    assert cmd[cmd.index("--scenario") + 1] == "braking" and cmd[cmd.index("--run-id") + 1] == r.json()["run_id"]

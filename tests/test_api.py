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


def test_websocket_envoie_chaque_image_puis_le_final(client, tmp_path):
    d = tmp_path / "r2"
    (d / "frames").mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": "finished", "error": None}))
    (d / "summary.json").write_text(json.dumps({"collided": False}))
    (d / "config.json").write_text(json.dumps({"kind": "aeb"}))
    recs = [{"idx": i, "img": f"{i:05d}.jpg", "t": 0.1 * i, "state": "NORMAL"} for i in range(5)]
    (d / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    for r in recs:
        (d / "frames" / r["img"]).write_bytes(b"jpg" + bytes([r["idx"]]))
    with client.websocket_connect("/ws/runs/r2") as ws:
        got = []
        while True:
            m = ws.receive_json()
            if m.get("final"):
                break
            got.append(m)
    assert [g["frame"]["idx"] for g in got] == [0, 1, 2, 3, 4]
    assert all(g["image"] for g in got) and m["summary"] == {"collided": False}


def test_commande_aeb_transmet_frequence_et_defauts(client, monkeypatch):
    monkeypatch.setattr(m, "carla_status", lambda: {"reachable": True})
    cap = {}

    class P:
        def poll(self):
            return None

    monkeypatch.setattr(m.subprocess, "Popen", lambda cmd, **k: (cap.setdefault("cmd", cmd), P())[1])
    r = client.post("/api/runs", json={"kind": "aeb", "perception_hz": 5, "ghost_prob": 0.01, "ghost_len": 3})
    assert r.status_code == 200
    c = cap["cmd"]
    assert c[c.index("--perception-hz") + 1] == "5" and c[c.index("--ghost-len") + 1] == "3"
    assert client.post("/api/runs", json={"kind": "aeb", "perception_hz": 7}).status_code == 422

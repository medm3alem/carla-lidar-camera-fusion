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


# ---------------------------------------------------------------- campagnes
ROWS = [
    {"case": "a", "v0": "50", "aeb": "True", "collided": "False", "min_gap_m": "7.6", "gap_brake": "34.5", "ttc_brake": "2.5", "error": ""},
    {"case": "b", "v0": "50", "aeb": "True", "collided": "False", "min_gap_m": "7.7", "gap_brake": "34.0", "ttc_brake": "2.4", "error": ""},
    {"case": "c", "v0": "70", "aeb": "True", "collided": "True", "impact_kmh": "16.3", "min_gap_m": "0.0", "error": ""},
    {"case": "d", "v0": "70", "aeb": "True", "collided": "False", "min_gap_m": "3.3", "error": ""},
    {"case": "e", "v0": "70", "aeb": "False", "collided": "True", "impact_kmh": "60", "min_gap_m": "0.0", "error": ""},
    {"case": "f", "v0": "30", "aeb": "True", "error": "pas de summary.json"},
]


def test_aggregate_dangereux_mediane_pire_cas_et_erreurs():
    a = m.aggregate(ROWS, "stopped")
    assert a["hazard"] and a["n_errors"] == 1
    l50, l70 = a["lines"]
    assert abs(l50["gap_median"] - 7.65) < 0.06 and l50["gap_worst"] == 7.6   # médiane de 2 valeurs = moyenne
    assert l70["n"] == 2 and l70["collisions"] == 1 and l70["impact_mean"] == 16.3 and l70["gap_worst"] == 0.0


def test_aggregate_non_dangereux_compte_les_freinages():
    rows = [{"v0": "50", "aeb": "True", "n_brake_ticks": "0", "max_state": "NORMAL", "error": ""},
            {"v0": "50", "aeb": "True", "n_brake_ticks": "4", "max_state": "EMERGENCY_BRAKING", "error": ""}]
    l = m.aggregate(rows, "adjacent")["lines"][0]
    assert l["brakes"] == 1 and l["warnings"] == 1 and l["n"] == 2


def _no_process(monkeypatch):
    """Aucun test ne doit lancer de vrai processus (une campagne lancerait des essais CARLA sur la machine)."""
    def boom(*a, **k):
        raise AssertionError("subprocess.Popen appelé dans un test")
    monkeypatch.setattr(m.subprocess, "Popen", boom)


def test_campagne_refusee_sans_carla_ou_trop_grande(client, monkeypatch, tmp_path):
    _no_process(monkeypatch)
    monkeypatch.setattr(m, "CAMPAIGNS", tmp_path / "c")
    monkeypatch.setattr(m, "carla_status", lambda: {"reachable": False})
    assert client.post("/api/campaigns", json={"scenario": "stopped"}).status_code == 503
    big = {"speeds": [30, 40, 50, 60, 70], "weathers": ["clear", "rain", "fog"], "seeds": [1, 2, 3, 4, 5]}
    assert client.post("/api/campaigns", json=big).status_code == 422
    assert client.post("/api/campaigns", json={"scenario": "nope"}).status_code == 422
    assert client.post("/api/campaigns", json={"speeds": [200]}).status_code == 422


def test_commande_de_campagne(client, monkeypatch, tmp_path):
    captured = {}

    class FakeProc:
        returncode = None

        def poll(self):
            return None

    def fake_popen(cmd, **k):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(m.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(m, "CAMPAIGNS", tmp_path)
    monkeypatch.setattr(m, "camp_procs", {})
    monkeypatch.setattr(m, "carla_status", lambda: {"reachable": True})
    r = client.post("/api/campaigns", json={"scenario": "adjacent", "speeds": [50], "seeds": [43], "perception_hz": 5})
    assert r.status_code == 200
    cmd = captured["cmd"]
    assert cmd[cmd.index("--scenario") + 1] == "adjacent" and "--skip-baseline" in cmd
    assert "--aeb-opts=--perception-hz 5" in cmd and cmd[cmd.index("--resume") + 1] == r.json()["id"]
    assert client.post("/api/campaigns", json={}).status_code == 409        # une campagne est déjà active


def test_suivi_d_une_campagne(client, monkeypatch, tmp_path):
    monkeypatch.setattr(m, "CAMPAIGNS", tmp_path)
    monkeypatch.setattr(m, "camp_procs", {})
    d = tmp_path / "c1"
    d.mkdir()
    (d / "studio_config.json").write_text(json.dumps({"scenario": "stopped", "speeds": [50, 70], "weathers": ["clear"],
                                                      "seeds": [42, 43], "skip_baseline": True}))
    (d / "results.csv").write_text("case,v0,aeb,collided,min_gap_m,error\nx,50,True,False,7.6,\ny,50,True,False,7.7,\n")
    g = client.get("/api/campaigns/c1").json()
    assert g["done"] == 2 and g["total"] == 4 and g["state"] == "interrupted"
    assert g["aggregate"]["lines"][0]["n"] == 2
    assert client.get("/api/campaigns").json()[0]["id"] == "c1"
    assert client.get("/api/campaigns/inconnue").status_code == 404



# ---------------------------------------------------------------- agent (langage naturel -> scénario)
def fake_llm(reply, monkeypatch):
    monkeypatch.setattr(m, "agent_model", lambda: ("llm", "test", "faux-modele", lambda text: reply))


GOOD = {"scenario": "stopped", "ego_speed_kmh": 50, "gap_m": 50, "weather": "clear", "seeds": [42, 43]}


def test_agent_mode_regles_sans_cle(client, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("AGENT_PROVIDER", "groq")
    assert client.get("/api/agent").json()["mode"] == "rules"


def test_agent_scenario_valide(client, monkeypatch):
    fake_llm(json.dumps(GOOD), monkeypatch)
    x = client.post("/api/agent/parse", json={"text": "voiture arrêtée à 50 m, ego à 50"}).json()
    assert x["ok"] and x["scenario"]["scenario"] == "stopped" and x["scenario"]["seeds"] == [42, 43]
    assert x["need_m"] == 80 and x["mode"] == "llm"


def test_agent_refus_avec_raison(client, monkeypatch):
    fake_llm('{"reject": "Town07 n\'est pas gérée"}', monkeypatch)
    x = client.post("/api/agent/parse", json={"text": "test sur Town07"}).json()
    assert not x["ok"] and x["kind"] == "rejet" and "Town07" in x["message"]


def test_agent_json_illisible(client, monkeypatch):
    fake_llm("désolé, je ne sais pas", monkeypatch)
    x = client.post("/api/agent/parse", json={"text": "bonjour"}).json()
    assert not x["ok"] and x["kind"] == "json_invalide"


def test_agent_hors_plage_et_infaisable(client, monkeypatch):
    fake_llm(json.dumps({**GOOD, "ego_speed_kmh": 150}), monkeypatch)
    x = client.post("/api/agent/parse", json={"text": "ego à 150"}).json()
    assert not x["ok"] and x["kind"] == "invalide"
    fake_llm(json.dumps({**GOOD, "gap_m": 80, "town": "Town10HD_Opt"}), monkeypatch)       # 80 + 30 = 110 m > 95 m
    x = client.post("/api/agent/parse", json={"text": "écart 80 m"}).json()
    assert not x["ok"] and x["kind"] == "infaisable" and "95" in x["message"]


def test_agent_erreur_reseau_et_aucun_lancement(client, monkeypatch):
    def boom(text):
        raise RuntimeError("quota dépassé")
    monkeypatch.setattr(m, "agent_model", lambda: ("llm", "test", "x", boom))
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: pytest.fail("l'agent ne doit rien lancer"))
    r = client.post("/api/agent/parse", json={"text": "voiture arrêtée"})
    assert r.status_code == 502 and "quota" in r.json()["detail"]


def test_agent_regles_comprennent_une_phrase_simple(client, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    x = client.post("/api/agent/parse", json={"text": "Voiture arrêtée à 50 m, ego à 50 km/h"}).json()
    assert x["mode"] == "rules" and x["ok"] and x["scenario"]["gap_m"] == 50


def test_campagne_avec_carte_et_defauts(client, monkeypatch, tmp_path):
    captured = {}

    class FakeProc:
        def poll(self):
            return None

    monkeypatch.setattr(m.subprocess, "Popen", lambda cmd, **k: captured.update(cmd=cmd) or FakeProc())
    monkeypatch.setattr(m, "CAMPAIGNS", tmp_path)
    monkeypatch.setattr(m, "camp_procs", {})
    monkeypatch.setattr(m, "carla_status", lambda: {"reachable": True})
    assert client.post("/api/campaigns", json={"town": "Mars"}).status_code == 422
    r = client.post("/api/campaigns", json={"town": "Town03", "gap": 30, "speeds": [50], "seeds": [42]})
    assert r.status_code == 200 and captured["cmd"][captured["cmd"].index("--town") + 1] == "Town03"

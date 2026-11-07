"""Tests de src/fusion/agent (schéma, faisabilité, métriques, jeu de test) : sans CARLA, sans LLM."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fusion.agent import check, normalize, to_studio_payload, validate
from fusion.agent.metrics import classify, evaluate, parse
from fusion.agent.rules import parse_text

ITEMS = [json.loads(line) for line in (ROOT / "reference" / "agent_benchmark.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def test_les_reponses_de_reference_sont_valides_et_realisables():
    for it in ITEMS:
        if it["gold"] is not None:
            n, fmt, infeasible = check(it["gold"])
            assert not fmt and not infeasible, (it["id"], fmt, infeasible)


def test_les_demandes_infaisables_sont_detectees_par_le_validateur():
    # JSON qu'un modèle docile produirait pour chaque demande infaisable du jeu de test : le validateur doit l'écarter
    literal = [{"scenario": "stopped", "ego_speed_kmh": 50, "gap_m": 90, "town": "Town03"},      # 90 m > plage + 80 m de route
               {"scenario": "stopped", "ego_speed_kmh": 200},
               {"scenario": "follow", "ego_speed_kmh": 50, "gap_m": 70, "town": "Town10HD_Opt"},
               {"scenario": "stopped", "ego_speed_kmh": 50, "town": "Town99"},
               {"scenario": "stopped", "ego_speed_kmh": 5}]
    assert sum(1 for it in ITEMS if it.get("reject_reason") == "infaisable") == len(literal)
    for d in literal:
        n, fmt, infeasible = check(d)
        assert fmt or infeasible, d


def test_faisabilite_geometrique_sans_erreur_de_format():
    n, fmt, infeasible = check({"scenario": "follow", "ego_speed_kmh": 50, "gap_m": 70})
    assert not fmt and infeasible and "146" in infeasible[0]
    n, fmt, infeasible = check({"scenario": "stopped", "ego_speed_kmh": 30, "town": "Town03", "gap_m": 50})
    assert not fmt and not infeasible                      # 80 m nécessaires sur 80 m : limite acceptée


def test_defauts_et_champs_inconnus():
    n = normalize({"scenario": "braking", "ego_speed_kmh": 50})
    assert n["gap_m"] == 45.0 and n["lead_speed_kmh"] == 30.0 and n["lead_brake_t_s"] == 1.0 and n["seeds"] == [42]
    assert normalize({"scenario": "follow", "ego_speed_kmh": 40})["lead_speed_kmh"] == 40
    assert any("inconnu" in e for e in validate(normalize({"scenario": "stopped", "ego_speed_kmh": 50, "couleur": "rouge"})))
    assert any("faults" in e for e in validate(normalize({"scenario": "stopped", "ego_speed_kmh": 50, "faults": {"x": 1}})))


def test_payload_accepte_par_l_api_du_studio():
    pytest.importorskip("fastapi")
    sys.path.insert(0, str(ROOT / "backend"))
    import app.main as m
    for it in ITEMS:
        if it["gold"] is None:
            continue
        n = check(it["gold"])[0]
        for many in (False, True):
            url, body = to_studio_payload(n, many=many)
            (m.CampaignConfig if many else m.RunConfig)(**body)       # lève si un champ est refusé par l'API
        assert to_studio_payload(n)[0] == ("/api/campaigns" if len(n["seeds"]) > 1 else "/api/runs")


def test_classification_des_erreurs():
    it = {"id": "x", "category": "c", "gold": {"scenario": "stopped", "ego_speed_kmh": 50}}
    assert classify(it, {"scenario": "stopped", "ego_speed_kmh": 50})[0]
    assert classify(it, "pas du json") == (False, ["json_invalide"])
    assert classify(it, {"reject": "vague"}) == (False, ["rejet_à_tort"])
    assert classify(it, {"scenario": "braking", "ego_speed_kmh": 50})[1] == ["mauvais_scénario"]
    assert classify(it, {"scenario": "stopped", "ego_speed_kmh": 13.9})[1] == ["erreur_unité"]       # 50 / 3,6
    assert classify(it, {"scenario": "stopped", "ego_speed_kmh": 180})[1] == ["hors_plage"]        # 50 x 3,6 sort de la plage
    assert classify(it, {"scenario": "stopped", "ego_speed_kmh": 50, "x": 1})[1] == ["champ_inconnu"]
    amb = {"id": "a", "category": "c", "gold": None, "reject_reason": "ambigu"}
    assert classify(amb, {"reject": "vague"})[0] and not classify(amb, {"scenario": "stopped", "ego_speed_kmh": 50})[0]


def test_parse_sortie_avec_bloc_de_code():
    d, st = parse('Voici :\n```json\n{"scenario": "stopped", "ego_speed_kmh": 50}\n```')
    assert st == "ok" and d["ego_speed_kmh"] == 50


def test_banc_d_evaluation_oracle_et_baseline():
    gold = {it["text"]: it["gold"] for it in ITEMS}
    assert evaluate(ITEMS, lambda t: gold[t])["accuracy"] == 1.0
    rep = evaluate(ITEMS, parse_text)
    assert 0.5 <= rep["accuracy"] < 1.0                    # baseline : utile mais loin du parfait (mesuré : 23/35)
    assert rep["correct"] == 23


# ---------------------------------------------------------------- client « compatible OpenAI » (serveur local factice)
def _serve(responses):
    """Serveur HTTP local qui renvoie successivement les (code, corps) donnés ; retourne (url de base, requêtes reçues)."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, self.headers.get("Authorization"), json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                         self.headers.get("User-Agent")))
            code, body = responses.pop(0)
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}/v1", seen, srv


def test_client_openai_reprise_apres_429_et_raisonnement_retire():
    from fusion.agent.llm import make_chat
    ok = {"choices": [{"message": {"content": '<think>je réfléchis</think>{"scenario": "stopped", "ego_speed_kmh": 50}'}}]}
    base, seen, srv = _serve([(429, {"error": "limite"}), (200, ok)])
    try:
        out = make_chat(base, "m", "clé", "sys", retries=2)("Cible arrêtée à 50 km/h")
    finally:
        srv.shutdown()
    assert json.loads(out) == {"scenario": "stopped", "ego_speed_kmh": 50}
    assert len(seen) == 2 and seen[0][0] == "/v1/chat/completions" and seen[0][1] == "Bearer clé"
    assert seen[0][2]["messages"][0] == {"role": "system", "content": "sys"} and seen[0][2]["temperature"] == 0
    assert seen[0][3] and "python-urllib" not in seen[0][3].lower()      # sinon Cloudflare répond 403 (code 1010)


def test_client_openai_erreur_http_remontee():
    from fusion.agent.llm import make_chat
    base, seen, srv = _serve([(401, {"error": "clé invalide"})])
    try:
        with pytest.raises(RuntimeError, match="HTTP 401"):
            make_chat(base, "m", "mauvaise")("x")
    finally:
        srv.shutdown()


def test_parse_ignore_les_blocs_de_raisonnement():
    d, st = parse('<think>{"scenario": "braking"}</think>{"scenario": "stopped", "ego_speed_kmh": 50}')
    assert st == "ok" and d["scenario"] == "stopped"

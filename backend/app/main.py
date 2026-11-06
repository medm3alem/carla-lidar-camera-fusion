"""
API du Studio (FastAPI). N'importe pas carla : la simulation tourne dans backend/worker.py,
lancé en sous-processus. Un seul run actif à la fois (une seule instance CARLA/GPU).

Lancer : python -m uvicorn app.main:app --app-dir backend --port 8000
Puis ouvrir http://localhost:8000
"""
import asyncio
import base64
import json
import csv
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "studio"
WORKER = ROOT / "backend" / "worker.py"
AEB_SCRIPT = ROOT / "scripts" / "11_aeb_closed_loop.py"
CAMP_SCRIPT = ROOT / "scripts" / "12_aeb_campaign.py"
CAMPAIGNS = ROOT / "runs" / "aeb_campaign"
MAX_CAMPAIGN_TRIALS = 60
SCENARIOS = ["stopped", "braking", "adjacent", "follow", "approach"]
STATIC = ROOT / "backend" / "static"
TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD_Opt"]
WEATHERS = ["clear", "rain", "fog"]
CARLA_HOST, CARLA_PORT = "localhost", 2000
MAX_BACKLOG = 20          # images au plus envoyées d'un coup à un client en retard

app = FastAPI(title="ADAS Validation Studio")
procs: dict = {}          # run_id -> Popen
camp_procs: dict = {}     # id de campagne -> Popen


class RunConfig(BaseModel):
    town: str = "Town10HD_Opt"
    weather: str = "clear"
    n_npc: int = Field(70, ge=0, le=150)
    seconds: float = Field(30.0, gt=0, le=300)
    seed: int = 42
    model: str = "finetuned"            # finetuned | coco
    conf: float = Field(0.25, ge=0.01, le=0.99)
    # kind="aeb" : scénario scripté en boucle fermée (scripts/11_aeb_closed_loop.py, 20 Hz, sans trafic)
    kind: str = "observe"               # observe | aeb
    scenario: str = "stopped"
    v0: float = Field(50.0, ge=10, le=90)
    gap: float = Field(60.0, ge=10, le=80)
    lead_speed: float = Field(50.0, ge=5, le=90)
    lead_brake_t: float = Field(3.0, ge=0, le=20)
    aeb: bool = True
    perception_hz: int = 20             # 20 | 10 | 5 : fréquence de perception + tracker + AEB
    ghost_prob: float = Field(0.0, ge=0.0, le=0.2)
    ghost_len: int = Field(1, ge=1, le=30)
    drop_prob: float = Field(0.0, ge=0.0, le=0.9)
    noise_sigma: float = Field(0.0, ge=0.0, le=2.0)


def active_run():
    for rid, p in procs.items():
        if p.poll() is None:
            return rid
    for cid, p in camp_procs.items():
        if p.poll() is None:
            return "campagne " + cid
    return None


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def run_dir(run_id: str) -> Path:
    d = RUNS / run_id
    if "/" in run_id or "\\" in run_id or ".." in run_id or not d.is_dir():
        raise HTTPException(404, "run inconnu")
    return d


def run_status(run_id: str) -> dict:
    d = run_dir(run_id)
    st = read_json(d / "status.json") or {"state": "starting", "error": None}
    st.setdefault("n_frames", 0)
    p = procs.get(run_id)
    if p is not None and p.poll() is not None and st["state"] in ("starting", "running"):
        st = {**st, "state": "error", "error": f"worker arrêté (code {p.returncode}), voir worker.log"}
    elif p is None and st["state"] in ("starting", "running"):
        st = {**st, "state": "error", "error": "run interrompu (serveur redémarré ?)"}
    return st


@app.get("/api/carla")
def carla_status():
    try:
        with socket.create_connection((CARLA_HOST, CARLA_PORT), timeout=1.0):
            return {"reachable": True, "host": CARLA_HOST, "port": CARLA_PORT}
    except OSError:
        return {"reachable": False, "host": CARLA_HOST, "port": CARLA_PORT}


@app.get("/api/options")
def options():
    return {"towns": TOWNS, "weathers": WEATHERS, "models": ["finetuned", "coco"], "scenarios": SCENARIOS}


@app.post("/api/runs")
def start_run(cfg: RunConfig):
    if (cfg.town not in TOWNS or cfg.weather not in WEATHERS or cfg.model not in ("finetuned", "coco")
            or cfg.kind not in ("observe", "aeb") or cfg.scenario not in SCENARIOS
            or cfg.perception_hz not in (20, 10, 5)):
        raise HTTPException(422, "paramètre invalide")
    busy = active_run()
    if busy:
        raise HTTPException(409, f"un run est déjà actif : {busy}")
    if not carla_status()["reachable"]:
        raise HTTPException(503, "serveur CARLA injoignable sur localhost:2000")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    if cfg.kind == "aeb":
        cmd = [sys.executable, str(AEB_SCRIPT), "--run-id", run_id, "--out", str(RUNS),
               "--scenario", cfg.scenario, "--v0", str(cfg.v0), "--gap", str(cfg.gap),
               "--lead-speed", str(cfg.lead_speed), "--lead-brake-t", str(cfg.lead_brake_t),
               "--town", cfg.town, "--weather", cfg.weather, "--seed", str(cfg.seed), "--conf", str(cfg.conf)]
        cmd += ["--perception-hz", str(cfg.perception_hz), "--ghost-prob", str(cfg.ghost_prob),
                "--ghost-len", str(cfg.ghost_len), "--drop-prob", str(cfg.drop_prob),
                "--noise-sigma", str(cfg.noise_sigma)]
        if not cfg.aeb:
            cmd.append("--no-aeb")
    else:
        cmd = [sys.executable, str(WORKER), "--run-id", run_id, "--town", cfg.town,
               "--weather", cfg.weather, "--n-npc", str(cfg.n_npc), "--seconds", str(cfg.seconds),
               "--seed", str(cfg.seed), "--conf", str(cfg.conf), "--every", "2"]   # 10 Hz : vidéo plus fluide
        if cfg.model == "coco":
            cmd += ["--coco", "--weights", str(ROOT / "weights" / "yolo26n.pt")]
        (d / "config.json").write_text(json.dumps({**(cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()), "run_id": run_id}))
    if cfg.kind == "aeb":
        (d / "config.json").write_text(json.dumps({**(cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()), "run_id": run_id}))
    log = open(d / "worker.log", "w")
    procs[run_id] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return {"run_id": run_id}


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    d = run_dir(run_id)
    (d / "STOP").write_text("stop")
    return {"ok": True}


@app.get("/api/runs")
def list_runs():
    out = []
    if RUNS.is_dir():
        for d in sorted(RUNS.iterdir(), reverse=True):
            if d.is_dir():
                st = run_status(d.name)
                s = read_json(d / "summary.json") or {}
                cfg_ = read_json(d / "config.json") or {}
                out.append({"run_id": d.name, "state": st["state"], "n_frames": st["n_frames"],
                            "kind": cfg_.get("kind", "observe"),
                            "n_detections": s.get("n_detections"), "n_with_lidar": s.get("n_with_lidar"),
                            "collided": s.get("collided"), "min_gap_m": s.get("min_gap_m"),
                            "max_state": s.get("max_state"), "config": cfg_})
    return out


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    d = run_dir(run_id)
    return {"run_id": run_id, "status": run_status(run_id),
            "summary": read_json(d / "summary.json"), "config": read_json(d / "config.json")}


@app.get("/api/runs/{run_id}/log")
def aeb_log(run_id: str):
    """log.csv d'un run AEB (une ligne par tick, 20 Hz) sous forme de colonnes."""
    import csv
    p = run_dir(run_id) / "log.csv"
    if not p.exists():
        raise HTTPException(404, "pas de log (run non terminé ou run d'observation)")
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in ("t", "ego_v", "lead_v", "gap_true", "a_req", "brake"):
        cols[k] = [float(r[k]) if r[k] not in ("", "inf") else None for r in rows]
    cols["state"] = [r["state"] for r in rows]
    return cols


@app.get("/api/runs/{run_id}/latest.jpg")
def latest_image(run_id: str):
    p = run_dir(run_id) / "latest.jpg"
    if not p.exists():
        raise HTTPException(404, "pas encore d'image")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/runs/{run_id}/frames")
def frames(run_id: str, start: int = 0, limit: int = 500):
    p = run_dir(run_id) / "frames.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return rows[start:start + min(limit, 2000)]


def read_new_lines(path: Path, offset: int):
    """Lignes JSON complètes ajoutées à `path` depuis `offset` ; renvoie (enregistrements, nouvel offset)."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return [], offset
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    recs = []
    for line in data[:end].splitlines():
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass
    return recs, offset + end + 1


def frame_image(d: Path, rec: dict):
    name = rec.get("img") or f"{rec.get('idx', 0):05d}.jpg"
    for p in (d / "frames" / name, d / "latest.jpg"):
        try:
            return base64.b64encode(p.read_bytes()).decode()
        except OSError:
            continue
    return None


@app.websocket("/ws/runs/{run_id}")
async def stream(ws: WebSocket, run_id: str):
    """Envoie CHAQUE image enregistrée (pas seulement la dernière) pour une vidéo fluide ; le client les
    rejoue avec une petite mémoire tampon. Si le client prend du retard, on saute les plus anciennes."""
    await ws.accept()
    try:
        d = run_dir(run_id)
    except HTTPException:
        await ws.close(code=4404)
        return
    offset = 0
    try:
        while True:
            st = run_status(run_id)
            recs, offset = read_new_lines(d / "frames.jsonl", offset)
            for rec in recs[-MAX_BACKLOG:]:
                await ws.send_json({"status": st, "frame": rec, "image": frame_image(d, rec)})
            if not recs and st["state"] not in ("starting", "running"):
                await ws.send_json({"status": st, "frame": None, "image": None, "final": True,
                                    "summary": read_json(d / "summary.json")})
                break
            await asyncio.sleep(0.04)
    except Exception:       # noqa: BLE001 : client parti
        pass


# ---------------------------------------------------------------- campagnes (scripts/12_aeb_campaign.py)
class CampaignConfig(BaseModel):
    scenario: str = "stopped"
    speeds: list[float] = [30.0, 50.0, 70.0]
    weathers: list[str] = ["clear"]
    seeds: list[int] = [42, 43]
    gap: float = Field(50.0, ge=10, le=80)
    lead_speed: float = Field(30.0, ge=5, le=90)
    same_speed: bool = False
    lead_brake_t: float = Field(3.0, ge=0, le=20)
    skip_baseline: bool = True          # ne pas relancer les témoins sans AEB (moitié moins d'essais)
    perception_hz: int = 20
    ghost_prob: float = Field(0.0, ge=0.0, le=0.2)
    ghost_len: int = Field(1, ge=1, le=30)
    drop_prob: float = Field(0.0, ge=0.0, le=0.9)
    noise_sigma: float = Field(0.0, ge=0.0, le=2.0)


def campaign_total(cfg: dict) -> int:
    return len(cfg["speeds"]) * len(cfg["weathers"]) * len(cfg["seeds"]) * (1 if cfg.get("skip_baseline") else 2)


def _truthy(x):
    return str(x) in ("True", "true", "1")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def aggregate(rows: list, scenario: str) -> dict:
    """Tableau agrégé par vitesse (essais AEB seulement, les témoins sont comptés à part). Même définitions que
    scripts/12_aeb_campaign.py : médiane statistique, écart minimal le plus faible = pire cas."""
    ok = [r for r in rows if r.get("error") in ("", None)]
    n_err = len(rows) - len(ok)
    hazard = scenario in ("stopped", "braking")
    out = []
    for v0 in sorted({float(r["v0"]) for r in ok}):
        rs = [r for r in ok if float(r["v0"]) == v0 and _truthy(r.get("aeb"))]
        if not rs:
            continue
        line = {"v0": v0, "n": len(rs)}
        if hazard:
            gaps = [g for g in (_num(r.get("min_gap_m")) for r in rs) if g is not None]
            imp = [_num(r.get("impact_kmh")) for r in rs if _truthy(r.get("collided"))]
            gb = [g for g in (_num(r.get("gap_brake")) for r in rs) if g is not None]
            tb = [g for g in (_num(r.get("ttc_brake")) for r in rs) if g is not None]
            line.update(collisions=sum(_truthy(r.get("collided")) for r in rs),
                        impact_mean=round(statistics.mean(i for i in imp if i is not None), 1) if imp else 0.0,
                        gap_median=round(statistics.median(gaps), 1) if gaps else None,
                        gap_worst=round(min(gaps), 1) if gaps else None,
                        gap_brake=round(statistics.median(gb), 1) if gb else None,
                        ttc_brake=round(statistics.median(tb), 1) if tb else None)
        else:
            line.update(brakes=sum((_num(r.get("n_brake_ticks")) or 0) > 0 for r in rs),
                        warnings=sum(r.get("max_state") not in ("NORMAL", "", None) for r in rs))
        out.append(line)
    return {"hazard": hazard, "lines": out, "n_errors": n_err}


def camp_dir(cid: str) -> Path:
    d = CAMPAIGNS / cid
    if "/" in cid or "\\" in cid or ".." in cid or not d.is_dir():
        raise HTTPException(404, "campagne inconnue")
    return d


def read_rows(d: Path) -> list:
    try:
        with open(d / "results.csv", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def campaign_info(cid: str) -> dict:
    d = camp_dir(cid)
    cfg = read_json(d / "studio_config.json") or {}
    rows = read_rows(d)
    total = campaign_total(cfg) if cfg.get("speeds") else None
    p = camp_procs.get(cid)
    if p is not None and p.poll() is None:
        state = "running"
    elif p is not None:
        state = "finished" if p.returncode == 0 and total is not None and len(rows) >= total else "error"
    else:
        state = "finished" if total is not None and len(rows) >= total else "interrupted"
    return {"id": cid, "state": state, "done": len(rows), "total": total, "config": cfg}


@app.post("/api/campaigns")
def start_campaign(cfg: CampaignConfig):
    if (cfg.scenario not in SCENARIOS or cfg.perception_hz not in (20, 10, 5) or not cfg.speeds or not cfg.weathers
            or not cfg.seeds or any(w not in WEATHERS for w in cfg.weathers)
            or any(not 10 <= v <= 90 for v in cfg.speeds)):
        raise HTTPException(422, "paramètre invalide")
    c = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
    if campaign_total(c) > MAX_CAMPAIGN_TRIALS:
        raise HTTPException(422, f"trop d'essais ({campaign_total(c)} > {MAX_CAMPAIGN_TRIALS})")
    busy = active_run()
    if busy:
        raise HTTPException(409, f"un run est déjà actif : {busy}")
    if not carla_status()["reachable"]:
        raise HTTPException(503, "serveur CARLA injoignable sur localhost:2000")
    cid = time.strftime("%Y%m%d-%H%M%S")
    d = CAMPAIGNS / cid
    d.mkdir(parents=True, exist_ok=True)
    (d / "studio_config.json").write_text(json.dumps(c))
    opts = [("--perception-hz", cfg.perception_hz, 20), ("--ghost-prob", cfg.ghost_prob, 0.0),
            ("--drop-prob", cfg.drop_prob, 0.0), ("--noise-sigma", cfg.noise_sigma, 0.0)]
    aeb_opts = " ".join(f"{k} {v}" for k, v, dflt in opts if v != dflt)
    if cfg.ghost_prob > 0:
        aeb_opts += f" --ghost-len {cfg.ghost_len}"
    cmd = [sys.executable, str(CAMP_SCRIPT), "--resume", cid, "--scenario", cfg.scenario,
           "--speeds", *[str(v) for v in cfg.speeds], "--weathers", *cfg.weathers,
           "--seeds", *[str(s) for s in cfg.seeds], "--gap", str(cfg.gap), "--lead-speed", str(cfg.lead_speed),
           "--lead-brake-t", str(cfg.lead_brake_t), f"--aeb-opts={aeb_opts.strip()}"]
    if cfg.same_speed:
        cmd.append("--same-speed")
    if cfg.skip_baseline:
        cmd.append("--skip-baseline")
    camp_procs[cid] = subprocess.Popen(cmd, stdout=open(d / "campaign.log", "w"), stderr=subprocess.STDOUT, cwd=str(ROOT))
    return {"id": cid}


@app.get("/api/campaigns")
def list_campaigns():
    if not CAMPAIGNS.is_dir():
        return []
    return [campaign_info(d.name) for d in sorted(CAMPAIGNS.iterdir(), reverse=True)
            if d.is_dir() and (d / "studio_config.json").exists()]


@app.get("/api/campaigns/{cid}")
def get_campaign(cid: str):
    info = campaign_info(cid)
    rows = read_rows(camp_dir(cid))
    scenario = (info["config"] or {}).get("scenario", "stopped")
    return {**info, "aggregate": aggregate(rows, scenario),
            "cases": [{k: r.get(k) for k in ("case", "collided", "impact_kmh", "min_gap_m", "n_brake_ticks", "max_state", "error")}
                      for r in rows]}


@app.post("/api/campaigns/{cid}/stop")
def stop_campaign(cid: str):
    camp_dir(cid)
    p = camp_procs.get(cid)
    if p is not None and p.poll() is None:
        p.terminate()             # l'essai en cours se termine tout seul ; les suivants ne sont pas lancés
    return {"ok": True}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

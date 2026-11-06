"""
API du Studio (FastAPI). N'importe pas carla : la simulation tourne dans backend/worker.py,
lancé en sous-processus. Un seul run actif à la fois (une seule instance CARLA/GPU).

Lancer : python -m uvicorn app.main:app --app-dir backend --port 8000
Puis ouvrir http://localhost:8000
"""
import asyncio
import base64
import json
import socket
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
SCENARIOS = ["stopped", "braking", "adjacent", "follow", "approach"]
STATIC = ROOT / "backend" / "static"
TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD_Opt"]
WEATHERS = ["clear", "rain", "fog"]
CARLA_HOST, CARLA_PORT = "localhost", 2000
MAX_BACKLOG = 20          # images au plus envoyées d'un coup à un client en retard

app = FastAPI(title="ADAS Validation Studio")
procs: dict = {}          # run_id -> Popen


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


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

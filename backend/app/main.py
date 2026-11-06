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
STATIC = ROOT / "backend" / "static"
TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD_Opt"]
WEATHERS = ["clear", "rain", "fog"]
CARLA_HOST, CARLA_PORT = "localhost", 2000

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
    st = read_json(d / "status.json") or {"state": "starting", "n_frames": 0, "error": None}
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
    return {"towns": TOWNS, "weathers": WEATHERS, "models": ["finetuned", "coco"]}


@app.post("/api/runs")
def start_run(cfg: RunConfig):
    if cfg.town not in TOWNS or cfg.weather not in WEATHERS or cfg.model not in ("finetuned", "coco"):
        raise HTTPException(422, "paramètre invalide")
    busy = active_run()
    if busy:
        raise HTTPException(409, f"un run est déjà actif : {busy}")
    if not carla_status()["reachable"]:
        raise HTTPException(503, "serveur CARLA injoignable sur localhost:2000")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(WORKER), "--run-id", run_id, "--town", cfg.town,
           "--weather", cfg.weather, "--n-npc", str(cfg.n_npc), "--seconds", str(cfg.seconds),
           "--seed", str(cfg.seed), "--conf", str(cfg.conf)]
    if cfg.model == "coco":
        cmd += ["--coco", "--weights", str(ROOT / "weights" / "yolo26n.pt")]
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
                out.append({"run_id": d.name, "state": st["state"], "n_frames": st["n_frames"],
                            "n_detections": s.get("n_detections"), "n_with_lidar": s.get("n_with_lidar"),
                            "config": (read_json(d / "config.json") or {})})
    return out


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    d = run_dir(run_id)
    return {"run_id": run_id, "status": run_status(run_id),
            "summary": read_json(d / "summary.json"), "config": read_json(d / "config.json")}


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


@app.websocket("/ws/runs/{run_id}")
async def stream(ws: WebSocket, run_id: str):
    await ws.accept()
    try:
        d = run_dir(run_id)
    except HTTPException:
        await ws.close(code=4404)
        return
    last = None
    try:
        while True:
            st = run_status(run_id)
            lj = d / "latest.json"
            m = lj.stat().st_mtime_ns if lj.exists() else None
            msg = None
            if m is not None and m != last:
                last = m
                rec = read_json(lj)
                try:
                    img = base64.b64encode((d / "latest.jpg").read_bytes()).decode()
                except OSError:
                    img = None
                if rec is not None:
                    msg = {"status": st, "frame": rec, "image": img}
            if msg is None and st["state"] not in ("starting", "running"):
                await ws.send_json({"status": st, "frame": None, "image": None, "final": True})
                break
            if msg is not None:
                await ws.send_json(msg)
            await asyncio.sleep(0.1)
    except Exception:       # noqa: BLE001 : client parti
        pass


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

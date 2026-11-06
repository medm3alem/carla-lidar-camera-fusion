r"""
Worker CARLA du Studio : un run = un processus. Connexion, simulation synchrone, YOLO + fusion LiDAR,
écriture des résultats dans runs\studio\<run_id>\ . Seul ce processus importe carla.

Sorties (dans le dossier du run) :
  config.json     paramètres du run
  status.json     état : starting | running | finished | stopped | error (+ compteurs)
  latest.jpg      dernière image annotée (écrite atomiquement)
  latest.json     détections de la dernière image
  frames.jsonl    une ligne JSON par image traitée
  frames\NNNNN.jpg  images annotées (si --save-images)
  summary.json    bilan du run
Arrêt propre : créer un fichier STOP dans le dossier du run.

Usage (serveur CARLA lancé) :
  python backend\worker.py --run-id test1 --town Town10HD_Opt --weather clear --seconds 20
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import carla
from fusion.config import CFG
from fusion import geometry as geo
from fusion.rig import WEATHER, fetch, spawn_sensors, lidar_points
from fusion.perception import fuse_box, mono_depth, project_lidar

COCO_VEHICLES = [2, 5, 7]


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def annotate(img, dets):
    out = img.copy()
    for d in dets:
        x0, y0, x1, y1 = (int(v) for v in d["box"])
        ok = d["lidar_depth"] is not None
        color = (0, 200, 0) if ok else (0, 140, 255)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        txt = (f"{d['lidar_depth']:.1f} m ({d['n_cluster']} pts)" if ok
               else f"~{d['mono_depth']:.0f} m (sans LiDAR)")
        cv2.putText(out, f"{txt} {d['conf']:.2f}", (x0, max(y0 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def spawn_traffic(world, tm, bpl, spawns, rng, n_npc):
    ego_bp = bpl.filter("vehicle.tesla.model3")[0]
    ego = None
    while ego is None and spawns:
        ego = world.try_spawn_actor(ego_bp, spawns.pop())
    if ego is None:
        raise RuntimeError("impossible de placer l'ego")
    actors = [ego]
    ego.set_autopilot(True, CFG.tm_port)
    tm.ignore_lights_percentage(ego, 100)
    npc_bps = [b for b in bpl.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
    n = 0
    for sp in spawns:
        if n >= n_npc:
            break
        npc = world.try_spawn_actor(rng.choice(npc_bps), sp)
        if npc is not None:
            npc.set_autopilot(True, CFG.tm_port)
            actors.append(npc)
            n += 1
    return ego, actors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--town", default=CFG.town)
    ap.add_argument("--weather", default="clear", choices=list(WEATHER))
    ap.add_argument("--n-npc", type=int, default=CFG.n_npc)
    ap.add_argument("--seed", type=int, default=CFG.seed)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--every", type=int, default=CFG.save_every, help="1 image traitée tous les N ticks")
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--coco", action="store_true", help="poids COCO (classes car/bus/truck)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--no-save-images", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "runs" / "studio"))
    args = ap.parse_args()

    run_dir = Path(args.out) / args.run_id
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", vars(args))
    status = {"state": "starting", "n_frames": 0, "error": None}
    write_json(run_dir / "status.json", status)

    client = world = tm = original = None
    sensors, actors = [], []
    t_start = time.time()
    n_det = n_lidar = 0
    infer_ms = []
    try:
        import torch
        from ultralytics import YOLO
        model = YOLO(args.weights)
        device = 0 if torch.cuda.is_available() else "cpu"

        client = carla.Client(CFG.host, CFG.port)
        client.set_timeout(20.0)             # CARLA lancé mais pas prêt : échec rapide et explicite
        world = client.get_world()
        if not world.get_map().name.endswith(args.town):
            world = client.load_world(args.town)
        tm = client.get_trafficmanager(CFG.tm_port)
        original = world.get_settings()
        s = world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = CFG.dt
        world.apply_settings(s)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(args.seed)
        world.set_weather(WEATHER[args.weather])

        rng = random.Random(args.seed)
        spawns = world.get_map().get_spawn_points()
        rng.shuffle(spawns)
        ego, actors = spawn_traffic(world, tm, world.get_blueprint_library(), spawns, rng, args.n_npc)
        for _ in range(int(CFG.warmup_seconds / CFG.dt)):
            world.tick()
        sensors, q = spawn_sensors(world, ego)
        for _ in range(3):
            world.tick()

        K = geo.intrinsics(CFG.img_w, CFG.img_h, CFG.fov)
        status["state"] = "running"
        write_json(run_dir / "status.json", status)
        classes = COCO_VEHICLES if args.coco else [0]
        n_ticks = int(args.seconds / CFG.dt)
        k = 0
        with open(run_dir / "frames.jsonl", "w") as jl:
            for i in range(1, n_ticks + 1):
                if (run_dir / "STOP").exists():
                    status["state"] = "stopped"
                    break
                frame = world.tick()
                rgb_d, _, lid_d = (fetch(q[n], frame) for n in ("rgb", "inst", "lidar"))
                if i % args.every:
                    continue
                img = np.ascontiguousarray(
                    np.frombuffer(rgb_d.raw_data, np.uint8).reshape(CFG.img_h, CFG.img_w, 4)[:, :, :3])
                T_wc = np.array(rgb_d.transform.get_matrix())
                T_wl = np.array(lid_d.transform.get_matrix())
                pts, _ = lidar_points(lid_d, T_wc, T_wl)
                uv, depth = project_lidar(pts, T_wl, T_wc, K, CFG.img_w, CFG.img_h, CFG.lidar_pos[2])

                t0 = time.time()
                r = model.predict(img, imgsz=args.imgsz, conf=args.conf, classes=classes,
                                  device=device, verbose=False)[0]
                infer_ms.append(1000 * (time.time() - t0))
                P, C = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
                dets = []
                for b, c in zip(P, C):
                    d_l, n_c = fuse_box(uv, depth, b)
                    dets.append({"box": [float(v) for v in b], "conf": float(c),
                                 "lidar_depth": d_l, "n_cluster": int(n_c),
                                 "mono_depth": mono_depth(K, b)})
                n_det += len(dets)
                n_lidar += sum(d["lidar_depth"] is not None for d in dets)

                vel = ego.get_velocity()
                rec = {"idx": k, "carla_frame": int(frame), "t": round(i * CFG.dt, 2),
                       "ego_speed": round(float(np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)), 2),
                       "detections": dets}
                jl.write(json.dumps(rec) + "\n")
                jl.flush()
                ann = annotate(img, dets)
                cv2.imwrite(str(run_dir / "latest.jpg.tmp.jpg"), ann)
                (run_dir / "latest.jpg.tmp.jpg").replace(run_dir / "latest.jpg")
                write_json(run_dir / "latest.json", rec)
                if not args.no_save_images:
                    cv2.imwrite(str(run_dir / "frames" / f"{k:05d}.jpg"), ann)
                k += 1
                status["n_frames"] = k
                if k % 5 == 0:
                    write_json(run_dir / "status.json", status)
            else:
                status["state"] = "finished"
        status["n_frames"] = k
    except Exception as e:                        # noqa: BLE001 : on veut tout reporter dans status.json
        status["state"] = "error"
        status["error"] = f"{type(e).__name__}: {e}"
        print(status["error"], file=sys.stderr)
    finally:
        for s_ in sensors:
            try:
                s_.stop()
            except Exception:
                pass
        try:
            if client is not None:
                client.apply_batch([carla.command.DestroyActor(a) for a in sensors + actors])
                world.tick()
            if world is not None and original is not None:
                world.apply_settings(original)
            if tm is not None:
                tm.set_synchronous_mode(False)
        except Exception as e:                    # noqa: BLE001
            print(f"nettoyage incomplet : {e}", file=sys.stderr)
        summary = {
            "state": status["state"], "error": status["error"], "n_frames": status["n_frames"],
            "n_detections": n_det, "n_with_lidar": n_lidar,
            "wall_seconds": round(time.time() - t_start, 1),
            "infer_ms_median": round(float(np.median(infer_ms)), 1) if infer_ms else None,
            "config": vars(args),
        }
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "status.json", status)
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

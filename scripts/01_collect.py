"""
Collecte v4 : épisodes courts, mêmes graines sous chaque météo.
Sortie : <out>/<météo>/ep###/{NNN_rgb.png, NNN_lidar.npy, labels.json}
  - lidar.npy : points LiDAR (repère LiDAR) restreints au cône de vue de la caméra
  - labels.json : poses, boîtes 2D serrées, visibilité, profondeurs GT

Usage :
  python scripts\01_collect.py --episodes 2 --weather clear fog --out data\v4_test
  python scripts\01_collect.py                      # collecte complète
Un épisode terminé (labels.json présent) est sauté : on peut reprendre après un arrêt.
"""
import argparse
import dataclasses
import json
import math
import queue
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
from fusion.rig import WEATHER, fetch, spawn_sensors, lidar_points, in_view, label_objects


def med(v):
    return f"{np.median(v):.0f}" if len(v) else "-"


def run_episode(client, world, tm, weather, ep, out_dir, K):
    ep_dir = out_dir / weather / f"ep{ep:03d}"
    if (ep_dir / "labels.json").exists():
        print(f"[{weather} ep{ep:03d}] déjà fait, sauté")
        return
    ep_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(CFG.seed * 1000 + ep)      # mêmes points de départ pour toutes les météos
    tm.set_random_device_seed(CFG.seed + ep)
    world.set_weather(WEATHER[weather])
    bpl = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    rng.shuffle(spawns)

    actors, sensors = [], []
    t0 = time.time()
    try:
        ego_bp = bpl.filter("vehicle.tesla.model3")[0]
        ego = None
        while ego is None and spawns:
            ego = world.try_spawn_actor(ego_bp, spawns.pop())
        if ego is None:
            raise RuntimeError("impossible de placer l'ego")
        actors.append(ego)
        ego.set_autopilot(True, CFG.tm_port)

        npc_bps = [b for b in bpl.filter("vehicle.*")
                   if int(b.get_attribute("number_of_wheels")) == 4]
        n_npc = 0
        for sp in spawns:
            if n_npc >= CFG.n_npc:
                break
            npc = world.try_spawn_actor(rng.choice(npc_bps), sp)
            if npc is not None:
                npc.set_autopilot(True, CFG.tm_port)
                actors.append(npc)
                n_npc += 1

        for _ in range(int(CFG.warmup_seconds / CFG.dt)):
            world.tick()
        sensors, q = spawn_sensors(world, ego)
        for _ in range(3):
            world.tick()

        n_ticks = int(CFG.episode_seconds / CFG.dt)
        frames, k, n_stopped, n_missed = [], 0, 0, 0
        ego_dist, prev = 0.0, ego.get_location()
        for i in range(1, n_ticks + 1):
            frame = world.tick()
            loc = ego.get_location()
            ego_dist += loc.distance(prev)
            prev = loc
            if i % CFG.save_every:
                continue
            vel = ego.get_velocity()
            if math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2) < CFG.min_ego_speed:
                n_stopped += 1
                continue
            try:
                rgb_d, inst_d, lid_d = (fetch(q[n], frame) for n in ("rgb", "inst", "lidar"))
            except (queue.Empty, RuntimeError):
                n_missed += 1
                continue

            T_wc = np.array(rgb_d.transform.get_matrix())
            T_wl = np.array(lid_d.transform.get_matrix())
            pts, pts_cam = lidar_points(lid_d, T_wc, T_wl)
            objs, n_occ = label_objects(world, ego, inst_d, pts_cam, T_wc, K)

            img = np.frombuffer(rgb_d.raw_data, np.uint8).reshape(CFG.img_h, CFG.img_w, 4)[:, :, :3]
            cv2.imwrite(str(ep_dir / f"{k:03d}_rgb.png"), img)
            np.save(ep_dir / f"{k:03d}_lidar.npy", pts[in_view(pts_cam)])
            frames.append({
                "idx": k, "carla_frame": frame,
                "T_world_cam": T_wc.tolist(), "T_world_lidar": T_wl.tolist(),
                "n_occluded": n_occ, "objects": objs,
            })
            k += 1

        labels = {"episode": ep, "split": CFG.split_of(ep), "weather": weather,
                  "ego_distance_m": round(ego_dist, 1), "n_stopped": n_stopped,
                  "n_missed": n_missed, "frames": frames}
        tmp = ep_dir / "labels.json.tmp"
        tmp.write_text(json.dumps(labels))
        tmp.replace(ep_dir / "labels.json")

        objs = [o for f in frames for o in f["objects"]]
        kept = [o for o in objs if not o["ignore"]]

        def pts_med(lo, hi):
            return med([o["n_lidar"] for o in kept if lo <= o["distance"] < hi])

        dt = time.time() - t0
        print(f"[{weather} ep{ep:03d}] {len(frames)} frames | véhicules {len(kept)} "
              f"(ignorés {len(objs) - len(kept)}), cachés {sum(f['n_occluded'] for f in frames)} | "
              f"ego {ego_dist:.0f} m, {n_stopped} frames à l'arrêt, {n_missed} manquées | "
              f"pts LiDAR/véhicule méd. <20 m : {pts_med(0, 20)}, 20-40 : {pts_med(20, 40)}, "
              f"40-60 : {pts_med(40, 60)} | {dt:.0f} s")
    finally:
        for s in sensors:
            try:
                s.stop()
            except Exception:
                pass
        client.apply_batch([carla.command.DestroyActor(a) for a in sensors + actors])
        world.tick()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather", nargs="+", default=list(WEATHER), choices=list(WEATHER))
    ap.add_argument("--episodes", type=int, default=CFG.n_episodes)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "v4"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = carla.Client(CFG.host, CFG.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if not world.get_map().name.endswith(CFG.town):
        world = client.load_world(CFG.town)
    tm = client.get_trafficmanager(CFG.tm_port)
    original = world.get_settings()
    K = geo.intrinsics(CFG.img_w, CFG.img_h, CFG.fov)
    (out_dir / "meta.json").write_text(
        json.dumps({"config": dataclasses.asdict(CFG), "K": K.tolist()}, indent=1))

    try:
        s = world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = CFG.dt
        world.apply_settings(s)
        tm.set_synchronous_mode(True)
        for ep in range(args.start, args.start + args.episodes):
            for weather in args.weather:
                run_episode(client, world, tm, weather, ep, out_dir, K)
    finally:
        world.apply_settings(original)
        tm.set_synchronous_mode(False)
        print("Terminé.")


if __name__ == "__main__":
    main()
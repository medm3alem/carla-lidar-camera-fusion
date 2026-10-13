"""
Test du banc de capteurs : ego + trafic, RGB + segmentation d'instance + LiDAR.
Valide : décodage des ids d'instance, boîtes serrées, visibilité, densité LiDAR par véhicule,
profondeur de référence (face la plus proche du véhicule).

Usage : python scripts\00_rig_test.py [--weather clear|rain|fog] [--frames 3]
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import carla
from fusion.config import CFG
from fusion import geometry as geo

OUT = ROOT / "runs" / "rig_test"
MIN_PIXELS = 20          # en dessous, le véhicule est considéré comme non visible
SETTLE_TICKS = 20        # 1 s entre deux frames analysées

WEATHER = {
    "clear": carla.WeatherParameters.ClearNoon,
    "rain": carla.WeatherParameters.HardRainNoon,
    "fog": carla.WeatherParameters(cloudiness=80.0, fog_density=60.0,
                                   fog_distance=20.0, sun_altitude_angle=45.0),
}


def fetch(q, frame, timeout=10.0):
    """Récupère la donnée du capteur pour CE frame (jette les plus anciennes)."""
    while True:
        data = q.get(timeout=timeout)
        if data.frame == frame:
            return data
        if data.frame > frame:
            raise RuntimeError(f"Frame {frame} manquante (reçu {data.frame})")


def spawn_sensors(world, ego, queues):
    bpl = world.get_blueprint_library()
    cam_tf = carla.Transform(carla.Location(*CFG.cam_pos))
    lid_tf = carla.Transform(carla.Location(*CFG.lidar_pos))

    def cam_bp(name):
        bp = bpl.find(name)
        bp.set_attribute("image_size_x", str(CFG.img_w))
        bp.set_attribute("image_size_y", str(CFG.img_h))
        bp.set_attribute("fov", str(CFG.fov))
        return bp

    rgb = world.spawn_actor(cam_bp("sensor.camera.rgb"), cam_tf, attach_to=ego)
    inst = world.spawn_actor(cam_bp("sensor.camera.instance_segmentation"), cam_tf, attach_to=ego)

    lb = bpl.find("sensor.lidar.ray_cast")
    for k, v in {
        "channels": CFG.lidar_channels,
        "points_per_second": CFG.lidar_points_per_second,
        "rotation_frequency": 1.0 / CFG.dt,
        "range": CFG.lidar_range,
        "upper_fov": CFG.lidar_upper_fov,
        "lower_fov": CFG.lidar_lower_fov,
        "dropoff_general_rate": CFG.lidar_dropoff,
    }.items():
        lb.set_attribute(k, str(v))
    lidar = world.spawn_actor(lb, lid_tf, attach_to=ego)

    rgb.listen(queues["rgb"].put)
    inst.listen(queues["inst"].put)
    lidar.listen(queues["lidar"].put)
    return [rgb, inst, lidar]


def analyze(world, ego, rgb_d, inst_d, lid_d, K, tag):
    W, H = CFG.img_w, CFG.img_h
    img = np.frombuffer(rgb_d.raw_data, np.uint8).reshape(H, W, 4)[:, :, :3].copy()
    seg = np.frombuffer(inst_d.raw_data, np.uint8).reshape(H, W, 4)   # BGRA
    ids = seg[:, :, 1].astype(np.int32) + seg[:, :, 0].astype(np.int32) * 256
    tags = seg[:, :, 2]

    pts = np.frombuffer(lid_d.raw_data, np.float32).reshape(-1, 4)
    T_wc = np.array(rgb_d.transform.get_matrix())
    T_wl = np.array(lid_d.transform.get_matrix())
    T_cw = np.linalg.inv(T_wc)
    pts_cam = geo.transform_points(geo.lidar_to_camera_matrix(T_wl, T_wc), pts[:, :3])
    uv, depth, _ = geo.project(pts_cam, K, W, H)
    px = uv.astype(int)
    pid = ids[px[:, 1], px[:, 0]]            # id d'instance sous chaque point LiDAR

    # points LiDAR en overlay, colorés par profondeur
    col = cv2.applyColorMap(
        (np.clip(depth, 0, CFG.lidar_range) / CFG.lidar_range * 255).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_JET).reshape(-1, 3)
    for (x, y), c in zip(px, col):
        cv2.circle(img, (int(x), int(y)), 1, c.tolist(), -1)

    ego_loc = ego.get_location()
    rows, n_in_range, n_hidden = [], 0, 0
    for v in world.get_actors().filter("vehicle.*"):
        if v.id == ego.id:
            continue
        dist = v.get_location().distance(ego_loc)
        if dist > CFG.max_label_distance:
            continue
        verts_w = np.array([[p.x, p.y, p.z]
                            for p in v.bounding_box.get_world_vertices(v.get_transform())])
        verts_c = geo.transform_points(T_cw, verts_w)
        gt_face = geo.nearest_face_depth(verts_c)
        if verts_c[:, 0].max() <= 0.1:       # entièrement derrière la caméra
            continue
        n_in_range += 1

        mask = ids == v.id
        n_pix = int(mask.sum())
        if n_pix < MIN_PIXELS:
            n_hidden += 1
            continue
        ys, xs = np.nonzero(mask)
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        uv_all, d_all = geo.project_all(verts_c, K)
        if (d_all > 0.1).all():
            hull = cv2.convexHull(uv_all.astype(np.float32))
            vis = min(1.0, n_pix / max(float(cv2.contourArea(hull)), 1.0))
        else:
            vis = None                       # boîte partiellement derrière la caméra

        sel = pid == v.id
        n_pts = int(sel.sum())
        rows.append({
            "id": v.id, "type": v.type_id.replace("vehicle.", ""), "dist": dist,
            "gt_face": gt_face, "gt_center": float(verts_c[:, 0].mean()),
            "vis": vis, "n_pix": n_pix, "n_pts": n_pts,
            "lidar_med": float(np.median(depth[sel])) if n_pts else float("nan"),
            "tag": int(np.bincount(tags[mask]).argmax()),
        })
        ignore = vis is None or vis < CFG.min_visibility
        color = (0, 165, 255) if ignore else (0, 255, 0)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 1)
        label = f"{dist:.0f}m v={'-' if vis is None else f'{vis:.2f}'} {n_pts}pts"
        cv2.putText(img, label, (box[0], max(box[1] - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"overlay_{tag}.png"), img)

    print(f"\n--- frame {tag} : {len(pts)} points LiDAR ({len(depth)} dans l'image) | "
          f"véhicules <= {CFG.max_label_distance:.0f} m devant : {n_in_range}, "
          f"vus en segmentation : {len(rows)}, cachés : {n_hidden}")
    print(f"{'id':>5} {'type':<24} {'dist':>5} {'face':>6} {'centre':>6} {'vis':>5} "
          f"{'pix':>6} {'pts':>5} {'lidar':>6} {'dL-face':>7} {'tag':>3}")
    for r in sorted(rows, key=lambda r: r["dist"]):
        vis = "  -  " if r["vis"] is None else f"{r['vis']:.2f}"
        d = r["lidar_med"] - r["gt_face"]
        print(f"{r['id']:>5} {r['type']:<24.24} {r['dist']:5.1f} {r['gt_face']:6.1f} "
              f"{r['gt_center']:6.1f} {vis:>5} {r['n_pix']:6d} {r['n_pts']:5d} "
              f"{r['lidar_med']:6.1f} {d:7.2f} {r['tag']:3d}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather", default="clear", choices=list(WEATHER))
    ap.add_argument("--frames", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(CFG.seed)
    client = carla.Client(CFG.host, CFG.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if not world.get_map().name.endswith(CFG.town):
        world = client.load_world(CFG.town)
    tm = client.get_trafficmanager(CFG.tm_port)
    original = world.get_settings()
    actors = []

    try:
        s = world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = CFG.dt
        world.apply_settings(s)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(CFG.seed)
        world.set_weather(WEATHER[args.weather])

        bpl = world.get_blueprint_library()
        spawns = world.get_map().get_spawn_points()
        rng.shuffle(spawns)

        ego_bp = bpl.filter("vehicle.tesla.model3")[0]
        ego = None
        while ego is None and spawns:
            ego = world.try_spawn_actor(ego_bp, spawns.pop())
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
        print(f"Ego + {n_npc} NPC, météo {args.weather}")

        for _ in range(int(CFG.warmup_seconds / CFG.dt)):
            world.tick()

        import queue
        queues = {k: queue.Queue() for k in ("rgb", "inst", "lidar")}
        actors += spawn_sensors(world, ego, queues)
        K = geo.intrinsics(CFG.img_w, CFG.img_h, CFG.fov)

        all_rows = []
        for k in range(args.frames):
            for _ in range(SETTLE_TICKS):
                world.tick()
            frame = world.tick()
            all_rows += analyze(world, ego,
                                fetch(queues["rgb"], frame), fetch(queues["inst"], frame),
                                fetch(queues["lidar"], frame), K, k)

        print("\n=== Bilan ===")
        for lo, hi in [(0, 20), (20, 40), (40, 60)]:
            r = [x for x in all_rows if lo <= x["dist"] < hi]
            if r:
                n = [x["n_pts"] for x in r]
                print(f"{lo:2d}-{hi} m : {len(r):3d} véhicules, points LiDAR/véhicule "
                      f"médiane {np.median(n):.0f}, min {min(n)}, sans aucun point : {sum(v == 0 for v in n)}")
        ok = [x for x in all_rows if x["n_pts"] > 0]
        if ok:
            print(f"Profondeur : LiDAR - face = {np.median([x['lidar_med'] - x['gt_face'] for x in ok]):+.2f} m "
                  f"(médiane) | LiDAR - centre = {np.median([x['lidar_med'] - x['gt_center'] for x in ok]):+.2f} m")
        print(f"Overlays : {OUT}")

    finally:
        for a in actors:
            if hasattr(a, "stop"):
                try:
                    a.stop()
                except Exception:
                    pass
        client.apply_batch([carla.command.DestroyActor(a) for a in actors if a is not None])
        world.apply_settings(original)
        tm.set_synchronous_mode(False)
        print("Nettoyage terminé.")


if __name__ == "__main__":
    main()
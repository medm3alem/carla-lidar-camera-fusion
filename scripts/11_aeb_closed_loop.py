r"""
AEB en boucle fermée dans CARLA : un scénario scripté, la perception réelle (YOLO + LiDAR),
le tracker et l'AEB commandent le frein de l'ego. Sans trafic : seuls l'ego et le véhicule cible.

Scénarios
  stopped : véhicule cible à l'arrêt devant l'ego (qui roule à --v0)
  braking : véhicule cible roulant à --lead-speed, qui freine à t = --lead-brake-t
Chaque tick (20 Hz) : capteurs -> YOLO -> LiDAR -> tracker -> AEB -> commande de l'ego.
Sorties : runs\aeb\<id>\ log.csv (une ligne par tick), summary.json, frames\ (images annotées à ~5 Hz).

Usage (serveur CARLA lancé, venv actif) :
  python scripts\11_aeb_closed_loop.py --scenario stopped --v0 50 --gap 80
  python scripts\11_aeb_closed_loop.py --scenario stopped --v0 50 --gap 80 --no-aeb   # témoin sans AEB
  python scripts\11_aeb_closed_loop.py --scenario braking --v0 50 --lead-speed 50 --gap 30 --lead-brake-t 3
"""
import argparse
import csv
import json
import math
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
from fusion.perception import fuse_box, project_lidar
from fusion.adas import AEB, AEBConfig, Tracker

STATE_COLOR = {0: (0, 200, 0), 1: (0, 200, 255), 2: (0, 140, 255), 3: (0, 0, 255)}


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def speed_of(v):
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def straight_length(wp, limit, step=5.0, max_dev=3.0):
    """Longueur (m) du tronçon droit sans carrefour qui suit wp, plafonnée à `limit`."""
    cur, n = wp, 0
    if wp.is_junction:
        return 0.0
    while n * step < limit:
        nxt = cur.next(step)
        if len(nxt) != 1 or nxt[0].is_junction:
            break
        if abs(wrap(nxt[0].transform.rotation.yaw - wp.transform.rotation.yaw)) > max_dev:
            break
        cur = nxt[0]
        n += 1
    return n * step


def find_straight(cmap, length, rng, step=5.0):
    """Retourne (waypoint de départ, longueur droite disponible). Prend le premier tronçon assez long,
    sinon le plus long trouvé (à l'appelant de juger)."""
    cands = [cmap.get_waypoint(sp.location) for sp in cmap.get_spawn_points()]
    extra = [w for w in cmap.generate_waypoints(step * 2) if w.lane_type == carla.LaneType.Driving]
    rng.shuffle(cands)
    rng.shuffle(extra)
    best, best_len = None, -1.0
    for wp in cands + extra:
        n = straight_length(wp, length, step)
        if n > best_len:
            best, best_len = wp, n
        if n >= length:
            break
    return best, best_len


def lane_steer(veh, cmap):
    tf = veh.get_transform()
    wp = cmap.get_waypoint(tf.location, project_to_road=True)
    dyaw = wrap(wp.transform.rotation.yaw - tf.rotation.yaw)
    d = tf.location - wp.transform.location
    r = wp.transform.get_right_vector()
    lat = d.x * r.x + d.y * r.y
    v = speed_of(veh.get_velocity())
    return float(np.clip((dyaw - math.degrees(math.atan2(0.8 * lat, v + 2.0))) / 70.0, -1, 1))


def cruise(veh, cmap, target):
    v = speed_of(veh.get_velocity())
    u = 0.3 * (target - v)
    return carla.VehicleControl(throttle=float(np.clip(u, 0, 0.75)), brake=float(np.clip(-u * 0.5, 0, 1)),
                                steer=lane_steer(veh, cmap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["stopped", "braking"], default="stopped")
    ap.add_argument("--v0", type=float, default=50.0, help="vitesse de l'ego, km/h")
    ap.add_argument("--gap", type=float, default=80.0, help="écart initial (pare-chocs à pare-chocs), m")
    ap.add_argument("--lead-speed", type=float, default=50.0, help="km/h (scénario braking)")
    ap.add_argument("--lead-brake-t", type=float, default=3.0)
    ap.add_argument("--lead-brake", type=float, default=0.75, help="consigne de frein du véhicule cible")
    ap.add_argument("--no-aeb", action="store_true")
    ap.add_argument("--town", default="Town10HD_Opt")
    ap.add_argument("--weather", default="clear", choices=list(WEATHER))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--t-max", type=float, default=25.0)
    ap.add_argument("--out", default=str(ROOT / "runs" / "aeb"))
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    model = YOLO(args.weights)
    device = 0 if torch.cuda.is_available() else "cpu"

    run_id = time.strftime("%Y%m%d-%H%M%S") + f"_{args.scenario}_{int(args.v0)}" + ("_noaeb" if args.no_aeb else "")
    run_dir = Path(args.out) / run_id
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)

    client = carla.Client(CFG.host, CFG.port)
    client.set_timeout(30.0)
    world = client.get_world()
    if not world.get_map().name.endswith(args.town):
        world = client.load_world(args.town)
    original = world.get_settings()
    cmap = world.get_map()
    actors, sensors = [], []
    summary = {"run_id": run_id, "config": vars(args)}
    try:
        s = world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = CFG.dt
        world.apply_settings(s)
        world.set_weather(WEATHER[args.weather])

        v0 = args.v0 / 3.6
        vl = 0.0 if args.scenario == "stopped" else args.lead_speed / 3.6
        # route nécessaire : écart initial + trajet de la cible + marge pour que l'ego s'arrête
        lead_travel = 0.0 if vl == 0 else vl * args.lead_brake_t + vl ** 2 / (2 * 4.0)
        need = args.gap + 5.0 + lead_travel + 25.0
        wp0, avail = find_straight(cmap, need, random.Random(args.seed))
        print(f"tronçon droit : {avail:.0f} m disponibles, {need:.0f} m nécessaires")
        if wp0 is None or avail < need:
            raise RuntimeError(f"plus long tronçon droit de {args.town} : {avail:.0f} m, il en faut {need:.0f} ; "
                               f"réduire --gap à {max(avail - 30 - lead_travel, 0):.0f} m ou changer de ville (--town Town03)")

        bpl = world.get_blueprint_library()
        ego = world.spawn_actor(bpl.find("vehicle.tesla.model3"), carla.Transform(
            wp0.transform.location + carla.Location(z=0.3), wp0.transform.rotation))
        actors.append(ego)
        ext_e = ego.bounding_box.extent.x
        lead_bp = bpl.find("vehicle.audi.etron")
        # distance centre à centre (demi-longueur de la cible estimée à 2,4 m), en compensant le
        # rapprochement pendant les 5 ticks de mise en place
        d_center = args.gap + ext_e + 2.4 + (v0 - vl) * 5 * CFG.dt
        wpl = wp0.next(d_center)[0]
        lead = world.spawn_actor(lead_bp, carla.Transform(wpl.transform.location + carla.Location(z=0.3),
                                                           wpl.transform.rotation))
        actors.append(lead)
        ext_l = lead.bounding_box.extent.x
        fwd = wp0.transform.get_forward_vector()
        ego.set_target_velocity(carla.Vector3D(fwd.x * v0, fwd.y * v0, 0))
        if vl > 0:
            lead.set_target_velocity(carla.Vector3D(fwd.x * vl, fwd.y * vl, 0))

        col_events = []
        col = world.spawn_actor(bpl.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
        col.listen(lambda e: col_events.append(e.other_actor.id))
        sensors_cam, q = spawn_sensors(world, ego, inst=False)
        sensors = [col] + sensors_cam

        def lead_control(t):
            if args.scenario == "stopped":
                return carla.VehicleControl(brake=1.0, hand_brake=True)
            if t >= args.lead_brake_t:
                return carla.VehicleControl(brake=args.lead_brake, steer=lane_steer(lead, cmap))
            return cruise(lead, cmap, vl)

        for _ in range(5):                                   # mise en place (chute, vitesse)
            ego.apply_control(cruise(ego, cmap, v0))
            lead.apply_control(lead_control(0.0))
            frame = world.tick()
            for n in ("rgb", "lidar"):
                fetch(q[n], frame)

        K = geo.intrinsics(CFG.img_w, CFG.img_h, CFG.fov)
        tracker = Tracker(CFG.dt)
        aeb = AEB(AEBConfig(), CFG.dt)
        cam_to_bumper = ext_e - CFG.cam_pos[0]               # la profondeur LiDAR est mesurée depuis la caméra

        def true_gap():
            et, lt = ego.get_transform(), lead.get_transform()
            f = et.get_forward_vector()
            d = lt.location - et.location
            return d.x * f.x + d.y * f.y + d.z * f.z - ext_e - ext_l

        log, infer_ms = [], []
        t_warn = t_brake = None
        collided, impact = False, 0.0
        min_gap = true_gap()
        gap_start = min_gap
        stopped_ticks = 0
        last_ev, last_vl = speed_of(ego.get_velocity()), speed_of(lead.get_velocity())
        n_ticks = int(args.t_max / CFG.dt)
        out = None
        for i in range(1, n_ticks + 1):
            t = i * CFG.dt
            frame = world.tick()
            rgb_d, lid_d = fetch(q["rgb"], frame), fetch(q["lidar"], frame)
            img = np.ascontiguousarray(
                np.frombuffer(rgb_d.raw_data, np.uint8).reshape(CFG.img_h, CFG.img_w, 4)[:, :, :3])
            T_wc, T_wl = np.array(rgb_d.transform.get_matrix()), np.array(lid_d.transform.get_matrix())
            pts, _ = lidar_points(lid_d, T_wc, T_wl)
            uv, depth = project_lidar(pts, T_wl, T_wc, K, CFG.img_w, CFG.img_h, CFG.lidar_pos[2])

            t0 = time.time()
            r = model.predict(img, imgsz=args.imgsz, conf=args.conf, classes=[0], device=device, verbose=False)[0]
            infer_ms.append(1000 * (time.time() - t0))
            meas, dets = [], []
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                d_l, n_c = fuse_box(uv, depth, b)
                if d_l is None:
                    continue
                u_c = (b[0] + b[2]) / 2
                meas.append((d_l - cam_to_bumper, (u_c - K[0, 2]) * d_l / K[0, 0]))
                dets.append((b, d_l, n_c))
            tracks = tracker.step(meas)

            ev, vlv = speed_of(ego.get_velocity()), speed_of(lead.get_velocity())
            out = aeb.step(tracks, ev)
            if out.warning and t_warn is None:
                t_warn = t
            if out.brake > 0 and t_brake is None:
                t_brake = t
            if args.no_aeb or out.brake <= 0:
                ego.apply_control(cruise(ego, cmap, v0))
            else:
                ego.apply_control(carla.VehicleControl(throttle=0.0, brake=out.brake, steer=lane_steer(ego, cmap)))
            lead.apply_control(lead_control(t))

            g = true_gap()
            min_gap = min(min_gap, g)
            log.append({"t": round(t, 3), "ego_v": round(ev, 3), "lead_v": round(vlv, 3), "gap_true": round(g, 3),
                        "n_det": len(dets), "n_tracks": len(tracks),
                        "meas_range": round(meas[0][0], 3) if meas else "",
                        "ttc": round(out.ttc, 3) if out.ttc is not None else "",
                        "a_req": round(out.a_req, 3) if np.isfinite(out.a_req) else "inf",
                        "state": out.state_name, "brake": out.brake})
            if i % 4 == 0:
                ann = img.copy()
                for b, d_l, n_c in dets:
                    cv2.rectangle(ann, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), STATE_COLOR[out.state], 2)
                    cv2.putText(ann, f"{d_l:.1f} m", (int(b[0]), max(int(b[1]) - 4, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, STATE_COLOR[out.state], 2)
                cv2.putText(ann, f"{out.state_name}  ego {ev * 3.6:.0f} km/h  gap {g:.1f} m", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, STATE_COLOR[out.state], 2)
                cv2.imwrite(str(run_dir / "frames" / f"{i:05d}.jpg"), ann)

            if lead.id in col_events:
                collided, impact = True, max(last_ev - last_vl, 0.0)
                break
            last_ev, last_vl = ev, vlv
            stopped_ticks = stopped_ticks + 1 if ev < 0.1 else 0
            if stopped_ticks > 20:
                break
            if g < -1.0:
                collided, impact = True, max(ev - vlv, 0.0)
                break
        summary.update({
            "collided": collided, "impact_speed_kmh": round(impact * 3.6, 1), "min_gap_m": round(min_gap, 2),
            "gap_start_m": round(gap_start, 2), "t_first_warning_s": t_warn, "t_first_brake_s": t_brake,
            "ego_final_speed_kmh": round(log[-1]["ego_v"] * 3.6, 1) if log else None,
            "n_ticks": len(log), "infer_ms_median": round(float(np.median(infer_ms)), 1) if infer_ms else None,
            "aeb_enabled": not args.no_aeb, "error": None,
        })
        with open(run_dir / "log.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(log[0].keys()))
            wr.writeheader()
            wr.writerows(log)
    except Exception as e:                                   # noqa: BLE001
        summary["error"] = f"{type(e).__name__}: {e}"
        print(summary["error"], file=sys.stderr)
    finally:
        for s_ in sensors:
            try:
                s_.stop()
            except Exception:
                pass
        try:
            client.apply_batch([carla.command.DestroyActor(a) for a in sensors + actors])
            world.tick()
            world.apply_settings(original)
        except Exception as e:                               # noqa: BLE001
            print(f"nettoyage incomplet : {e}", file=sys.stderr)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

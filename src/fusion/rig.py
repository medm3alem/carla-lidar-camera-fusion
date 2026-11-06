"""Banc de capteurs CARLA partagé : météo, capteurs, synchronisation, étiquetage."""
import queue

import cv2
import numpy as np
import carla

from .config import CFG
from . import geometry as geo

MIN_PIXELS = 20   # en dessous, le véhicule est considéré comme non visible

WEATHER = {
    "clear": carla.WeatherParameters.ClearNoon,
    "rain": carla.WeatherParameters.HardRainNoon,
    "fog": carla.WeatherParameters(cloudiness=80.0, fog_density=60.0,
                                   fog_distance=20.0, sun_altitude_angle=45.0),
}


def fetch(q, frame, timeout=10.0):
    """Donnée du capteur pour CE frame (jette les plus anciennes)."""
    while True:
        data = q.get(timeout=timeout)
        if data.frame == frame:
            return data
        if data.frame > frame:
            raise RuntimeError(f"frame {frame} manquante (reçu {data.frame})")


def spawn_sensors(world, ego, inst=True):
    """RGB + segmentation d'instance + LiDAR attachés à l'ego. Retourne (capteurs, files).
    inst=False : pas de caméra d'instance (inutile hors collecte, coûteuse en boucle fermée)."""
    bpl = world.get_blueprint_library()
    cam_tf = carla.Transform(carla.Location(*CFG.cam_pos))
    lid_tf = carla.Transform(carla.Location(*CFG.lidar_pos))

    def cam_bp(name):
        bp = bpl.find(name)
        bp.set_attribute("image_size_x", str(CFG.img_w))
        bp.set_attribute("image_size_y", str(CFG.img_h))
        bp.set_attribute("fov", str(CFG.fov))
        return bp

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

    rgb = world.spawn_actor(cam_bp("sensor.camera.rgb"), cam_tf, attach_to=ego)
    lidar = world.spawn_actor(lb, lid_tf, attach_to=ego)
    queues = {k: queue.Queue() for k in ("rgb", "inst", "lidar")}
    rgb.listen(queues["rgb"].put)
    lidar.listen(queues["lidar"].put)
    sensors = [rgb, lidar]
    if inst:
        seg = world.spawn_actor(cam_bp("sensor.camera.instance_segmentation"), cam_tf, attach_to=ego)
        seg.listen(queues["inst"].put)
        sensors.insert(1, seg)
    return sensors, queues


def lidar_points(lid_d, T_wc, T_wl):
    """Points bruts (N,4) dans le repère LiDAR, et leur position (N,3) dans le repère caméra."""
    pts = np.frombuffer(lid_d.raw_data, np.float32).reshape(-1, 4)
    pts_cam = geo.transform_points(geo.lidar_to_camera_matrix(T_wl, T_wc), pts[:, :3])
    return pts, pts_cam


def in_view(pts_cam):
    """Points dans le cône de vue de la caméra, avec marge (limite le stockage)."""
    x = pts_cam[:, 0]
    return (x > 0.1) & (np.abs(pts_cam[:, 1]) < 1.2 * x) & (np.abs(pts_cam[:, 2]) < 0.9 * x)


def label_objects(world, ego, inst_d, pts_cam, T_wc, K):
    """Boîtes 2D serrées (segmentation d'instance), visibilité, profondeurs GT, points LiDAR.
    Retourne (objets, nombre de véhicules dans le champ mais cachés)."""
    W, H = CFG.img_w, CFG.img_h
    seg = np.frombuffer(inst_d.raw_data, np.uint8).reshape(H, W, 4)   # BGRA : R = classe, G/B = id
    ids = seg[:, :, 1].astype(np.int32) + seg[:, :, 0].astype(np.int32) * 256
    T_cw = np.linalg.inv(T_wc)

    uv, _, _ = geo.project(pts_cam, K, W, H)
    px = uv.astype(int)
    pid = ids[px[:, 1], px[:, 0]]            # id d'instance sous chaque point LiDAR

    ego_loc = ego.get_location()
    objs, n_occluded = [], 0
    for v in world.get_actors().filter("vehicle.*"):
        if v.id == ego.id:
            continue
        dist = v.get_location().distance(ego_loc)
        if dist > CFG.max_label_distance:
            continue
        verts_w = np.array([[p.x, p.y, p.z]
                            for p in v.bounding_box.get_world_vertices(v.get_transform())])
        verts_c = geo.transform_points(T_cw, verts_w)
        uv_b, d_b = geo.project_all(verts_c, K)
        front = d_b > 0.1
        if not front.any():
            continue
        p = uv_b[front]
        if p[:, 0].max() < 0 or p[:, 0].min() >= W or p[:, 1].max() < 0 or p[:, 1].min() >= H:
            continue                          # hors du champ de la caméra
        mask = ids == v.id
        n_pix = int(mask.sum())
        if n_pix < MIN_PIXELS:
            n_occluded += 1                   # dans le champ, mais caché
            continue
        ys, xs = np.nonzero(mask)

        vis = None                            # inconnue si la boîte passe derrière la caméra
        if front.all():
            hull = cv2.convexHull(np.clip(uv_b, -20000, 20000).astype(np.int32))
            hm = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(hm, hull, 1)
            vis = min(1.0, n_pix / max(int(hm.sum()), 1))

        objs.append({
            "id": v.id,
            "type_id": v.type_id,
            "box2d": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "visibility": None if vis is None else round(vis, 3),
            "ignore": bool(vis is None or vis < CFG.min_visibility),
            "n_pix": n_pix,
            "n_lidar": int((pid == v.id).sum()),
            "distance": round(float(dist), 2),
            "depth_face": round(geo.nearest_face_depth(verts_c), 3),
            "depth_center": round(float(d_b.mean()), 3),
            "box3d_cam": np.round(verts_c, 3).tolist(),
        })
    return objs, n_occluded
"""
CARLA LiDAR-Camera Dataset Collector V2

Collecte :
- RGB
- LiDAR
- Bounding boxes 2D
- Ground truth 3D
- Plusieurs conditions meteo
- Synchronisation stricte Camera / LiDAR
"""

import carla
import numpy as np
import cv2
import json
import queue
import os
import random
import time


# ============================================================
# CONFIGURATION DATASET
# ============================================================

OUTPUT_DIR = "dataset_full_v3"

FRAMES_PER_WEATHER = 2000

IMG_WIDTH = 800
IMG_HEIGHT = 600
FOV = 90.0

FIXED_DELTA_SECONDS = 0.05       # 20 Hz

N_TRAFFIC_VEHICLES = 50

# Cohérent avec la portée du LiDAR
MAX_RENDER_DISTANCE = 50.0

# LiDAR
LIDAR_RANGE = 50.0
LIDAR_CHANNELS = 32
LIDAR_POINTS_PER_SECOND = 200000


# ============================================================
# METEO
# ============================================================

WEATHER_PRESETS = {

    "clear": carla.WeatherParameters.ClearNoon,

    "rain": carla.WeatherParameters.HardRainNoon,

    "fog": carla.WeatherParameters(
        cloudiness=80.0,
        fog_density=60.0,
        fog_distance=20.0,
        sun_altitude_angle=45.0
    ),
}


# ============================================================
# POSITION DES CAPTEURS
# ============================================================

CAMERA_TRANSFORM = carla.Transform(
    carla.Location(
        x=1.5,
        y=0.0,
        z=2.4
    )
)

LIDAR_TRANSFORM = carla.Transform(
    carla.Location(
        x=0.0,
        y=0.0,
        z=2.4
    )
)


# ============================================================
# CALIBRATION CAMERA
# ============================================================

def build_projection_matrix(width, height, fov):

    focal = width / (
        2.0 * np.tan(
            fov * np.pi / 360.0
        )
    )

    K = np.identity(3)

    K[0, 0] = focal
    K[1, 1] = focal

    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0

    return K


# ============================================================
# CALLBACK CAPTEURS
# ============================================================

def sensor_callback(data, sensor_queue, sensor_name):

    sensor_queue.put(
        (
            data.frame,
            sensor_name,
            data
        )
    )


# ============================================================
# PROJECTION MONDE -> CAMERA
# ============================================================

def world_to_camera_std(
    loc,
    camera,
    K,
    img_w,
    img_h,
    require_inside_image=True,
    near_plane=0.1
):
    """
    Projette un point monde CARLA dans le repere camera standard.

    IMPORTANT :
    - Un point devant la camera peut etre retourne meme s'il est hors image
      lorsque require_inside_image=False.
    - Cela evite de perdre des sommets 3D avant le calcul d'une bbox 2D.
    """

    cam_transform = camera.get_transform()

    world_to_cam = np.array(
        cam_transform.get_inverse_matrix(),
        dtype=np.float64
    )

    point_world = np.array(
        [
            loc.x,
            loc.y,
            loc.z,
            1.0
        ],
        dtype=np.float64
    )

    point_cam = world_to_cam @ point_world

    # CARLA : x avant, y droite, z haut
    # Camera standard : X droite, Y bas, Z avant
    point_std = np.array(
        [
            point_cam[1],
            -point_cam[2],
            point_cam[0]
        ],
        dtype=np.float64
    )

    depth = float(point_std[2])

    if depth <= near_plane:
        return None

    p2d = K @ point_std

    u = float(p2d[0] / p2d[2])
    v = float(p2d[1] / p2d[2])

    if require_inside_image:
        if not (
            0 <= u < img_w
            and
            0 <= v < img_h
        ):
            return None

    return u, v, depth


def world_to_camera_coordinates(loc, camera):
    """
    Retourne un point monde dans le repere camera standard :
    [X=droite, Y=bas, Z=avant].
    """

    world_to_cam = np.array(
        camera.get_transform().get_inverse_matrix(),
        dtype=np.float64
    )

    point_world = np.array(
        [loc.x, loc.y, loc.z, 1.0],
        dtype=np.float64
    )

    point_cam = world_to_cam @ point_world

    return np.array(
        [
            point_cam[1],
            -point_cam[2],
            point_cam[0]
        ],
        dtype=np.float64
    )


def project_camera_point(point_std, K):
    """Projette [X,Y,Z] avec Z strictement positif."""

    p2d = K @ point_std

    return np.array(
        [
            p2d[0] / p2d[2],
            p2d[1] / p2d[2]
        ],
        dtype=np.float64
    )


# Les 12 aretes d'une bounding box CARLA a 8 sommets.
# bbox.get_world_vertices() retourne les sommets dans un ordre stable ;
# pour eviter de dependre aveuglement de cet ordre, les aretes sont
# reconstruites ci-dessous a partir des coordonnees locales +/- extent.
BBOX_EDGE_PAIRS = (
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7)
)


def get_bbox_world_vertices_ordered(actor):
    """
    Construit explicitement les 8 sommets de la bbox dans l'ordre binaire :
      0=(-x,-y,-z), 1=(-x,-y,+z), 2=(-x,+y,-z), ...
    puis applique le transform complet bbox -> acteur -> monde.

    Cela prend bien en compte bbox.location ET bbox.rotation.
    """

    bbox = actor.bounding_box
    extent = bbox.extent

    local_vertices = []

    for sx, sy, sz in (
        (-1, -1, -1),
        (-1, -1, +1),
        (-1, +1, -1),
        (-1, +1, +1),
        (+1, -1, -1),
        (+1, -1, +1),
        (+1, +1, -1),
        (+1, +1, +1),
    ):
        local_vertices.append(
            carla.Location(
                x=sx * extent.x,
                y=sy * extent.y,
                z=sz * extent.z
            )
        )

    # Transform de la bbox relativement a l'acteur.
    bbox_transform = carla.Transform(
        bbox.location,
        bbox.rotation
    )

    actor_transform = actor.get_transform()

    world_vertices = []

    for vertex in local_vertices:
        # transform() modifie la Location en place.
        p = carla.Location(
            x=vertex.x,
            y=vertex.y,
            z=vertex.z
        )

        bbox_transform.transform(p)
        actor_transform.transform(p)

        world_vertices.append(p)

    return world_vertices


def get_2d_bbox_from_actor(
    actor,
    camera,
    K,
    img_w,
    img_h,
    near_plane=0.1
):
    """
    Calcule la bbox 2D visible d'un acteur.

    Correction V3 :
    1. ne supprime PAS les sommets simplement parce qu'ils sont hors image ;
    2. clippe les aretes 3D contre le near-plane camera ;
    3. projette ensuite les points valides ;
    4. calcule la bbox projetee ;
    5. intersecte finalement cette bbox avec l'image.

    Cette procedure est nettement plus robuste pour les vehicules proches,
    partiellement hors champ ou traversant le plan proche de la camera.
    """

    world_vertices = get_bbox_world_vertices_ordered(actor)

    camera_vertices = [
        world_to_camera_coordinates(vertex, camera)
        for vertex in world_vertices
    ]

    projected_points = []

    # Sommets deja devant le near-plane.
    for point in camera_vertices:
        if point[2] >= near_plane:
            projected_points.append(
                project_camera_point(point, K)
            )

    # Intersections des aretes avec le near-plane.
    # Necessaire si une bbox 3D entoure / traverse la camera.
    for index_a, index_b in BBOX_EDGE_PAIRS:
        a = camera_vertices[index_a]
        b = camera_vertices[index_b]

        za = float(a[2])
        zb = float(b[2])

        a_front = za >= near_plane
        b_front = zb >= near_plane

        if a_front == b_front:
            continue

        denominator = zb - za

        if abs(denominator) < 1e-12:
            continue

        t = (near_plane - za) / denominator

        if not (0.0 <= t <= 1.0):
            continue

        intersection = a + t * (b - a)

        # Petite marge numerique : la projection exige Z > 0.
        intersection[2] = max(
            float(intersection[2]),
            near_plane
        )

        projected_points.append(
            project_camera_point(intersection, K)
        )

    if len(projected_points) == 0:
        return None

    projected = np.asarray(
        projected_points,
        dtype=np.float64
    )

    finite_mask = np.isfinite(projected).all(axis=1)
    projected = projected[finite_mask]

    if len(projected) == 0:
        return None

    # Rectangle projete AVANT clipping image.
    xmin = float(np.min(projected[:, 0]))
    ymin = float(np.min(projected[:, 1]))
    xmax = float(np.max(projected[:, 0]))
    ymax = float(np.max(projected[:, 1]))

    # Rejet si le rectangle projete ne touche pas du tout l'image.
    if (
        xmax < 0.0
        or xmin > img_w - 1
        or ymax < 0.0
        or ymin > img_h - 1
    ):
        return None

    # Intersection finale avec l'image.
    xmin = float(np.clip(xmin, 0.0, img_w - 1.0))
    xmax = float(np.clip(xmax, 0.0, img_w - 1.0))
    ymin = float(np.clip(ymin, 0.0, img_h - 1.0))
    ymax = float(np.clip(ymax, 0.0, img_h - 1.0))

    if (
        xmax - xmin < 2.0
        or ymax - ymin < 2.0
    ):
        return None

    return [
        xmin,
        ymin,
        xmax,
        ymax
    ]


# ============================================================
# VIDER LA QUEUE
# ============================================================

def clear_sensor_queue(sensor_queue):

    while not sensor_queue.empty():

        try:
            sensor_queue.get_nowait()

        except queue.Empty:
            break


# ============================================================
# ATTENDRE CAMERA + LIDAR DU MEME FRAME
# ============================================================

def get_synchronized_sensor_data(
    sensor_queue,
    target_frame,
    timeout=2.0
):

    data_frame = {}

    while len(data_frame) < 2:

        frame_id, sensor_name, data = (
            sensor_queue.get(
                timeout=timeout
            )
        )

        # Ancienne frame
        if frame_id < target_frame:
            continue

        # Une frame future ne devrait normalement
        # pas arriver ici en mode synchrone
        if frame_id > target_frame:

            print(
                f"[WARN] {sensor_name}: "
                f"frame {frame_id}, "
                f"attendue {target_frame}"
            )

            continue

        data_frame[sensor_name] = data

    if (
        "camera" not in data_frame
        or
        "lidar" not in data_frame
    ):

        return None

    return data_frame


# ============================================================
# COLLECTE POUR UNE METEO
# ============================================================

def collect_frames_for_weather(
    client,
    world,
    weather_name,
    weather_params,
    blueprint_library,
    K,
    start_index,
    n_frames,
    tm_port
):

    world.set_weather(
        weather_params
    )

    print(
        f"\n======================================"
    )

    print(
        f"METEO : {weather_name}"
    )

    print(
        f"Frames : {n_frames}"
    )

    print(
        f"======================================"
    )

    actor_list = []
    npc_vehicles = []

    sensor_queue = queue.Queue()

    try:

        # ====================================================
        # SPAWN POINTS
        # ====================================================

        spawn_points = (
            world
            .get_map()
            .get_spawn_points()
        )

        random.shuffle(
            spawn_points
        )

        if len(spawn_points) == 0:
            raise RuntimeError(
                "Aucun spawn point disponible."
            )

        # ====================================================
        # EGO VEHICLE
        # ====================================================

        ego_bp = (
            blueprint_library
            .filter(
                "vehicle.tesla.model3"
            )[0]
        )

        ego_vehicle = world.spawn_actor(
            ego_bp,
            spawn_points[0]
        )

        ego_vehicle.set_autopilot(
            True,
            tm_port
        )

        actor_list.append(
            ego_vehicle
        )

        # ====================================================
        # TRAFIC
        # ====================================================

        vehicle_bps = (
            blueprint_library
            .filter("vehicle.*")
        )

        vehicle_bps = [
            bp
            for bp in vehicle_bps
            if int(
                bp.get_attribute(
                    "number_of_wheels"
                )
            ) == 4
        ]

        max_npc = min(
            N_TRAFFIC_VEHICLES,
            len(spawn_points) - 1
        )

        for spawn_point in (
            spawn_points[
                1:max_npc + 1
            ]
        ):

            bp = random.choice(
                vehicle_bps
            )

            # Couleur aleatoire si disponible
            if bp.has_attribute("color"):

                colors = (
                    bp
                    .get_attribute("color")
                    .recommended_values
                )

                if colors:

                    bp.set_attribute(
                        "color",
                        random.choice(colors)
                    )

            npc = world.try_spawn_actor(
                bp,
                spawn_point
            )

            if npc is not None:

                npc.set_autopilot(
                    True,
                    tm_port
                )

                npc_vehicles.append(
                    npc
                )

        print(
            f"Trafic : "
            f"{len(npc_vehicles)} vehicules"
        )

        # ====================================================
        # CAMERA
        # ====================================================

        camera_bp = (
            blueprint_library
            .find(
                "sensor.camera.rgb"
            )
        )

        camera_bp.set_attribute(
            "image_size_x",
            str(IMG_WIDTH)
        )

        camera_bp.set_attribute(
            "image_size_y",
            str(IMG_HEIGHT)
        )

        camera_bp.set_attribute(
            "fov",
            str(FOV)
        )

        camera = world.spawn_actor(
            camera_bp,
            CAMERA_TRANSFORM,
            attach_to=ego_vehicle
        )

        actor_list.append(
            camera
        )

        # ====================================================
        # LIDAR
        # ====================================================

        lidar_bp = (
            blueprint_library
            .find(
                "sensor.lidar.ray_cast"
            )
        )

        lidar_bp.set_attribute(
            "channels",
            str(LIDAR_CHANNELS)
        )

        lidar_bp.set_attribute(
            "points_per_second",
            str(LIDAR_POINTS_PER_SECOND)
        )

        lidar_bp.set_attribute(
            "rotation_frequency",
            str(
                1.0 /
                FIXED_DELTA_SECONDS
            )
        )

        lidar_bp.set_attribute(
            "range",
            str(LIDAR_RANGE)
        )

        lidar = world.spawn_actor(
            lidar_bp,
            LIDAR_TRANSFORM,
            attach_to=ego_vehicle
        )

        actor_list.append(
            lidar
        )

        # ====================================================
        # CALLBACKS
        # ====================================================

        camera.listen(
            lambda data:
            sensor_callback(
                data,
                sensor_queue,
                "camera"
            )
        )

        lidar.listen(
            lambda data:
            sensor_callback(
                data,
                sensor_queue,
                "lidar"
            )
        )

        # ====================================================
        # WARM-UP
        # ====================================================

        print(
            "Warm-up du trafic..."
        )

        for _ in range(30):
            world.tick()

        clear_sensor_queue(
            sensor_queue
        )

        print(
            "Debut collecte."
        )

        # ====================================================
        # COLLECTE
        # ====================================================

        frame_count = 0

        while frame_count < n_frames:

            target_frame = world.tick()

            try:

                data_frame = (
                    get_synchronized_sensor_data(
                        sensor_queue,
                        target_frame
                    )
                )

                if data_frame is None:
                    continue

                # ============================================
                # INDEX DATASET
                # ============================================

                global_idx = (
                    start_index
                    +
                    frame_count
                )

                frame_dir = os.path.join(
                    OUTPUT_DIR,
                    f"frame_{global_idx:05d}"
                )

                os.makedirs(
                    frame_dir,
                    exist_ok=True
                )

                # ============================================
                # RGB
                # ============================================

                img_data = (
                    data_frame["camera"]
                )

                img_array = np.frombuffer(
                    img_data.raw_data,
                    dtype=np.uint8
                )

                img_array = (
                    img_array
                    .reshape(
                        (
                            IMG_HEIGHT,
                            IMG_WIDTH,
                            4
                        )
                    )
                    [:, :, :3]
                )

                cv2.imwrite(
                    os.path.join(
                        frame_dir,
                        "rgb.png"
                    ),
                    img_array
                )

                # ============================================
                # LIDAR
                # ============================================

                lidar_data = (
                    data_frame["lidar"]
                )

                points = np.frombuffer(
                    lidar_data.raw_data,
                    dtype=np.float32
                ).reshape(
                    (-1, 4)
                )

                np.save(
                    os.path.join(
                        frame_dir,
                        "lidar.npy"
                    ),
                    points
                )

                # ============================================
                # ANNOTATIONS
                # ============================================

                annotations = []

                ego_transform = (
                    ego_vehicle
                    .get_transform()
                )

                ego_loc = (
                    ego_transform.location
                )

                all_actors = (
                    list(npc_vehicles)
                    +
                    list(
                        world
                        .get_actors()
                        .filter(
                            "walker.pedestrian.*"
                        )
                    )
                )

                for actor in all_actors:

                    if not actor.is_alive:
                        continue

                    actor_transform = (
                        actor.get_transform()
                    )

                    actor_loc = (
                        actor_transform.location
                    )

                    distance = (
                        actor_loc.distance(
                            ego_loc
                        )
                    )

                    # Hors portee LiDAR
                    if (
                        distance >
                        MAX_RENDER_DISTANCE
                    ):
                        continue

                    bbox_2d = (
                        get_2d_bbox_from_actor(
                            actor,
                            camera,
                            K,
                            IMG_WIDTH,
                            IMG_HEIGHT
                        )
                    )

                    if bbox_2d is None:
                        continue

                    bbox_3d = (
                        actor.bounding_box
                    )

                    annotations.append(
                        {
                            "type":
                                "vehicle"
                                if "vehicle"
                                in actor.type_id
                                else "pedestrian",

                            "type_id":
                                actor.type_id,

                            "bbox_2d":
                                bbox_2d,

                            "location_3d":
                                [
                                    actor_transform.location.x,
                                    actor_transform.location.y,
                                    actor_transform.location.z
                                ],

                            "extent_3d":
                                [
                                    bbox_3d.extent.x,
                                    bbox_3d.extent.y,
                                    bbox_3d.extent.z
                                ],

                            "bbox_location_3d":
                                [
                                    bbox_3d.location.x,
                                    bbox_3d.location.y,
                                    bbox_3d.location.z
                                ],

                            "bbox_rotation_3d":
                                [
                                    bbox_3d.rotation.pitch,
                                    bbox_3d.rotation.yaw,
                                    bbox_3d.rotation.roll
                                ],

                            "rotation_3d":
                                [
                                    actor_transform.rotation.pitch,
                                    actor_transform.rotation.yaw,
                                    actor_transform.rotation.roll
                                ],

                            "rotation_yaw":
                                actor_transform.rotation.yaw,

                            "distance":
                                float(distance)
                        }
                    )

                # ============================================
                # METADATA
                # ============================================

                metadata = {

                    "dataset_index":
                        global_idx,

                    "carla_frame":
                        int(target_frame),

                    "weather":
                        weather_name,

                    "ego": {

                        "location": [
                            ego_transform.location.x,
                            ego_transform.location.y,
                            ego_transform.location.z
                        ],

                        "rotation": [
                            ego_transform.rotation.pitch,
                            ego_transform.rotation.yaw,
                            ego_transform.rotation.roll
                        ]
                    },

                    "objects":
                        annotations
                }

                with open(
                    os.path.join(
                        frame_dir,
                        "annotations.json"
                    ),
                    "w"
                ) as f:

                    json.dump(
                        metadata,
                        f,
                        indent=2
                    )

                # ============================================
                # PROGRESSION
                # ============================================

                if frame_count % 100 == 0:

                    print(
                        f"Frame "
                        f"{global_idx:05d} | "
                        f"{weather_name:5s} | "
                        f"{len(points):5d} pts | "
                        f"{len(annotations):2d} objets"
                    )

                frame_count += 1

            except queue.Empty:

                print(
                    f"[TIMEOUT] "
                    f"frame CARLA "
                    f"{target_frame}"
                )

                continue

            except Exception as e:

                print(
                    "\n[ERREUR COLLECTE]"
                )

                print(
                    "Type :",
                    type(e).__name__
                )

                print(
                    "Message :",
                    repr(e)
                )

                raise

        return (
            start_index
            +
            frame_count
        )

    # ========================================================
    # NETTOYAGE
    # ========================================================

    finally:

        print(
            f"Nettoyage "
            f"({weather_name})..."
        )

        for actor in actor_list:

            if actor.is_alive:
                actor.destroy()

        for npc in npc_vehicles:

            if npc.is_alive:
                npc.destroy()


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(42)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # CONNEXION CARLA
    # ========================================================

    client = carla.Client(
        "localhost",
        2000
    )

    client.set_timeout(
        20.0
    )

    world = client.get_world()

    # ========================================================
    # TRAFFIC MANAGER
    # ========================================================

    traffic_manager = (
        client.get_trafficmanager(
            8000
        )
    )

    traffic_manager.set_synchronous_mode(
        True
    )

    tm_port = (
        traffic_manager.get_port()
    )

    # Quelques variations de conduite
    traffic_manager.set_global_distance_to_leading_vehicle(
        2.5
    )

    # ========================================================
    # MODE SYNCHRONE
    # ========================================================

    original_settings = (
        world.get_settings()
    )

    settings = (
        world.get_settings()
    )

    settings.synchronous_mode = True

    settings.fixed_delta_seconds = (
        FIXED_DELTA_SECONDS
    )

    world.apply_settings(
        settings
    )

    # ========================================================
    # BLUEPRINTS
    # ========================================================

    blueprint_library = (
        world.get_blueprint_library()
    )

    # ========================================================
    # INTRINSEQUES CAMERA
    # ========================================================

    K = build_projection_matrix(
        IMG_WIDTH,
        IMG_HEIGHT,
        FOV
    )

    # ========================================================
    # CALIB.JSON
    # ========================================================

    calibration = {

        "camera": {

            "K":
                K.tolist(),

            "img_width":
                IMG_WIDTH,

            "img_height":
                IMG_HEIGHT,

            "fov":
                FOV,

            "transform_ego": {

                "location": [
                    CAMERA_TRANSFORM.location.x,
                    CAMERA_TRANSFORM.location.y,
                    CAMERA_TRANSFORM.location.z
                ],

                "rotation": [
                    CAMERA_TRANSFORM.rotation.pitch,
                    CAMERA_TRANSFORM.rotation.yaw,
                    CAMERA_TRANSFORM.rotation.roll
                ]
            }
        },

        "lidar": {

            "range":
                LIDAR_RANGE,

            "channels":
                LIDAR_CHANNELS,

            "points_per_second":
                LIDAR_POINTS_PER_SECOND,

            "rotation_frequency":
                1.0 / FIXED_DELTA_SECONDS,

            "transform_ego": {

                "location": [
                    LIDAR_TRANSFORM.location.x,
                    LIDAR_TRANSFORM.location.y,
                    LIDAR_TRANSFORM.location.z
                ],

                "rotation": [
                    LIDAR_TRANSFORM.rotation.pitch,
                    LIDAR_TRANSFORM.rotation.yaw,
                    LIDAR_TRANSFORM.rotation.roll
                ]
            }
        },

        "collection": {

            "fixed_delta_seconds":
                FIXED_DELTA_SECONDS,

            "max_annotation_distance":
                MAX_RENDER_DISTANCE,

            "frames_per_weather":
                FRAMES_PER_WEATHER,

            "weather_presets":
                list(
                    WEATHER_PRESETS.keys()
                )
        }
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "calib.json"
        ),
        "w"
    ) as f:

        json.dump(
            calibration,
            f,
            indent=2
        )

    # ========================================================
    # COLLECTE
    # ========================================================

    start_time = time.time()

    try:

        start_index = 0

        for (
            weather_name,
            weather_params
        ) in WEATHER_PRESETS.items():

            start_index = (
                collect_frames_for_weather(
                    client,
                    world,
                    weather_name,
                    weather_params,
                    blueprint_library,
                    K,
                    start_index,
                    FRAMES_PER_WEATHER,
                    tm_port
                )
            )

        elapsed = (
            time.time()
            -
            start_time
        )

        print(
            "\n======================================"
        )

        print(
            "COLLECTE TERMINEE"
        )

        print(
            f"Frames : {start_index}"
        )

        print(
            f"Dossier : {OUTPUT_DIR}"
        )

        print(
            f"Temps : {elapsed / 60:.1f} min"
        )

        print(
            "======================================"
        )

    finally:

        traffic_manager.set_synchronous_mode(
            False
        )

        world.apply_settings(
            original_settings
        )

        print(
            "Nettoyage final termine."
        )


if __name__ == "__main__":
    main()
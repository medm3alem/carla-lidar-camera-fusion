import os
import json
import argparse

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset_full_v2"

DEFAULT_FRAME = 4992

# Association LiDAR / bbox
CROP_RATIO = 0.15
MIN_POINTS = 3
GAP_THRESHOLD = 2.0


# ============================================================
# CHARGEMENT CALIBRATION
# ============================================================

def load_calibration():

    calib_path = os.path.join(
        DATASET_DIR,
        "calib.json"
    )

    if not os.path.isfile(calib_path):
        raise FileNotFoundError(
            f"Calibration introuvable : {calib_path}"
        )

    with open(calib_path, "r") as f:
        calib = json.load(f)

    camera_calib = calib["camera"]

    K = np.array(
        camera_calib["K"],
        dtype=np.float32
    )

    width = camera_calib["img_width"]
    height = camera_calib["img_height"]

    return K, width, height, calib


# ============================================================
# CHARGEMENT FRAME
# ============================================================

def load_frame(frame_index):

    frame_dir = os.path.join(
        DATASET_DIR,
        f"frame_{frame_index:05d}"
    )

    rgb_path = os.path.join(
        frame_dir,
        "rgb.png"
    )

    lidar_path = os.path.join(
        frame_dir,
        "lidar.npy"
    )

    annotations_path = os.path.join(
        frame_dir,
        "annotations.json"
    )

    if not os.path.isdir(frame_dir):
        raise FileNotFoundError(
            f"Frame introuvable : {frame_dir}"
        )

    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(rgb_path)

    if not os.path.isfile(lidar_path):
        raise FileNotFoundError(lidar_path)

    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(annotations_path)

    image = cv2.imread(rgb_path)

    if image is None:
        raise RuntimeError(
            f"Impossible de lire : {rgb_path}"
        )

    points = np.load(lidar_path)

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    return image, points, annotations


# ============================================================
# PROJECTION LIDAR -> CAMERA
# ============================================================

def project_lidar_to_camera(
    points,
    K,
    width,
    height,
    calib
):

    xyz = points[:, :3].copy()

    # --------------------------------------------------------
    # Transformations relatives enregistrees dans calib.json
    # --------------------------------------------------------

    cam_loc = np.array(
        calib["camera"]["transform_ego"]["location"],
        dtype=np.float32
    )

    lidar_loc = np.array(
        calib["lidar"]["transform_ego"]["location"],
        dtype=np.float32
    )

    # LiDAR -> camera dans le repere CARLA
    #
    # Les deux capteurs ont actuellement la meme rotation,
    # donc une translation suffit pour ce dataset.
    translation = lidar_loc - cam_loc

    xyz = xyz + translation

    # --------------------------------------------------------
    # Convention CARLA -> convention camera
    #
    # CARLA :
    # x = forward
    # y = right
    # z = up
    #
    # Camera :
    # X = right
    # Y = down
    # Z = forward
    # --------------------------------------------------------

    X = xyz[:, 1]
    Y = -xyz[:, 2]
    Z = xyz[:, 0]

    camera_points = np.stack(
        [X, Y, Z],
        axis=1
    )

    # Garder uniquement les points devant la camera
    mask_front = Z > 0.1

    camera_points = camera_points[
        mask_front
    ]

    intensities = points[
        mask_front,
        3
    ]

    # --------------------------------------------------------
    # Projection perspective
    # --------------------------------------------------------

    projected = (
        K @ camera_points.T
    ).T

    u = (
        projected[:, 0]
        /
        projected[:, 2]
    )

    v = (
        projected[:, 1]
        /
        projected[:, 2]
    )

    depth = camera_points[:, 2]

    # --------------------------------------------------------
    # Points situes dans l'image
    # --------------------------------------------------------

    mask_image = (
        (u >= 0)
        &
        (u < width)
        &
        (v >= 0)
        &
        (v < height)
    )

    return (
        u[mask_image],
        v[mask_image],
        depth[mask_image],
        intensities[mask_image]
    )


# ============================================================
# CLUSTERING DES PROFONDEURS
# ============================================================

def cluster_depths(
    depths,
    gap_threshold=GAP_THRESHOLD
):

    if len(depths) == 0:
        return []

    depths = np.sort(depths)

    clusters = []

    current_cluster = [
        depths[0]
    ]

    for i in range(
        1,
        len(depths)
    ):

        gap = (
            depths[i]
            -
            depths[i - 1]
        )

        if gap > gap_threshold:

            clusters.append(
                np.array(
                    current_cluster
                )
            )

            current_cluster = []

        current_cluster.append(
            depths[i]
        )

    clusters.append(
        np.array(
            current_cluster
        )
    )

    return clusters


# ============================================================
# ASSOCIATION LIDAR -> BBOX
# ============================================================

def associate_lidar_to_bboxes(
    annotations,
    u,
    v,
    depth,
    min_points=MIN_POINTS,
    crop_ratio=CROP_RATIO,
    gap_threshold=GAP_THRESHOLD
):

    results = []

    objects = annotations.get(
        "objects",
        []
    )

    for obj in objects:

        bbox = obj.get(
            "bbox_2d"
        )

        if bbox is None:
            continue

        xmin, ymin, xmax, ymax = bbox

        # ----------------------------------------------------
        # Crop central
        # ----------------------------------------------------

        bbox_width = (
            xmax - xmin
        )

        bbox_height = (
            ymax - ymin
        )

        margin_x = (
            crop_ratio
            *
            bbox_width
        )

        margin_y = (
            crop_ratio
            *
            bbox_height
        )

        xmin_inner = (
            xmin + margin_x
        )

        xmax_inner = (
            xmax - margin_x
        )

        ymin_inner = (
            ymin + margin_y
        )

        ymax_inner = (
            ymax - margin_y
        )

        # ----------------------------------------------------
        # Points LiDAR projetes dans la zone centrale
        # ----------------------------------------------------

        mask = (
            (u >= xmin_inner)
            &
            (u <= xmax_inner)
            &
            (v >= ymin_inner)
            &
            (v <= ymax_inner)
        )

        depths_bbox = depth[
            mask
        ]

        result = {
            "object": obj,

            "n_lidar_points":
                len(depths_bbox),

            "n_cluster_points":
                0,

            "lidar_median_depth":
                None,

            "lidar_mean_depth":
                None,

            "lidar_min_depth":
                None
        }

        # ----------------------------------------------------
        # Pas assez de points
        # ----------------------------------------------------

        if len(depths_bbox) < min_points:

            results.append(
                result
            )

            continue

        # ----------------------------------------------------
        # Clustering
        # ----------------------------------------------------

        clusters = cluster_depths(
            depths_bbox,
            gap_threshold
        )

        valid_clusters = [
            cluster
            for cluster in clusters
            if len(cluster) >= min_points
        ]

        if len(valid_clusters) == 0:

            results.append(
                result
            )

            continue

        # ----------------------------------------------------
        # Cluster dominant
        #
        # IMPORTANT :
        # aucun Ground Truth n'est utilise ici.
        # ----------------------------------------------------

        best_cluster = max(
            valid_clusters,
            key=len
        )

        result[
            "n_cluster_points"
        ] = len(best_cluster)

        result[
            "lidar_median_depth"
        ] = float(
            np.median(
                best_cluster
            )
        )

        result[
            "lidar_mean_depth"
        ] = float(
            np.mean(
                best_cluster
            )
        )

        result[
            "lidar_min_depth"
        ] = float(
            np.min(
                best_cluster
            )
        )

        results.append(
            result
        )

    return results


# ============================================================
# GROUND TRUTH :
# WORLD -> EGO -> CAMERA
# ============================================================

def world_to_camera_depth(
    obj,
    annotations,
    calib
):

    # --------------------------------------------------------
    # Centre de l'objet dans le monde
    # --------------------------------------------------------

    obj_world = np.array(
        obj["location_3d"],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Pose ego
    # --------------------------------------------------------

    ego_location = np.array(
        annotations["ego"]["location"],
        dtype=np.float32
    )

    ego_rotation = annotations[
        "ego"
    ]["rotation"]

    # rotation sauvegardee :
    # [pitch, yaw, roll]
    ego_yaw = np.deg2rad(
        ego_rotation[1]
    )

    # --------------------------------------------------------
    # Translation world -> ego
    # --------------------------------------------------------

    delta = (
        obj_world
        -
        ego_location
    )

    # --------------------------------------------------------
    # Rotation inverse world -> ego
    # --------------------------------------------------------

    cos_yaw = np.cos(
        ego_yaw
    )

    sin_yaw = np.sin(
        ego_yaw
    )

    x_ego = (
        cos_yaw * delta[0]
        +
        sin_yaw * delta[1]
    )

    y_ego = (
        -sin_yaw * delta[0]
        +
        cos_yaw * delta[1]
    )

    z_ego = delta[2]

    point_ego = np.array(
        [
            x_ego,
            y_ego,
            z_ego
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Ego -> camera
    #
    # Pour notre dataset :
    # camera relative ego =
    # (x=1.5, y=0, z=2.4)
    #
    # rotation relative = 0
    # --------------------------------------------------------

    camera_location = np.array(
        calib[
            "camera"
        ][
            "transform_ego"
        ][
            "location"
        ],
        dtype=np.float32
    )

    point_camera_carla = (
        point_ego
        -
        camera_location
    )

    # --------------------------------------------------------
    # Convention CARLA -> camera
    # --------------------------------------------------------

    X = point_camera_carla[1]
    Y = -point_camera_carla[2]
    Z = point_camera_carla[0]

    camera_depth = float(
        Z
    )

    camera_distance = float(
        np.linalg.norm(
            point_camera_carla
        )
    )

    return (
        camera_depth,
        camera_distance
    )


# ============================================================
# AFFICHAGE POINTS LIDAR
# ============================================================

def draw_lidar_points(
    image,
    u,
    v,
    depth
):

    if len(depth) == 0:
        return image

    max_depth = 50.0

    for px, py, d in zip(
        u,
        v,
        depth
    ):

        px = int(px)
        py = int(py)

        # ----------------------------------------------------
        # Couleur selon profondeur
        # ----------------------------------------------------

        normalized = np.clip(
            d / max_depth,
            0.0,
            1.0
        )

        # proche -> rouge
        # loin   -> bleu
        red = int(
            255
            *
            (1.0 - normalized)
        )

        blue = int(
            255
            *
            normalized
        )

        green = int(
            255
            *
            (
                1.0
                -
                abs(
                    normalized
                    -
                    0.5
                )
                *
                2.0
            )
        )

        color = (
            blue,
            green,
            red
        )

        cv2.circle(
            image,
            (px, py),
            2,
            color,
            -1
        )

    return image


# ============================================================
# AFFICHAGE BBOX + FUSION
# ============================================================

def draw_fusion_bboxes(
    image,
    associations,
    annotations,
    calib
):

    for result in associations:

        obj = result[
            "object"
        ]

        bbox = obj.get(
            "bbox_2d"
        )

        if bbox is None:
            continue

        xmin, ymin, xmax, ymax = map(
            int,
            bbox
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # GT profondeur camera
        # ----------------------------------------------------

        gt_camera_depth, _ = (
            world_to_camera_depth(
                obj,
                annotations,
                calib
            )
        )

        lidar_depth = result[
            "lidar_median_depth"
        ]

        n_bbox = result[
            "n_lidar_points"
        ]

        n_cluster = result[
            "n_cluster_points"
        ]

        # ----------------------------------------------------
        # Texte
        # ----------------------------------------------------

        label1 = (
            f"{obj['type']} "
            f"GT Z:{gt_camera_depth:.1f}m"
        )

        if lidar_depth is not None:

            diff = (
                lidar_depth
                -
                gt_camera_depth
            )

            label2 = (
                f"L:{lidar_depth:.1f}m "
                f"d:{diff:+.1f}m "
                f"{n_cluster}/{n_bbox}pts"
            )

        else:

            label2 = (
                f"L:N/A "
                f"{n_bbox}pts"
            )

        # ----------------------------------------------------
        # Position texte
        # ----------------------------------------------------

        y1 = max(
            20,
            ymin - 25
        )

        y2 = max(
            20,
            ymin - 8
        )

        cv2.putText(
            image,
            label1,
            (xmin, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            label2,
            (xmin, y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    return image


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Visualisation fusion "
            "Camera + LiDAR CARLA"
        )
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=DEFAULT_FRAME,
        help="Index de la frame"
    )

    parser.add_argument(
        "--no-lidar",
        action="store_true",
        help="Ne pas afficher les points LiDAR"
    )

    parser.add_argument(
        "--no-bbox",
        action="store_true",
        help="Ne pas afficher les bounding boxes"
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Sauvegarder l'image"
    )

    args = parser.parse_args()

    # ========================================================
    # Calibration
    # ========================================================

    K, width, height, calib = (
        load_calibration()
    )

    # ========================================================
    # Frame
    # ========================================================

    image, points, annotations = (
        load_frame(
            args.frame
        )
    )

    # ========================================================
    # Informations
    # ========================================================

    print(
        f"\n=== Visualisation "
        f"frame {args.frame:05d} ==="
    )

    print(
        "\nMatrice K :"
    )

    print(K)

    print(
        f"\nPoints LiDAR : "
        f"{len(points)}"
    )

    print(
        f"Objets annotes : "
        f"{len(annotations.get('objects', []))}"
    )

    print(
        f"Meteo : "
        f"{annotations.get('weather', 'unknown')}"
    )

    # ========================================================
    # Projection
    # ========================================================

    u, v, depth, intensity = (
        project_lidar_to_camera(
            points,
            K,
            width,
            height,
            calib
        )
    )

    print(
        f"Points LiDAR projetes "
        f"dans l'image : {len(u)}"
    )

    # ========================================================
    # Association
    # ========================================================

    associations = (
        associate_lidar_to_bboxes(
            annotations,
            u,
            v,
            depth
        )
    )

    # ========================================================
    # RESULTATS CONSOLE
    # ========================================================

    print(
        "\n=== Association LiDAR / objets ==="
    )

    for i, result in enumerate(
        associations
    ):

        obj = result[
            "object"
        ]

        lidar_dist = result[
            "lidar_median_depth"
        ]

        n_bbox = result[
            "n_lidar_points"
        ]

        n_cluster = result[
            "n_cluster_points"
        ]

        gt_camera_depth, gt_camera_distance = (
            world_to_camera_depth(
                obj,
                annotations,
                calib
            )
        )

        if lidar_dist is not None:

            diff = (
                lidar_dist
                -
                gt_camera_depth
            )

            print(
                f"Objet {i:02d} | "
                f"{obj['type']:10s} | "
                f"GT center Z="
                f"{gt_camera_depth:5.1f} m | "
                f"LiDAR="
                f"{lidar_dist:5.1f} m | "
                f"diff="
                f"{diff:+5.1f} m | "
                f"cluster="
                f"{n_cluster}/{n_bbox} pts"
            )

        else:

            print(
                f"Objet {i:02d} | "
                f"{obj['type']:10s} | "
                f"GT center Z="
                f"{gt_camera_depth:5.1f} m | "
                f"LiDAR=N/A | "
                f"{n_bbox} pts"
            )

    # ========================================================
    # VISUALISATION
    # ========================================================

    if not args.no_bbox:

        image = draw_fusion_bboxes(
            image,
            associations,
            annotations,
            calib
        )

    if not args.no_lidar:

        image = draw_lidar_points(
            image,
            u,
            v,
            depth
        )

    # ========================================================
    # SAUVEGARDE OPTIONNELLE
    # ========================================================

    if args.save is not None:

        cv2.imwrite(
            args.save,
            image
        )

        print(
            f"\nImage sauvegardee : "
            f"{args.save}"
        )

    # ========================================================
    # AFFICHAGE
    # ========================================================

    cv2.imshow(
        "CARLA Camera-LiDAR Fusion",
        image
    )

    print(
        "\nAppuyez sur une touche "
        "dans la fenetre pour fermer."
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
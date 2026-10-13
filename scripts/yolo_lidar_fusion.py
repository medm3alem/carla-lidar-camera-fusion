import os
import argparse

import cv2
import numpy as np
from ultralytics import YOLO

from visualize_fusion import (
    load_calibration,
    load_frame,
    project_lidar_to_camera,
    cluster_depths
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset_full_v2"

MODEL_NAME = "yolo26n.pt"

CONF_THRESHOLD = 0.25

VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle"
}

# Association LiDAR
CROP_RATIO = 0.15
MIN_POINTS = 3
GAP_THRESHOLD = 2.0


# ============================================================
# ASSOCIATION LIDAR A UNE BBOX YOLO
# ============================================================

def associate_lidar_to_yolo_bbox(
    bbox,
    u,
    v,
    depth,
    crop_ratio=CROP_RATIO,
    min_points=MIN_POINTS,
    gap_threshold=GAP_THRESHOLD
):

    x1, y1, x2, y2 = bbox

    # --------------------------------------------------------
    # Dimensions bbox
    # --------------------------------------------------------

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return None

    # --------------------------------------------------------
    # Zone centrale
    #
    # On retire 15 % sur chaque bord pour diminuer
    # la contamination par le fond.
    # --------------------------------------------------------

    margin_x = crop_ratio * width
    margin_y = crop_ratio * height

    inner_x1 = x1 + margin_x
    inner_x2 = x2 - margin_x

    inner_y1 = y1 + margin_y
    inner_y2 = y2 - margin_y

    # --------------------------------------------------------
    # Points LiDAR projetes dans la bbox
    # --------------------------------------------------------

    mask = (
        (u >= inner_x1)
        &
        (u <= inner_x2)
        &
        (v >= inner_y1)
        &
        (v <= inner_y2)
    )

    depths_bbox = depth[mask]

    result = {
        "n_bbox_points": len(depths_bbox),
        "n_cluster_points": 0,
        "depth": None,
        "mean_depth": None,
        "min_depth": None
    }

    # --------------------------------------------------------
    # Pas assez de points
    # --------------------------------------------------------

    if len(depths_bbox) < min_points:
        return result

    # --------------------------------------------------------
    # Clustering en profondeur
    # --------------------------------------------------------

    clusters = cluster_depths(
        depths_bbox,
        gap_threshold=gap_threshold
    )

    valid_clusters = [
        cluster
        for cluster in clusters
        if len(cluster) >= min_points
    ]

    if len(valid_clusters) == 0:
        return result

    # --------------------------------------------------------
    # Cluster dominant
    # --------------------------------------------------------

    best_cluster = max(
        valid_clusters,
        key=len
    )

    result["n_cluster_points"] = len(
        best_cluster
    )

    result["depth"] = float(
        np.median(best_cluster)
    )

    result["mean_depth"] = float(
        np.mean(best_cluster)
    )

    result["min_depth"] = float(
        np.min(best_cluster)
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Fusion YOLO + LiDAR sur dataset CARLA"
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=4992
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None
    )

    args = parser.parse_args()

    # ========================================================
    # CALIBRATION
    # ========================================================

    K, width, height, calib = (
        load_calibration()
    )

    # ========================================================
    # CHARGEMENT FRAME
    # ========================================================

    image, points, annotations = (
        load_frame(args.frame)
    )

    # Image originale pour YOLO
    yolo_image = image.copy()

    # ========================================================
    # PROJECTION LIDAR
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

    print("\n========================================")
    print(" FUSION YOLO + LIDAR")
    print("========================================")

    print(
        f"Frame             : {args.frame:05d}"
    )

    print(
        f"Meteo             : "
        f"{annotations.get('weather', 'unknown')}"
    )

    print(
        f"Points LiDAR      : {len(points)}"
    )

    print(
        f"Points projetes   : {len(u)}"
    )

    # ========================================================
    # YOLO
    # ========================================================

    print("\nChargement YOLO...")

    model = YOLO(
        MODEL_NAME
    )

    print("Inference...")

    results = model.predict(
        source=yolo_image,
        conf=args.conf,
        verbose=False
    )

    result = results[0]

    # ========================================================
    # DETECTIONS
    # ========================================================

    detections = []

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        class_name = model.names[
            class_id
        ]

        if class_name not in VEHICLE_CLASSES:
            continue

        bbox = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        x1, y1, x2, y2 = bbox

        # ----------------------------------------------------
        # Association LiDAR
        # ----------------------------------------------------

        lidar_result = (
            associate_lidar_to_yolo_bbox(
                bbox,
                u,
                v,
                depth
            )
        )

        detections.append({
            "class_name": class_name,
            "confidence": confidence,
            "bbox": bbox,
            "lidar": lidar_result
        })

    # ========================================================
    # RESULTATS
    # ========================================================

    print("\n========================================")
    print(" DETECTIONS FUSIONNEES")
    print("========================================")

    for i, detection in enumerate(
        detections
    ):

        class_name = detection[
            "class_name"
        ]

        confidence = detection[
            "confidence"
        ]

        lidar_result = detection[
            "lidar"
        ]

        n_bbox = lidar_result[
            "n_bbox_points"
        ]

        n_cluster = lidar_result[
            "n_cluster_points"
        ]

        lidar_depth = lidar_result[
            "depth"
        ]

        if lidar_depth is not None:

            print(
                f"{i:02d} | "
                f"{class_name:10s} | "
                f"conf={confidence:.3f} | "
                f"distance={lidar_depth:5.1f} m | "
                f"cluster={n_cluster}/{n_bbox} pts"
            )

        else:

            print(
                f"{i:02d} | "
                f"{class_name:10s} | "
                f"conf={confidence:.3f} | "
                f"distance=N/A | "
                f"{n_bbox} pts"
            )

    print("----------------------------------------")

    print(
        f"Vehicules YOLO : {len(detections)}"
    )

    n_with_lidar = sum(
        1
        for detection in detections
        if detection["lidar"]["depth"] is not None
    )

    print(
        f"Avec distance   : {n_with_lidar}"
    )

    # ========================================================
    # VISUALISATION POINTS LIDAR
    # ========================================================

    max_depth = 50.0

    for px, py, d in zip(
        u,
        v,
        depth
    ):

        normalized = np.clip(
            d / max_depth,
            0.0,
            1.0
        )

        red = int(
            255 * (1.0 - normalized)
        )

        blue = int(
            255 * normalized
        )

        green = int(
            255
            *
            (
                1.0
                -
                abs(normalized - 0.5)
                *
                2.0
            )
        )

        cv2.circle(
            image,
            (int(px), int(py)),
            2,
            (blue, green, red),
            -1
        )

    # ========================================================
    # VISUALISATION YOLO + DISTANCE
    # ========================================================

    for i, detection in enumerate(
        detections
    ):

        x1, y1, x2, y2 = (
            detection["bbox"]
            .astype(int)
        )

        class_name = detection[
            "class_name"
        ]

        confidence = detection[
            "confidence"
        ]

        lidar_result = detection[
            "lidar"
        ]

        lidar_depth = lidar_result[
            "depth"
        ]

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Texte
        # ----------------------------------------------------

        if lidar_depth is not None:

            label = (
                f"{class_name} "
                f"{confidence:.2f} | "
                f"{lidar_depth:.1f}m"
            )

        else:

            label = (
                f"{class_name} "
                f"{confidence:.2f} | "
                f"LiDAR N/A"
            )

        cv2.putText(
            image,
            label,
            (
                x1,
                max(20, y1 - 8)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # ========================================================
    # INFO FRAME
    # ========================================================

    info = (
        f"Frame {args.frame:05d} | "
        f"{annotations.get('weather', 'unknown')}"
    )

    cv2.putText(
        image,
        info,
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # ========================================================
    # SAUVEGARDE
    # ========================================================

    if args.save is not None:

        cv2.imwrite(
            args.save,
            image
        )

        print(
            f"\nImage sauvegardee : {args.save}"
        )

    # ========================================================
    # AFFICHAGE
    # ========================================================

    cv2.imshow(
        "YOLO + LiDAR Fusion",
        image
    )

    print(
        "\nAppuyez sur une touche pour fermer."
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
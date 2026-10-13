import os
import json
import glob
import csv
import argparse

import cv2
import numpy as np
from ultralytics import YOLO

from visualize_fusion import (
    load_calibration,
    project_lidar_to_camera,
    world_to_camera_depth
)

from yolo_lidar_fusion import (
    associate_lidar_to_yolo_bbox
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset_full_v3"

MODEL_NAME = "yolo26n.pt"

YOLO_CONF = 0.25
IOU_THRESHOLD = 0.50

OUTPUT_CSV = "yolo_fusion_results.csv"

VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle"
}

DISTANCE_BINS = [
    (0, 10),
    (10, 20),
    (20, 30),
    (30, 40),
    (40, 50)
]


# ============================================================
# IoU
# ============================================================

def compute_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)

    intersection = (
        intersection_width
        *
        intersection_height
    )

    area1 = max(
        0.0,
        (box1[2] - box1[0])
        *
        (box1[3] - box1[1])
    )

    area2 = max(
        0.0,
        (box2[2] - box2[0])
        *
        (box2[3] - box2[1])
    )

    union = (
        area1
        +
        area2
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# MATCHING YOLO <-> GT
# ============================================================

def match_yolo_to_gt(
    detections,
    gt_objects,
    iou_threshold=IOU_THRESHOLD
):
    """
    Matching greedy 1-to-1.

    Une bbox GT ne peut etre associee qu'a une seule
    detection YOLO.
    """

    candidates = []

    for det_idx, detection in enumerate(detections):

        yolo_bbox = detection["bbox"]

        for gt_idx, obj in enumerate(gt_objects):

            gt_bbox = obj.get("bbox_2d")

            if gt_bbox is None:
                continue

            iou = compute_iou(
                yolo_bbox,
                gt_bbox
            )

            if iou >= iou_threshold:

                candidates.append(
                    (
                        iou,
                        det_idx,
                        gt_idx
                    )
                )

    # Meilleures IoU d'abord
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    used_detections = set()
    used_gt = set()

    matches = []

    for iou, det_idx, gt_idx in candidates:

        if det_idx in used_detections:
            continue

        if gt_idx in used_gt:
            continue

        used_detections.add(
            det_idx
        )

        used_gt.add(
            gt_idx
        )

        matches.append({
            "detection_index": det_idx,
            "gt_index": gt_idx,
            "iou": iou
        })

    return (
        matches,
        used_detections,
        used_gt
    )


# ============================================================
# DISTANCE BIN
# ============================================================

def get_distance_bin(distance):

    for dmin, dmax in DISTANCE_BINS:

        if dmin <= distance < dmax:
            return f"{dmin}-{dmax}"

    return None


# ============================================================
# METRIQUES ERREUR
# ============================================================

def compute_error_metrics(errors):

    if len(errors) == 0:
        return None

    errors = np.asarray(
        errors,
        dtype=np.float32
    )

    return {
        "mae": float(
            np.mean(np.abs(errors))
        ),
        "rmse": float(
            np.sqrt(
                np.mean(errors ** 2)
            )
        ),
        "bias": float(
            np.mean(errors)
        ),
        "median": float(
            np.median(errors)
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--conf",
        type=float,
        default=YOLO_CONF
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=IOU_THRESHOLD
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limiter le nombre de frames pour un test"
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Evaluer uniquement une frame"
    )

    parser.add_argument(
        "--sample-step",
        type=int,
        default=None,
        help="Prendre une frame toutes les N frames"
    )


    args = parser.parse_args()

    # ========================================================
    # CALIBRATION
    # ========================================================

    K, width, height, calib = (
        load_calibration()
    )

    # ========================================================
    # YOLO
    # ========================================================

    print("\nChargement YOLO...")

    model = YOLO(
        MODEL_NAME
    )

    # ========================================================
    # FRAMES
    # ========================================================

    frame_dirs = sorted(
        glob.glob(
            os.path.join(
                DATASET_DIR,
                "frame_*"
            )
        )
    )



    if args.frame is not None:

        target = os.path.join(
            DATASET_DIR,
            f"frame_{args.frame:05d}"
        )

        if not os.path.isdir(target):
            raise FileNotFoundError(
                f"Frame introuvable : {target}"
            )

        frame_dirs = [target]

    elif args.sample_step is not None:

        frame_dirs = frame_dirs[::args.sample_step]

    elif args.max_frames is not None:

        frame_dirs = frame_dirs[:args.max_frames]



    print("\n========================================")
    print(" EVALUATION YOLO + LIDAR")
    print("========================================")

    print(
        f"Frames : {len(frame_dirs)}"
    )

    print(
        f"YOLO confidence : {args.conf}"
    )

    print(
        f"IoU matching    : {args.iou}"
    )

    # ========================================================
    # COMPTEURS
    # ========================================================

    total_gt = 0

    total_yolo = 0

    total_matches = 0

    total_false_positives = 0

    total_false_negatives = 0

    total_lidar_available = 0

    depth_errors = []

    rows = []

    # ========================================================
    # STATS METEO
    # ========================================================

    weather_stats = {}

    # ========================================================
    # STATS DISTANCE
    # ========================================================

    distance_stats = {
        f"{a}-{b}": {
            "gt": 0,
            "detected": 0,
            "lidar": 0,
            "errors": []
        }
        for a, b in DISTANCE_BINS
    }

    # ========================================================
    # DATASET
    # ========================================================

    for frame_idx, frame_dir in enumerate(
        frame_dirs
    ):

        frame_name = os.path.basename(
            frame_dir
        )

        frame_number = int(
            frame_name.split("_")[1]
        )

        image_path = os.path.join(
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

        if not (
            os.path.isfile(image_path)
            and
            os.path.isfile(lidar_path)
            and
            os.path.isfile(annotations_path)
        ):
            continue

        # ----------------------------------------------------
        # Chargement
        # ----------------------------------------------------

        image = cv2.imread(
            image_path
        )

        points = np.load(
            lidar_path
        )

        with open(
            annotations_path,
            "r"
        ) as f:

            annotations = json.load(f)

        weather = annotations.get(
            "weather",
            "unknown"
        )

        gt_objects = [
            obj
            for obj in annotations.get(
                "objects",
                []
            )
            if obj.get("bbox_2d") is not None
        ]

        # ----------------------------------------------------
        # Initialisation meteo
        # ----------------------------------------------------

        if weather not in weather_stats:

            weather_stats[weather] = {
                "frames": 0,
                "gt": 0,
                "yolo": 0,
                "matched": 0,
                "fp": 0,
                "fn": 0,
                "lidar": 0,
                "errors": []
            }

        ws = weather_stats[
            weather
        ]

        ws["frames"] += 1

        # ====================================================
        # GT
        # ====================================================

        total_gt += len(
            gt_objects
        )

        ws["gt"] += len(
            gt_objects
        )

        # Distance bins pour TOUS les GT
        for obj in gt_objects:

            gt_distance = obj.get(
                "distance"
            )

            if gt_distance is None:
                continue

            bin_name = get_distance_bin(
                gt_distance
            )

            if bin_name is not None:

                distance_stats[
                    bin_name
                ]["gt"] += 1

        # ====================================================
        # PROJECTION LIDAR
        # ====================================================

        u, v, depth, intensity = (
            project_lidar_to_camera(
                points,
                K,
                width,
                height,
                calib
            )
        )

        # ====================================================
        # YOLO
        # ====================================================

        predictions = model.predict(
            source=image,
            conf=args.conf,
            verbose=False
        )[0]

        detections = []

        for box in predictions.boxes:

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
                .astype(np.float32)
            )

            lidar_result = (
                associate_lidar_to_yolo_bbox(
                    bbox,
                    u,
                    v,
                    depth
                )
            )

            detections.append({
                "bbox": bbox,
                "class_name": class_name,
                "confidence": confidence,
                "lidar": lidar_result
            })

        total_yolo += len(
            detections
        )

        ws["yolo"] += len(
            detections
        )

        # ====================================================
        # MATCHING YOLO <-> GT
        # ====================================================



        if args.frame is not None:

            print("\n=== BBOX GT ===")

            for i, obj in enumerate(gt_objects):
                print(
                    f"GT {i:02d}: "
                    f"{obj['bbox_2d']} | "
                    f"distance={obj.get('distance')}"
                )

            print("\n=== BBOX YOLO ===")

            for i, det in enumerate(detections):
                print(
                    f"YOLO {i:02d}: "
                    f"{det['bbox']} | "
                    f"conf={det['confidence']:.3f}"
                )

            print("\n=== MATRICE IoU ===")

            for det_idx, det in enumerate(detections):

                for gt_idx, obj in enumerate(gt_objects):

                    iou = compute_iou(
                        det["bbox"],
                        obj["bbox_2d"]
                    )

                    print(
                        f"YOLO {det_idx:02d} "
                        f"<-> GT {gt_idx:02d} : "
                        f"IoU={iou:.3f}"
                    )




        (
            matches,
            used_detections,
            used_gt
        ) = match_yolo_to_gt(
            detections,
            gt_objects,
            args.iou
        )

        n_matches = len(
            matches
        )

        n_fp = (
            len(detections)
            -
            n_matches
        )

        n_fn = (
            len(gt_objects)
            -
            n_matches
        )

        total_matches += n_matches

        total_false_positives += n_fp

        total_false_negatives += n_fn

        ws["matched"] += n_matches
        ws["fp"] += n_fp
        ws["fn"] += n_fn

        # ====================================================
        # MATCHES
        # ====================================================

        for match in matches:

            det_idx = match[
                "detection_index"
            ]

            gt_idx = match[
                "gt_index"
            ]

            iou = match[
                "iou"
            ]

            detection = detections[
                det_idx
            ]

            obj = gt_objects[
                gt_idx
            ]

            # ------------------------------------------------
            # GT
            # ------------------------------------------------

            gt_distance = obj.get(
                "distance"
            )

            gt_camera_depth, _ = (
                world_to_camera_depth(
                    obj,
                    annotations,
                    calib
                )
            )

            bin_name = get_distance_bin(
                gt_distance
            )

            if bin_name is not None:

                distance_stats[
                    bin_name
                ]["detected"] += 1

            # ------------------------------------------------
            # LiDAR
            # ------------------------------------------------

            lidar_result = detection[
                "lidar"
            ]

            lidar_depth = lidar_result[
                "depth"
            ]

            n_bbox_points = lidar_result[
                "n_bbox_points"
            ]

            n_cluster_points = lidar_result[
                "n_cluster_points"
            ]

            error = None

            if lidar_depth is not None:

                total_lidar_available += 1

                ws["lidar"] += 1

                if bin_name is not None:

                    distance_stats[
                        bin_name
                    ]["lidar"] += 1

                # --------------------------------------------
                # Attention :
                # LiDAR = surface visible
                # GT Z = centre de l'objet
                #
                # On garde cette metrique pour l'instant
                # comme comparaison geometrique.
                # --------------------------------------------

                error = (
                    lidar_depth
                    -
                    gt_camera_depth
                )

                depth_errors.append(
                    error
                )

                ws["errors"].append(
                    error
                )

                if bin_name is not None:

                    distance_stats[
                        bin_name
                    ]["errors"].append(
                        error
                    )

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            rows.append({
                "frame": frame_number,
                "weather": weather,

                "gt_index": gt_idx,
                "gt_distance": gt_distance,
                "gt_camera_depth": gt_camera_depth,

                "yolo_class":
                    detection["class_name"],

                "yolo_confidence":
                    detection["confidence"],

                "iou": iou,

                "lidar_depth":
                    ""
                    if lidar_depth is None
                    else lidar_depth,

                "depth_error":
                    ""
                    if error is None
                    else error,

                "abs_depth_error":
                    ""
                    if error is None
                    else abs(error),

                "n_bbox_points":
                    n_bbox_points,

                "n_cluster_points":
                    n_cluster_points
            })

        # ====================================================
        # PROGRESSION
        # ====================================================

        if (
            frame_idx % 100 == 0
            or
            frame_idx
            ==
            len(frame_dirs) - 1
        ):

            print(
                f"Analyse : "
                f"{frame_idx + 1}/"
                f"{len(frame_dirs)}"
            )

    # ========================================================
    # CSV
    # ========================================================

    fieldnames = [
        "frame",
        "weather",
        "gt_index",
        "gt_distance",
        "gt_camera_depth",
        "yolo_class",
        "yolo_confidence",
        "iou",
        "lidar_depth",
        "depth_error",
        "abs_depth_error",
        "n_bbox_points",
        "n_cluster_points"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # RESULTATS YOLO
    # ========================================================

    precision = (
        total_matches
        /
        (
            total_matches
            +
            total_false_positives
        )
        if (
            total_matches
            +
            total_false_positives
        ) > 0
        else 0
    )

    recall = (
        total_matches
        /
        (
            total_matches
            +
            total_false_negatives
        )
        if (
            total_matches
            +
            total_false_negatives
        ) > 0
        else 0
    )

    f1 = (
        2
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )
        if (
            precision
            +
            recall
        ) > 0
        else 0
    )

    # ========================================================
    # RESULTATS
    # ========================================================

    print("\n========================================")
    print(" DETECTION YOLO")
    print("========================================")

    print(
        f"GT vehicules      : {total_gt}"
    )

    print(
        f"Detections YOLO   : {total_yolo}"
    )

    print(
        f"Matches IoU       : {total_matches}"
    )

    print(
        f"False positives   : "
        f"{total_false_positives}"
    )

    print(
        f"False negatives   : "
        f"{total_false_negatives}"
    )

    print(
        f"Precision         : "
        f"{precision * 100:.1f}%"
    )

    print(
        f"Recall            : "
        f"{recall * 100:.1f}%"
    )

    print(
        f"F1                : "
        f"{f1 * 100:.1f}%"
    )

    # ========================================================
    # FUSION
    # ========================================================

    print("\n========================================")
    print(" FUSION LIDAR")
    print("========================================")

    fusion_rate = (
        total_lidar_available
        /
        total_matches
        if total_matches > 0
        else 0
    )

    print(
        f"Objets detectes   : "
        f"{total_matches}"
    )

    print(
        f"Distance LiDAR    : "
        f"{total_lidar_available}"
    )

    print(
        f"Taux fusion       : "
        f"{fusion_rate * 100:.1f}%"
    )

    metrics = compute_error_metrics(
        depth_errors
    )

    if metrics is not None:

        print(
            f"MAE               : "
            f"{metrics['mae']:.2f} m"
        )

        print(
            f"RMSE              : "
            f"{metrics['rmse']:.2f} m"
        )

        print(
            f"Biais              : "
            f"{metrics['bias']:+.2f} m"
        )

        print(
            f"Erreur mediane    : "
            f"{metrics['median']:+.2f} m"
        )

    # ========================================================
    # METEO
    # ========================================================

    print("\n========================================")
    print(" PAR METEO")
    print("========================================")

    for weather, stats in weather_stats.items():

        p = (
            stats["matched"]
            /
            (
                stats["matched"]
                +
                stats["fp"]
            )
            if (
                stats["matched"]
                +
                stats["fp"]
            ) > 0
            else 0
        )

        r = (
            stats["matched"]
            /
            (
                stats["matched"]
                +
                stats["fn"]
            )
            if (
                stats["matched"]
                +
                stats["fn"]
            ) > 0
            else 0
        )

        fusion = (
            stats["lidar"]
            /
            stats["matched"]
            if stats["matched"] > 0
            else 0
        )

        metrics = compute_error_metrics(
            stats["errors"]
        )

        print(
            f"\n{weather.upper()}"
        )

        print(
            f"  Frames      : "
            f"{stats['frames']}"
        )

        print(
            f"  GT          : "
            f"{stats['gt']}"
        )

        print(
            f"  YOLO        : "
            f"{stats['yolo']}"
        )

        print(
            f"  Matches     : "
            f"{stats['matched']}"
        )

        print(
            f"  Precision   : "
            f"{p * 100:.1f}%"
        )

        print(
            f"  Recall      : "
            f"{r * 100:.1f}%"
        )

        print(
            f"  Fusion LiDAR: "
            f"{fusion * 100:.1f}%"
        )

        if metrics is not None:

            print(
                f"  MAE depth   : "
                f"{metrics['mae']:.2f} m"
            )

            print(
                f"  RMSE depth  : "
                f"{metrics['rmse']:.2f} m"
            )

            print(
                f"  Biais depth : "
                f"{metrics['bias']:+.2f} m"
            )

    # ========================================================
    # DISTANCE
    # ========================================================

    print("\n========================================")
    print(" PAR DISTANCE")
    print("========================================")

    print(
        "\nDistance | GT | YOLO recall | "
        "Fusion | MAE | RMSE"
    )

    print(
        "-" * 65
    )

    for bin_name, stats in distance_stats.items():

        gt = stats["gt"]

        detected = stats[
            "detected"
        ]

        lidar = stats[
            "lidar"
        ]

        detection_rate = (
            detected / gt
            if gt > 0
            else 0
        )

        fusion_rate = (
            lidar / detected
            if detected > 0
            else 0
        )

        metrics = compute_error_metrics(
            stats["errors"]
        )

        if metrics is not None:

            print(
                f"{bin_name:7s}m | "
                f"{gt:4d} | "
                f"{detection_rate * 100:10.1f}% | "
                f"{fusion_rate * 100:5.1f}% | "
                f"{metrics['mae']:4.2f} | "
                f"{metrics['rmse']:5.2f}"
            )

        else:

            print(
                f"{bin_name:7s}m | "
                f"{gt:4d} | "
                f"{detection_rate * 100:10.1f}% | "
                f"{fusion_rate * 100:5.1f}% | "
                f" N/A | N/A"
            )

    print(
        f"\nCSV sauvegarde : {OUTPUT_CSV}"
    )

    print("\n========================================")
    print(" EVALUATION TERMINEE")
    print("========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
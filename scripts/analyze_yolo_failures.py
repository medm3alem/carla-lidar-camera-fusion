import os
import json
import glob
import argparse

import cv2
import numpy as np
from ultralytics import YOLO

from evaluate_yolo_fusion import (
    compute_iou,
    match_yolo_to_gt,
    get_distance_bin
)

from visualize_fusion import (
    load_calibration,
    world_to_camera_depth
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset_full_v3"
MODEL_NAME = "yolo26n.pt"

YOLO_CONF = 0.25
IOU_THRESHOLD = 0.50

OUTPUT_DIR = "yolo_failure_analysis"

VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle"
}


# ============================================================
# DETECTIONS YOLO
# ============================================================

def run_yolo(model, image, confidence):

    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )[0]

    detections = []

    for box in results.boxes:

        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        class_name = model.names[class_id]

        if class_name not in VEHICLE_CLASSES:
            continue

        bbox = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        detections.append({
            "bbox": bbox,
            "class_name": class_name,
            "confidence": conf
        })

    return detections


# ============================================================
# MEILLEURE IoU POUR UN GT
# ============================================================

def find_best_detection(gt_bbox, detections):

    best_iou = 0.0
    best_detection = None

    for detection in detections:

        iou = compute_iou(
            gt_bbox,
            detection["bbox"]
        )

        if iou > best_iou:

            best_iou = iou
            best_detection = detection

    return best_iou, best_detection


# ============================================================
# DESSIN
# ============================================================

def draw_failure(
    image,
    gt_bbox,
    gt_distance,
    best_iou,
    best_detection,
    frame_number,
    weather
):

    output = image.copy()

    # --------------------------------------------------------
    # GT : rouge
    # --------------------------------------------------------

    x1, y1, x2, y2 = map(
        int,
        gt_bbox
    )

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        3
    )

    gt_label = (
        f"GT {gt_distance:.1f}m"
    )

    cv2.putText(
        output,
        gt_label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Meilleure detection YOLO : vert
    # --------------------------------------------------------

    if best_detection is not None:

        bx1, by1, bx2, by2 = map(
            int,
            best_detection["bbox"]
        )

        cv2.rectangle(
            output,
            (bx1, by1),
            (bx2, by2),
            (0, 255, 0),
            2
        )

        label = (
            f"YOLO "
            f"{best_detection['confidence']:.2f} "
            f"IoU={best_iou:.2f}"
        )

        cv2.putText(
            output,
            label,
            (
                bx1,
                min(
                    output.shape[0] - 10,
                    by2 + 20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Informations
    # --------------------------------------------------------

    info = (
        f"Frame {frame_number:05d} | "
        f"{weather} | "
        f"GT={gt_distance:.1f}m | "
        f"best IoU={best_iou:.2f}"
    )

    cv2.putText(
        output,
        info,
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output





def compute_bbox_visibility(bbox, image_width, image_height):
    """
    Fraction de la bbox GT theorique qui se trouve dans l'image.

    1.0 = bbox entierement dans l'image
    0.5 = environ la moitie dans l'image
    0.0 = bbox entierement hors image
    """

    x1, y1, x2, y2 = bbox

    full_width = max(0.0, x2 - x1)
    full_height = max(0.0, y2 - y1)

    full_area = full_width * full_height

    if full_area <= 0:
        return 0.0

    clipped_x1 = max(0.0, x1)
    clipped_y1 = max(0.0, y1)

    clipped_x2 = min(float(image_width), x2)
    clipped_y2 = min(float(image_height), y2)

    clipped_width = max(
        0.0,
        clipped_x2 - clipped_x1
    )

    clipped_height = max(
        0.0,
        clipped_y2 - clipped_y1
    )

    visible_area = (
        clipped_width
        *
        clipped_height
    )

    return visible_area / full_area






# ============================================================
# CHEVAUCHEMENT ENTRE BBOX GT
# ============================================================

def compute_max_gt_overlap(gt_index, gt_objects):
    """
    Calcule l'IoU maximale entre le GT courant et les autres
    bbox GT de la meme image.

    Cette mesure sert uniquement au diagnostic :
    - IoU elevee : plusieurs objets GT se chevauchent fortement
      dans l'image ;
    - IoU faible : le faux negatif est probablement explique
      par une autre cause.
    """

    bbox = gt_objects[gt_index].get("bbox_2d")

    if bbox is None:
        return 0.0

    max_iou = 0.0

    for other_index, other in enumerate(gt_objects):

        if other_index == gt_index:
            continue

        other_bbox = other.get("bbox_2d")

        if other_bbox is None:
            continue

        iou = compute_iou(
            bbox,
            other_bbox
        )

        max_iou = max(
            max_iou,
            iou
        )

    return max_iou


# ============================================================
# DIAGNOSTIC GEOMETRIQUE YOLO <-> GT
# ============================================================

def bbox_geometry(bbox):
    """
    Retourne centre, largeur, hauteur et aire d'une bbox xyxy.
    """
    x1, y1, x2, y2 = map(float, bbox)

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)

    area = width * height

    return {
        "cx": center_x,
        "cy": center_y,
        "width": width,
        "height": height,
        "area": area
    }


def compute_bbox_geometry_difference(gt_bbox, yolo_bbox):
    """
    Compare la geometrie de la bbox YOLO a celle de la bbox GT.

    Les erreurs de centre sont normalisees par la largeur/hauteur GT.
    Les ratios > 1 indiquent une bbox YOLO plus grande que le GT.
    """
    gt = bbox_geometry(gt_bbox)
    pred = bbox_geometry(yolo_bbox)

    if gt["width"] <= 0 or gt["height"] <= 0 or gt["area"] <= 0:
        return None

    center_dx_px = pred["cx"] - gt["cx"]
    center_dy_px = pred["cy"] - gt["cy"]

    center_dx_norm = center_dx_px / gt["width"]
    center_dy_norm = center_dy_px / gt["height"]

    center_error_norm = float(
        np.sqrt(
            center_dx_norm ** 2
            +
            center_dy_norm ** 2
        )
    )

    width_ratio = pred["width"] / gt["width"]
    height_ratio = pred["height"] / gt["height"]
    area_ratio = pred["area"] / gt["area"]

    return {
        "center_dx_px": float(center_dx_px),
        "center_dy_px": float(center_dy_px),
        "center_dx_norm": float(center_dx_norm),
        "center_dy_norm": float(center_dy_norm),
        "center_error_norm": center_error_norm,
        "width_ratio": float(width_ratio),
        "height_ratio": float(height_ratio),
        "area_ratio": float(area_ratio)
    }


def get_best_iou_interval(best_iou):
    """
    Intervalles utilises pour le diagnostic du meilleur IoU.
    """
    if best_iou < 0.10:
        return "0-.10"

    if best_iou < 0.25:
        return ".10-.25"

    if best_iou < 0.40:
        return ".25-.40"

    if best_iou < 0.50:
        return ".40-.50"

    if best_iou < 0.75:
        return ".50-.75"

    return ">=.75"


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-step",
        type=int,
        default=20
    )

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
        "--max-save-per-bin",
        type=int,
        default=20
    )

    args = parser.parse_args()

    # ========================================================
    # DOSSIERS SORTIE
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    distance_bins = [
        "0-10",
        "10-20",
        "20-30",
        "30-40",
        "40-50"
    ]

    for bin_name in distance_bins:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                bin_name
            ),
            exist_ok=True
        )



    K, image_width, image_height, calib = (
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

    frame_dirs = frame_dirs[
        ::args.sample_step
    ]

    print("\n========================================")
    print(" ANALYSE DES FAUX NEGATIFS YOLO")
    print("========================================")

    print(
        f"Frames analysees : {len(frame_dirs)}"
    )

    # ========================================================
    # STATISTIQUES
    # ========================================================

    stats = {
        bin_name: {
            "gt": 0,
            "matched": 0,
            "fn": 0,
            "near_miss": 0,
            "no_overlap": 0,
            "saved": 0,

            "fully_visible": 0,
            "partially_truncated": 0,
            "strongly_truncated": 0,

            "matched_fully_visible": 0,

            # Chevauchement avec les autres GT de la meme image
            "overlap_gt_025": 0,
            "overlap_gt_050": 0,

            # Parmi les faux negatifs uniquement
            "fn_overlap_gt_025": 0,
            "fn_overlap_gt_050": 0,

            # Distribution du meilleur IoU pour chaque GT
            "best_iou_hist": {
                "0-.10": 0,
                ".10-.25": 0,
                ".25-.40": 0,
                ".40-.50": 0,
                ".50-.75": 0,
                ">=.75": 0
            },

            # Diagnostic geometrique des near misses
            "near_miss_center_error_norm": [],
            "near_miss_width_ratio": [],
            "near_miss_height_ratio": [],
            "near_miss_area_ratio": []
        }
        for bin_name in distance_bins
    }

    # ========================================================
    # BOUCLE
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

        annotations_path = os.path.join(
            frame_dir,
            "annotations.json"
        )

        image = cv2.imread(
            image_path
        )

        if image is None:
            continue

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

        # ====================================================
        # YOLO
        # ====================================================

        detections = run_yolo(
            model,
            image,
            args.conf
        )

        # ====================================================
        # MATCHING
        # ====================================================

        (
            matches,
            used_detections,
            used_gt
        ) = match_yolo_to_gt(
            detections,
            gt_objects,
            args.iou
        )

        matched_gt = {
            match["gt_index"]
            for match in matches
        }

        # ====================================================
        # ANALYSE GT
        # ====================================================

        for gt_idx, obj in enumerate(
            gt_objects
        ):



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

            if gt_camera_depth <= 0:
                continue

            bin_name = get_distance_bin(
                gt_camera_depth
            )

            if bin_name is None:
                continue


            # ------------------------------------------------
            # Troncature de la bbox
            # ------------------------------------------------

            gt_bbox = obj[
                "bbox_2d"
            ]

            visibility = compute_bbox_visibility(
                gt_bbox,
                image_width,
                image_height
            )

            # ------------------------------------------------
            # Chevauchement avec les autres bbox GT
            # ------------------------------------------------

            max_gt_overlap = compute_max_gt_overlap(
                gt_idx,
                gt_objects
            )

            if max_gt_overlap >= 0.25:

                stats[
                    bin_name
                ]["overlap_gt_025"] += 1

            if max_gt_overlap >= 0.50:

                stats[
                    bin_name
                ]["overlap_gt_050"] += 1

            # ------------------------------------------------
            # Meilleure detection YOLO pour ce GT
            # Utilisee pour la distribution IoU, qu'il soit
            # matche ou non.
            # ------------------------------------------------

            best_iou, best_detection = (
                find_best_detection(
                    gt_bbox,
                    detections
                )
            )

            iou_interval = get_best_iou_interval(
                best_iou
            )

            stats[
                bin_name
            ]["best_iou_hist"][iou_interval] += 1


            if bin_name is None:
                continue

            stats[
                bin_name
            ]["gt"] += 1


            if visibility >= 0.90:

                stats[
                    bin_name
                ]["fully_visible"] += 1

            elif visibility >= 0.50:

                stats[
                    bin_name
                ]["partially_truncated"] += 1

            else:

                stats[
                    bin_name
                ]["strongly_truncated"] += 1


            # ------------------------------------------------
            # Detection correcte
            # ------------------------------------------------

            if gt_idx in matched_gt:

                stats[
                    bin_name
                ]["matched"] += 1

                if visibility >= 0.90:

                    stats[
                        bin_name
                    ]["matched_fully_visible"] += 1

                continue

            # ------------------------------------------------
            # Faux negatif
            # ------------------------------------------------

            stats[
                bin_name
            ]["fn"] += 1

            if max_gt_overlap >= 0.25:

                stats[
                    bin_name
                ]["fn_overlap_gt_025"] += 1

            if max_gt_overlap >= 0.50:

                stats[
                    bin_name
                ]["fn_overlap_gt_050"] += 1

            gt_bbox = obj[
                "bbox_2d"
            ]

            # best_iou et best_detection ont deja ete calcules
            # plus haut pour tous les GT.

            # ------------------------------------------------
            # Classification simple
            # ------------------------------------------------

            if best_iou >= 0.25:

                stats[
                    bin_name
                ]["near_miss"] += 1

                if best_detection is not None:

                    geometry_diff = (
                        compute_bbox_geometry_difference(
                            gt_bbox,
                            best_detection["bbox"]
                        )
                    )

                    if geometry_diff is not None:

                        stats[
                            bin_name
                        ]["near_miss_center_error_norm"].append(
                            geometry_diff["center_error_norm"]
                        )

                        stats[
                            bin_name
                        ]["near_miss_width_ratio"].append(
                            geometry_diff["width_ratio"]
                        )

                        stats[
                            bin_name
                        ]["near_miss_height_ratio"].append(
                            geometry_diff["height_ratio"]
                        )

                        stats[
                            bin_name
                        ]["near_miss_area_ratio"].append(
                            geometry_diff["area_ratio"]
                        )

            elif best_iou == 0:

                stats[
                    bin_name
                ]["no_overlap"] += 1

            # ------------------------------------------------
            # Sauvegarde
            # ------------------------------------------------

            if (
                stats[bin_name]["saved"]
                <
                args.max_save_per_bin
            ):

                visualization = draw_failure(
                    image,
                    gt_bbox,
                    gt_camera_depth,
                    best_iou,
                    best_detection,
                    frame_number,
                    weather
                )

                filename = (
                    f"frame_{frame_number:05d}"
                    f"_gt{gt_idx:02d}"
                    f"_z{gt_camera_depth:.1f}"
                    f"_vis{visibility:.2f}"
                    f"_iou{best_iou:.2f}.jpg"
                )

                output_path = os.path.join(
                    OUTPUT_DIR,
                    bin_name,
                    filename
                )

                cv2.imwrite(
                    output_path,
                    visualization
                )

                stats[
                    bin_name
                ]["saved"] += 1

        # ====================================================
        # PROGRESSION
        # ====================================================

        if (
            frame_idx % 50 == 0
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
    # RESULTATS
    # ========================================================

    print("\n========================================")
    print(" RESULTATS")
    print("========================================")

    print(
        "\nDistance | GT | Match | FN | "
        "Near miss | No overlap"
    )

    print("-" * 65)

    for bin_name in distance_bins:

        s = stats[
            bin_name
        ]

        print(
            f"{bin_name:7s}m | "
            f"{s['gt']:3d} | "
            f"{s['matched']:5d} | "
            f"{s['fn']:3d} | "
            f"{s['near_miss']:9d} | "
            f"{s['no_overlap']:10d}"
        )

    print(
        f"\nImages sauvegardees dans : "
        f"{OUTPUT_DIR}"
    )

    print("\nLegende :")
    print(
        "  Rouge = bbox GT CARLA"
    )
    print(
        "  Vert  = meilleure bbox YOLO"
    )

    print(
        "\nNear miss = meilleure IoU >= 0.25 "
        "mais < seuil de matching"
    )

    print(
        "No overlap = aucune bbox YOLO "
        "ne chevauche le GT"
    )



    print("\n========================================")
    print(" TRONCATURE DES GT")
    print("========================================")

    print(
        "\nDistance | GT | >=90% | "
        "50-90% | <50% | Recall >=90%"
    )

    print("-" * 70)

    for bin_name in distance_bins:

        s = stats[
            bin_name
        ]

        visible = s[
            "fully_visible"
        ]

        visible_matches = s[
            "matched_fully_visible"
        ]

        recall_visible = (
            visible_matches / visible
            if visible > 0
            else 0
        )

        print(
            f"{bin_name:7s}m | "
            f"{s['gt']:3d} | "
            f"{visible:5d} | "
            f"{s['partially_truncated']:6d} | "
            f"{s['strongly_truncated']:4d} | "
            f"{recall_visible * 100:10.1f}%"
        )



    print("\n========================================")
    print(" CHEVAUCHEMENT ENTRE GT")
    print("========================================")

    print(
        "\nDistance | GT | GT IoU>=.25 | "
        "GT IoU>=.50 | FN | FN overlap>=.25 | FN overlap>=.50"
    )

    print("-" * 100)

    for bin_name in distance_bins:

        s = stats[
            bin_name
        ]

        fn = s["fn"]

        fn_overlap_025 = s[
            "fn_overlap_gt_025"
        ]

        fn_overlap_050 = s[
            "fn_overlap_gt_050"
        ]

        ratio_025 = (
            fn_overlap_025 / fn
            if fn > 0
            else 0
        )

        ratio_050 = (
            fn_overlap_050 / fn
            if fn > 0
            else 0
        )

        print(
            f"{bin_name:7s}m | "
            f"{s['gt']:3d} | "
            f"{s['overlap_gt_025']:11d} | "
            f"{s['overlap_gt_050']:11d} | "
            f"{fn:3d} | "
            f"{fn_overlap_025:15d} ({ratio_025 * 100:5.1f}%) | "
            f"{fn_overlap_050:15d} ({ratio_050 * 100:5.1f}%)"
        )


    print("\n========================================")
    print(" DISTRIBUTION DU MEILLEUR IoU")
    print("========================================")

    iou_intervals = [
        "0-.10",
        ".10-.25",
        ".25-.40",
        ".40-.50",
        ".50-.75",
        ">=.75"
    ]

    print(
        "\nDistance | 0-.10 | .10-.25 | .25-.40 | "
        ".40-.50 | .50-.75 | >=.75"
    )

    print("-" * 78)

    for bin_name in distance_bins:

        h = stats[
            bin_name
        ]["best_iou_hist"]

        print(
            f"{bin_name:7s}m | "
            f"{h['0-.10']:5d} | "
            f"{h['.10-.25']:7d} | "
            f"{h['.25-.40']:7d} | "
            f"{h['.40-.50']:7d} | "
            f"{h['.50-.75']:7d} | "
            f"{h['>=.75']:5d}"
        )


    print("\n========================================")
    print(" GEOMETRIE DES NEAR MISSES")
    print("========================================")

    print(
        "\nDistance | N | Center err | Width ratio | "
        "Height ratio | Area ratio"
    )

    print("-" * 82)

    for bin_name in distance_bins:

        s = stats[
            bin_name
        ]

        centers = np.asarray(
            s["near_miss_center_error_norm"],
            dtype=np.float32
        )

        widths = np.asarray(
            s["near_miss_width_ratio"],
            dtype=np.float32
        )

        heights = np.asarray(
            s["near_miss_height_ratio"],
            dtype=np.float32
        )

        areas = np.asarray(
            s["near_miss_area_ratio"],
            dtype=np.float32
        )

        if len(centers) == 0:

            print(
                f"{bin_name:7s}m | "
                f"{0:3d} | "
                f"{'N/A':>10s} | "
                f"{'N/A':>11s} | "
                f"{'N/A':>12s} | "
                f"{'N/A':>10s}"
            )

            continue

        print(
            f"{bin_name:7s}m | "
            f"{len(centers):3d} | "
            f"{np.median(centers):10.3f} | "
            f"{np.median(widths):11.3f} | "
            f"{np.median(heights):12.3f} | "
            f"{np.median(areas):10.3f}"
        )

    print(
        "\nInterpretation geometrie :"
    )

    print(
        "  Center err = distance entre centres YOLO/GT, "
        "normalisee par la taille GT."
    )

    print(
        "  Width/Height/Area ratio = taille YOLO / taille GT."
    )

    print(
        "  Ratio = 1.0 : tailles identiques ; "
        "< 1.0 : YOLO plus petit ; > 1.0 : YOLO plus grand."
    )


    print("\n========================================")
    print(" ANALYSE TERMINEE")
    print("========================================")


if __name__ == "__main__":
    main()
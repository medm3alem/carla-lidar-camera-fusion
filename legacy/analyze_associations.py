import os
import json
import glob
import csv

import numpy as np

from visualize_fusion import (
    load_calibration,
    project_lidar_to_camera,
    associate_lidar_to_bboxes,
    world_to_camera_depth
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset_full_v2"
OUTPUT_CSV = "analysis_results.csv"

TOP_N = 20


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    K, width, height, calib = load_calibration()

    # --------------------------------------------------------
    # Frames
    # --------------------------------------------------------

    frame_dirs = sorted(
        glob.glob(
            os.path.join(
                DATASET_DIR,
                "frame_*"
            )
        )
    )

    print("\n========================================")
    print(" ANALYSE DETAILLEE DES ASSOCIATIONS")
    print("========================================")

    print(f"Frames trouvees : {len(frame_dirs)}")

    results_csv = []

    # ========================================================
    # PARCOURS DATASET
    # ========================================================

    for frame_idx, frame_dir in enumerate(frame_dirs):

        frame_name = os.path.basename(frame_dir)

        try:
            frame_number = int(
                frame_name.split("_")[1]
            )
        except (IndexError, ValueError):
            continue

        annotations_path = os.path.join(
            frame_dir,
            "annotations.json"
        )

        lidar_path = os.path.join(
            frame_dir,
            "lidar.npy"
        )

        if not os.path.isfile(annotations_path):
            continue

        if not os.path.isfile(lidar_path):
            continue

        # ----------------------------------------------------
        # Chargement annotations
        # ----------------------------------------------------

        with open(annotations_path, "r") as f:
            annotations = json.load(f)

        weather = annotations.get(
            "weather",
            "unknown"
        )

        # ----------------------------------------------------
        # Chargement LiDAR
        # ----------------------------------------------------

        points = np.load(lidar_path)

        # ----------------------------------------------------
        # Projection LiDAR -> camera
        # ----------------------------------------------------

        u, v, depth, intensity = (
            project_lidar_to_camera(
                points,
                K,
                width,
                height,
                calib
            )
        )

        # ----------------------------------------------------
        # Association
        # ----------------------------------------------------

        associations = associate_lidar_to_bboxes(
            annotations,
            u,
            v,
            depth
        )

        # ====================================================
        # OBJETS
        # ====================================================

        for object_index, result in enumerate(associations):

            obj = result["object"]

            gt_distance = obj.get("distance")

            if gt_distance is None:
                continue

            # ------------------------------------------------
            # GT dans le repere camera
            # ------------------------------------------------

            gt_camera_depth, gt_camera_distance = (
                world_to_camera_depth(
                    obj,
                    annotations,
                    calib
                )
            )

            if gt_camera_depth <= 0:
                continue

            lidar_depth = result[
                "lidar_median_depth"
            ]

            n_bbox_points = result[
                "n_lidar_points"
            ]

            n_cluster_points = result[
                "n_cluster_points"
            ]

            # ------------------------------------------------
            # Pas d'association
            # ------------------------------------------------

            if lidar_depth is None:

                results_csv.append({
                    "frame": frame_number,
                    "weather": weather,
                    "object_index": object_index,
                    "type": obj.get("type", "unknown"),
                    "gt_distance": gt_distance,
                    "gt_camera_depth": gt_camera_depth,
                    "lidar_depth": "",
                    "error": "",
                    "abs_error": "",
                    "n_bbox_points": n_bbox_points,
                    "n_cluster_points": n_cluster_points,
                    "associated": 0
                })

                continue

            # ------------------------------------------------
            # Erreur
            # ------------------------------------------------

            error = (
                lidar_depth
                -
                gt_camera_depth
            )

            abs_error = abs(error)

            results_csv.append({
                "frame": frame_number,
                "weather": weather,
                "object_index": object_index,
                "type": obj.get("type", "unknown"),
                "gt_distance": gt_distance,
                "gt_camera_depth": gt_camera_depth,
                "lidar_depth": lidar_depth,
                "error": error,
                "abs_error": abs_error,
                "n_bbox_points": n_bbox_points,
                "n_cluster_points": n_cluster_points,
                "associated": 1
            })

        # ----------------------------------------------------
        # Progression
        # ----------------------------------------------------

        if (
            frame_idx % 500 == 0
            or
            frame_idx == len(frame_dirs) - 1
        ):

            print(
                f"Analyse : "
                f"{frame_idx + 1}/"
                f"{len(frame_dirs)}"
            )

    # ========================================================
    # SAUVEGARDE CSV
    # ========================================================

    fieldnames = [
        "frame",
        "weather",
        "object_index",
        "type",
        "gt_distance",
        "gt_camera_depth",
        "lidar_depth",
        "error",
        "abs_error",
        "n_bbox_points",
        "n_cluster_points",
        "associated"
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
        writer.writerows(results_csv)

    print(
        f"\nCSV sauvegarde : {OUTPUT_CSV}"
    )

    # ========================================================
    # ASSOCIATIONS VALIDES
    # ========================================================

    valid_results = [
        r
        for r in results_csv
        if r["associated"] == 1
    ]

    if len(valid_results) == 0:

        print("Aucune association valide.")
        return

    errors = np.array(
        [
            r["error"]
            for r in valid_results
        ],
        dtype=np.float32
    )

    abs_errors = np.abs(errors)

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    print("\n========================================")
    print(" DISTRIBUTION DES ERREURS")
    print("========================================")

    print(
        f"Associations : {len(valid_results)}"
    )

    print(
        f"MAE          : "
        f"{np.mean(abs_errors):.2f} m"
    )

    print(
        f"RMSE         : "
        f"{np.sqrt(np.mean(errors ** 2)):.2f} m"
    )

    print(
        f"Biais        : "
        f"{np.mean(errors):+.2f} m"
    )

    print(
        f"Median error : "
        f"{np.median(errors):+.2f} m"
    )

    print("\nPercentiles erreur absolue :")

    for percentile in [
        50,
        75,
        90,
        95,
        99
    ]:

        value = np.percentile(
            abs_errors,
            percentile
        )

        print(
            f"P{percentile:02d} : "
            f"{value:.2f} m"
        )

    # ========================================================
    # SEUILS D'ERREUR
    # ========================================================

    print("\n========================================")
    print(" ASSOCIATIONS SELON L'ERREUR")
    print("========================================")

    for threshold in [
        1,
        2,
        3,
        5,
        10
    ]:

        count = np.sum(
            abs_errors <= threshold
        )

        percentage = (
            100.0
            *
            count
            /
            len(abs_errors)
        )

        print(
            f"|erreur| <= {threshold:2d} m : "
            f"{count:5d} "
            f"({percentage:5.1f}%)"
        )

    # ========================================================
    # PIRES ASSOCIATIONS
    # ========================================================

    worst_results = sorted(
        valid_results,
        key=lambda r: r["abs_error"],
        reverse=True
    )

    print("\n========================================")
    print(f" TOP {TOP_N} PIRES ASSOCIATIONS")
    print("========================================")

    for rank, result in enumerate(
        worst_results[:TOP_N],
        start=1
    ):

        print(
            f"{rank:02d}. "
            f"frame={result['frame']:05d} | "
            f"{result['weather']:5s} | "
            f"obj={result['object_index']:02d} | "
            f"GT_Z={result['gt_camera_depth']:5.1f} m | "
            f"LiDAR={result['lidar_depth']:5.1f} m | "
            f"error={result['error']:+6.1f} m | "
            f"bbox_pts={result['n_bbox_points']:3d} | "
            f"cluster_pts={result['n_cluster_points']:3d}"
        )

    # ========================================================
    # QUELQUES STATISTIQUES SUR LES OUTLIERS
    # ========================================================

    print("\n========================================")
    print(" OUTLIERS")
    print("========================================")

    for threshold in [
        5,
        10,
        15,
        20
    ]:

        mask = (
            abs_errors > threshold
        )

        count = np.sum(mask)

        percentage = (
            100.0
            *
            count
            /
            len(abs_errors)
        )

        print(
            f"|erreur| > {threshold:2d} m : "
            f"{count:5d} "
            f"({percentage:5.2f}%)"
        )

    print("\n========================================")
    print(" ANALYSE TERMINEE")
    print("========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
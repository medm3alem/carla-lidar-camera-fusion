import os
import json
import glob

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

DISTANCE_BINS = [
    (0, 10),
    (10, 20),
    (20, 30),
    (30, 40),
    (40, 50)
]


# ============================================================
# OUTILS
# ============================================================

def get_distance_bin(distance):

    for dmin, dmax in DISTANCE_BINS:

        if dmin <= distance < dmax:
            return f"{dmin}-{dmax}"

    return None


def compute_error_metrics(errors):
    """
    Calcule les metriques d'erreur de profondeur.

    error = LiDAR depth - GT center depth
    """

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
            np.sqrt(np.mean(errors ** 2))
        ),
        "bias": float(
            np.mean(errors)
        ),
        "median": float(
            np.median(errors)
        ),
        "std": float(
            np.std(errors)
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CALIBRATION
    # ========================================================

    K, width, height, calib = load_calibration()

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

    print("\n========================================")
    print(" ANALYSE DU DATASET")
    print("========================================")

    print(
        f"Frames trouvees : {len(frame_dirs)}"
    )

    # ========================================================
    # STATISTIQUES GENERALES
    # ========================================================

    total_objects = 0
    associated_objects = 0

    all_points = []

    # Ancienne comparaison :
    # distance ego -> centre
    # moins profondeur LiDAR.
    #
    # On la conserve uniquement pour comparaison.
    all_old_deltas = []

    # Nouvelle vraie comparaison :
    # LiDAR depth - GT center camera depth
    all_depth_errors = []

    # ========================================================
    # STATISTIQUES METEO
    # ========================================================

    weather_stats = {}

    # ========================================================
    # STATISTIQUES DISTANCE
    # ========================================================

    distance_stats = {
        f"{a}-{b}": {
            "total": 0,
            "associated": 0,
            "points": [],
            "old_delta": [],
            "depth_error": []
        }
        for a, b in DISTANCE_BINS
    }

    # ========================================================
    # PARCOURS DU DATASET
    # ========================================================

    for frame_number, frame_dir in enumerate(
        frame_dirs
    ):

        annotations_path = os.path.join(
            frame_dir,
            "annotations.json"
        )

        lidar_path = os.path.join(
            frame_dir,
            "lidar.npy"
        )

        # ----------------------------------------------------
        # Verification fichiers
        # ----------------------------------------------------

        if (
            not os.path.isfile(annotations_path)
            or
            not os.path.isfile(lidar_path)
        ):
            continue

        # ----------------------------------------------------
        # Annotations
        # ----------------------------------------------------

        with open(
            annotations_path,
            "r"
        ) as f:

            annotations = json.load(f)

        weather = annotations.get(
            "weather",
            "unknown"
        )

        # ----------------------------------------------------
        # Initialisation stats meteo
        # ----------------------------------------------------

        if weather not in weather_stats:

            weather_stats[weather] = {
                "frames": 0,
                "objects": 0,
                "associated": 0,
                "points": [],
                "old_delta": [],
                "depth_error": []
            }

        weather_stats[
            weather
        ]["frames"] += 1

        # ----------------------------------------------------
        # LiDAR
        # ----------------------------------------------------

        points = np.load(
            lidar_path
        )

        # ----------------------------------------------------
        # Projection LiDAR -> Camera
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
        # Association LiDAR -> bbox
        # ----------------------------------------------------

        associations = (
            associate_lidar_to_bboxes(
                annotations,
                u,
                v,
                depth
            )
        )

        # ====================================================
        # ANALYSE DES OBJETS
        # ====================================================

        for result in associations:

            obj = result[
                "object"
            ]

            # ------------------------------------------------
            # Ancienne distance GT
            # ------------------------------------------------

            gt_distance = obj.get(
                "distance"
            )

            if gt_distance is None:
                continue

            # ------------------------------------------------
            # Nouveau GT :
            # profondeur du centre objet dans le repere camera
            # ------------------------------------------------

            gt_camera_depth, gt_camera_distance = (
                world_to_camera_depth(
                    obj,
                    annotations,
                    calib
                )
            )

            # On ignore les objets qui seraient derriere
            # la camera, par securite.
            if gt_camera_depth <= 0:
                continue

            # ------------------------------------------------
            # Compteurs
            # ------------------------------------------------

            total_objects += 1

            weather_stats[
                weather
            ]["objects"] += 1

            n_points = result[
                "n_lidar_points"
            ]

            all_points.append(
                n_points
            )

            weather_stats[
                weather
            ]["points"].append(
                n_points
            )

            # ------------------------------------------------
            # Classe de distance
            #
            # On garde ici la distance GT originale pour
            # conserver les memes bins que l'analyse precedente.
            # ------------------------------------------------

            bin_name = get_distance_bin(
                gt_distance
            )

            if bin_name is not None:

                distance_stats[
                    bin_name
                ]["total"] += 1

                distance_stats[
                    bin_name
                ]["points"].append(
                    n_points
                )

            # ------------------------------------------------
            # Estimation LiDAR disponible ?
            # ------------------------------------------------

            lidar_depth = result[
                "lidar_median_depth"
            ]

            if lidar_depth is None:
                continue

            # =================================================
            # ASSOCIATION VALIDE
            # =================================================

            associated_objects += 1

            weather_stats[
                weather
            ]["associated"] += 1

            if bin_name is not None:

                distance_stats[
                    bin_name
                ]["associated"] += 1

            # ------------------------------------------------
            # Ancien delta
            #
            # Ce n'est PAS une vraie erreur de profondeur.
            # ------------------------------------------------

            old_delta = (
                gt_distance
                -
                lidar_depth
            )

            all_old_deltas.append(
                old_delta
            )

            weather_stats[
                weather
            ]["old_delta"].append(
                old_delta
            )

            if bin_name is not None:

                distance_stats[
                    bin_name
                ]["old_delta"].append(
                    old_delta
                )

            # ------------------------------------------------
            # NOUVELLE ERREUR DE PROFONDEUR
            #
            # e = profondeur LiDAR
            #     - profondeur centre GT camera
            #
            # Un biais negatif est attendu car le LiDAR
            # touche generalement la surface visible du
            # vehicule avant son centre.
            # ------------------------------------------------

            depth_error = (
                lidar_depth
                -
                gt_camera_depth
            )

            all_depth_errors.append(
                depth_error
            )

            weather_stats[
                weather
            ]["depth_error"].append(
                depth_error
            )

            if bin_name is not None:

                distance_stats[
                    bin_name
                ]["depth_error"].append(
                    depth_error
                )

        # ====================================================
        # PROGRESSION
        # ====================================================

        if (
            frame_number % 500 == 0
            or
            frame_number
            ==
            len(frame_dirs) - 1
        ):

            print(
                f"Analyse : "
                f"{frame_number + 1}/"
                f"{len(frame_dirs)}"
            )

    # ========================================================
    # RESULTATS GENERAUX
    # ========================================================

    print("\n========================================")
    print(" RESULTATS GENERAUX")
    print("========================================")

    print(
        f"Frames          : {len(frame_dirs)}"
    )

    print(
        f"Objets          : {total_objects}"
    )

    print(
        f"Associes LiDAR  : {associated_objects}"
    )

    if total_objects > 0:

        association_rate = (
            100.0
            *
            associated_objects
            /
            total_objects
        )

        print(
            f"Taux association: "
            f"{association_rate:.1f}%"
        )

    # --------------------------------------------------------
    # Points LiDAR
    # --------------------------------------------------------

    if all_points:

        print(
            f"Points/bbox moy.: "
            f"{np.mean(all_points):.1f}"
        )

        print(
            f"Points/bbox med.: "
            f"{np.median(all_points):.1f}"
        )

    # ========================================================
    # ANCIEN DELTA
    # ========================================================

    if all_old_deltas:

        print(
            "\n--- Ancienne comparaison "
            "(reference uniquement) ---"
        )

        print(
            f"Delta moyen     : "
            f"{np.mean(all_old_deltas):.2f} m"
        )

        print(
            f"Delta median    : "
            f"{np.median(all_old_deltas):.2f} m"
        )

    # ========================================================
    # NOUVELLE EVALUATION DE PROFONDEUR
    # ========================================================

    metrics = compute_error_metrics(
        all_depth_errors
    )

    if metrics is not None:

        print(
            "\n--- Evaluation profondeur camera ---"
        )

        print(
            "Erreur = "
            "LiDAR depth - GT center Z"
        )

        print(
            f"MAE             : "
            f"{metrics['mae']:.2f} m"
        )

        print(
            f"RMSE            : "
            f"{metrics['rmse']:.2f} m"
        )

        print(
            f"Biais moyen     : "
            f"{metrics['bias']:+.2f} m"
        )

        print(
            f"Erreur mediane  : "
            f"{metrics['median']:+.2f} m"
        )

        print(
            f"Ecart-type      : "
            f"{metrics['std']:.2f} m"
        )

    # ========================================================
    # PAR METEO
    # ========================================================

    print("\n========================================")
    print(" PAR METEO")
    print("========================================")

    for weather, stats in weather_stats.items():

        total = stats[
            "objects"
        ]

        associated = stats[
            "associated"
        ]

        rate = (
            100.0
            *
            associated
            /
            total
            if total > 0
            else 0.0
        )

        mean_points = (
            np.mean(
                stats["points"]
            )
            if stats["points"]
            else 0.0
        )

        metrics = compute_error_metrics(
            stats["depth_error"]
        )

        print(
            f"\n{weather.upper()}"
        )

        print(
            f"  Frames       : "
            f"{stats['frames']}"
        )

        print(
            f"  Objets       : "
            f"{total}"
        )

        print(
            f"  Associes     : "
            f"{associated}"
        )

        print(
            f"  Taux         : "
            f"{rate:.1f}%"
        )

        print(
            f"  Points moyens: "
            f"{mean_points:.1f}"
        )

        if metrics is not None:

            print(
                f"  MAE depth    : "
                f"{metrics['mae']:.2f} m"
            )

            print(
                f"  RMSE depth   : "
                f"{metrics['rmse']:.2f} m"
            )

            print(
                f"  Biais depth  : "
                f"{metrics['bias']:+.2f} m"
            )

            print(
                f"  Err. mediane : "
                f"{metrics['median']:+.2f} m"
            )

    # ========================================================
    # PAR DISTANCE
    # ========================================================

    print("\n========================================")
    print(" PAR DISTANCE")
    print("========================================")

    print(
        "\n"
        "Distance | Objets | Assoc. | "
        "Pts moy. | MAE | RMSE | Biais"
    )

    print(
        "-" * 72
    )

    for bin_name, stats in distance_stats.items():

        total = stats[
            "total"
        ]

        associated = stats[
            "associated"
        ]

        rate = (
            100.0
            *
            associated
            /
            total
            if total > 0
            else 0.0
        )

        mean_points = (
            np.mean(
                stats["points"]
            )
            if stats["points"]
            else 0.0
        )

        metrics = compute_error_metrics(
            stats["depth_error"]
        )

        if metrics is not None:

            print(
                f"{bin_name:7s}m | "
                f"{total:6d} | "
                f"{rate:5.1f}% | "
                f"{mean_points:7.1f} | "
                f"{metrics['mae']:4.2f} | "
                f"{metrics['rmse']:5.2f} | "
                f"{metrics['bias']:+5.2f}"
            )

        else:

            print(
                f"{bin_name:7s}m | "
                f"{total:6d} | "
                f"{rate:5.1f}% | "
                f"{mean_points:7.1f} | "
                f" N/A | "
                f" N/A | "
                f" N/A"
            )

    # ========================================================
    # FIN
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        " Analyse terminee."
    )

    print(
        "========================================\n"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
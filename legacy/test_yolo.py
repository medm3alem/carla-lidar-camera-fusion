import os
import argparse

import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset_full_v2"

MODEL_NAME = "yolo26n.pt"

CONF_THRESHOLD = 0.25

# Classes COCO qui nous intéressent pour l'automobile
VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle"
}


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    frame_dir = os.path.join(
        DATASET_DIR,
        f"frame_{args.frame:05d}"
    )

    image_path = os.path.join(
        frame_dir,
        "rgb.png"
    )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image introuvable : {image_path}"
        )

    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(
            f"Impossible de lire {image_path}"
        )

    # --------------------------------------------------------
    # Modele YOLO
    # --------------------------------------------------------

    print("\nChargement YOLO...")

    model = YOLO(MODEL_NAME)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print(
        f"Inference frame {args.frame:05d}..."
    )

    results = model.predict(
        source=image,
        conf=args.conf,
        verbose=False
    )

    result = results[0]

    print("\n========================================")
    print(" DETECTIONS YOLO")
    print("========================================")

    vehicle_count = 0

    # --------------------------------------------------------
    # Detections
    # --------------------------------------------------------

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

        # On ne garde que les vehicules
        if class_name not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        print(
            f"{vehicle_count:02d} | "
            f"{class_name:10s} | "
            f"conf={confidence:.3f} | "
            f"bbox="
            f"[{x1}, {y1}, {x2}, {y2}]"
        )

        # ----------------------------------------------------
        # Dessin bbox
        # ----------------------------------------------------

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        vehicle_count += 1

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    print("----------------------------------------")

    print(
        f"Vehicules detectes : {vehicle_count}"
    )

    print("========================================")

    # --------------------------------------------------------
    # Affichage
    # --------------------------------------------------------

    cv2.imshow(
        "YOLO - CARLA",
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
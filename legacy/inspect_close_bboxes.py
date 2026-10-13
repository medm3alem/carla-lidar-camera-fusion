import os
import json
import glob
import argparse

import cv2
import numpy as np
from ultralytics import YOLO

from evaluate_yolo_fusion import compute_iou
from visualize_fusion import load_calibration, world_to_camera_depth


DATASET_DIR = "dataset_full_v2"
MODEL_NAME = "yolo26n.pt"
OUTPUT_DIR = "close_bbox_inspection"

VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle"
}


def run_yolo(model, image, confidence):
    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )[0]

    detections = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence_value = float(box.conf[0])
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
            "confidence": confidence_value
        })

    return detections


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


def bbox_geometry(bbox):
    x1, y1, x2, y2 = map(float, bbox)

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return {
        "cx": 0.5 * (x1 + x2),
        "cy": 0.5 * (y1 + y2),
        "width": width,
        "height": height,
        "area": width * height
    }


def draw_label(image, text, origin, color):
    x, y = origin

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        1
    )

    cv2.rectangle(
        image,
        (x, y - text_height - baseline - 5),
        (x + text_width + 6, y + 3),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        image,
        text,
        (x + 3, y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA
    )


def draw_inspection(
    image,
    gt_bbox,
    best_detection,
    detections,
    gt_depth,
    best_iou,
    frame_number,
    gt_index,
    weather
):
    output = image.copy()

    # --------------------------------------------------------
    # Toutes les detections YOLO : cyan fin
    # --------------------------------------------------------
    for detection_index, detection in enumerate(detections):
        x1, y1, x2, y2 = map(
            int,
            detection["bbox"]
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            1
        )

        draw_label(
            output,
            (
                f"Y{detection_index} "
                f"{detection['class_name']} "
                f"{detection['confidence']:.2f}"
            ),
            (x1, max(18, y1)),
            (255, 255, 0)
        )

    # --------------------------------------------------------
    # GT courant : rouge
    # --------------------------------------------------------
    gx1, gy1, gx2, gy2 = map(
        int,
        gt_bbox
    )

    cv2.rectangle(
        output,
        (gx1, gy1),
        (gx2, gy2),
        (0, 0, 255),
        3
    )

    gt_geom = bbox_geometry(
        gt_bbox
    )

    gt_center = (
        int(round(gt_geom["cx"])),
        int(round(gt_geom["cy"]))
    )

    cv2.drawMarker(
        output,
        gt_center,
        (0, 0, 255),
        cv2.MARKER_CROSS,
        18,
        2
    )

    draw_label(
        output,
        (
            f"GT{gt_index} Z={gt_depth:.2f}m "
            f"W={gt_geom['width']:.0f} "
            f"H={gt_geom['height']:.0f}"
        ),
        (gx1, max(38, gy1)),
        (0, 0, 255)
    )

    # --------------------------------------------------------
    # Meilleure detection : vert epais
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
            3
        )

        pred_geom = bbox_geometry(
            best_detection["bbox"]
        )

        pred_center = (
            int(round(pred_geom["cx"])),
            int(round(pred_geom["cy"]))
        )

        cv2.drawMarker(
            output,
            pred_center,
            (0, 255, 0),
            cv2.MARKER_CROSS,
            18,
            2
        )

        cv2.line(
            output,
            gt_center,
            pred_center,
            (255, 255, 255),
            1
        )

        width_ratio = (
            pred_geom["width"] / gt_geom["width"]
            if gt_geom["width"] > 0
            else 0.0
        )

        height_ratio = (
            pred_geom["height"] / gt_geom["height"]
            if gt_geom["height"] > 0
            else 0.0
        )

        area_ratio = (
            pred_geom["area"] / gt_geom["area"]
            if gt_geom["area"] > 0
            else 0.0
        )

        center_dx_norm = (
            (pred_geom["cx"] - gt_geom["cx"])
            / gt_geom["width"]
            if gt_geom["width"] > 0
            else 0.0
        )

        center_dy_norm = (
            (pred_geom["cy"] - gt_geom["cy"])
            / gt_geom["height"]
            if gt_geom["height"] > 0
            else 0.0
        )

        center_error_norm = float(
            np.sqrt(
                center_dx_norm ** 2
                +
                center_dy_norm ** 2
            )
        )

        info_lines = [
            (
                f"Frame {frame_number:05d} | {weather} | "
                f"GT {gt_index} | Z={gt_depth:.2f} m"
            ),
            (
                f"Best IoU={best_iou:.3f} | "
                f"YOLO conf={best_detection['confidence']:.3f}"
            ),
            (
                f"GT   : W={gt_geom['width']:.1f} "
                f"H={gt_geom['height']:.1f} "
                f"A={gt_geom['area']:.0f}"
            ),
            (
                f"YOLO : W={pred_geom['width']:.1f} "
                f"H={pred_geom['height']:.1f} "
                f"A={pred_geom['area']:.0f}"
            ),
            (
                f"Ratios YOLO/GT : "
                f"W={width_ratio:.2f} "
                f"H={height_ratio:.2f} "
                f"A={area_ratio:.2f}"
            ),
            (
                f"Center error norm={center_error_norm:.3f} "
                f"(dx={center_dx_norm:.3f}, dy={center_dy_norm:.3f})"
            )
        ]

    else:
        info_lines = [
            (
                f"Frame {frame_number:05d} | {weather} | "
                f"GT {gt_index} | Z={gt_depth:.2f} m"
            ),
            "Aucune detection YOLO avec chevauchement."
        ]

    # Bandeau noir pour rendre les mesures lisibles
    panel_height = 24 + 23 * len(info_lines)

    overlay = output.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (output.shape[1], panel_height),
        (0, 0, 0),
        -1
    )

    output = cv2.addWeighted(
        overlay,
        0.70,
        output,
        0.30,
        0
    )

    for line_index, line in enumerate(info_lines):
        cv2.putText(
            output,
            line,
            (12, 24 + line_index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-step",
        type=int,
        default=20,
        help="Analyse une frame sur N."
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Seuil de confiance YOLO."
    )

    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0
    )

    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--max-iou",
        type=float,
        default=0.40
    )

    parser.add_argument(
        "--max-save",
        type=int,
        default=30
    )

    args = parser.parse_args()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    _, _, _, calib = load_calibration()

    print("\nChargement YOLO...")
    model = YOLO(MODEL_NAME)

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
    print(" INSPECTION BBOX PROCHES")
    print("========================================")
    print(
        f"Frames parcourues : {len(frame_dirs)}"
    )
    print(
        f"Profondeur : "
        f"[{args.min_depth}, {args.max_depth}) m"
    )
    print(
        f"Best IoU : "
        f"[{args.min_iou}, {args.max_iou})"
    )
    print(
        f"Maximum d'images : {args.max_save}"
    )

    saved = 0
    candidates = 0

    for frame_idx, frame_dir in enumerate(frame_dirs):
        if saved >= args.max_save:
            break

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
            and obj.get("type") == "vehicle"
        ]

        detections = run_yolo(
            model,
            image,
            args.conf
        )

        for gt_index, obj in enumerate(gt_objects):
            if saved >= args.max_save:
                break

            gt_depth, _ = world_to_camera_depth(
                obj,
                annotations,
                calib
            )

            if not (
                args.min_depth
                <= gt_depth
                < args.max_depth
            ):
                continue

            gt_bbox = obj["bbox_2d"]

            best_iou, best_detection = (
                find_best_detection(
                    gt_bbox,
                    detections
                )
            )

            if not (
                args.min_iou
                <= best_iou
                < args.max_iou
            ):
                continue

            candidates += 1

            visualization = draw_inspection(
                image=image,
                gt_bbox=gt_bbox,
                best_detection=best_detection,
                detections=detections,
                gt_depth=gt_depth,
                best_iou=best_iou,
                frame_number=frame_number,
                gt_index=gt_index,
                weather=weather
            )

            filename = (
                f"frame_{frame_number:05d}"
                f"_gt{gt_index:02d}"
                f"_z{gt_depth:.2f}"
                f"_iou{best_iou:.3f}.jpg"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                filename
            )

            cv2.imwrite(
                output_path,
                visualization
            )

            print(
                f"[{saved + 1:02d}/{args.max_save}] "
                f"frame={frame_number:05d} "
                f"GT={gt_index:02d} "
                f"Z={gt_depth:.2f}m "
                f"IoU={best_iou:.3f}"
            )

            saved += 1

        if (
            frame_idx % 50 == 0
            and saved < args.max_save
        ):
            print(
                f"Parcours : "
                f"{frame_idx + 1}/"
                f"{len(frame_dirs)} | "
                f"{saved} images sauvegardees"
            )

    print("\n========================================")
    print(" INSPECTION TERMINEE")
    print("========================================")
    print(
        f"Candidats trouves : {candidates}"
    )
    print(
        f"Images sauvegardees : {saved}"
    )
    print(
        f"Dossier : {OUTPUT_DIR}"
    )
    print(
        "\nLegende :"
    )
    print(
        "  Rouge      = bbox GT CARLA examinee"
    )
    print(
        "  Vert epais = meilleure bbox YOLO"
    )
    print(
        "  Cyan fin   = toutes les detections YOLO"
    )
    print(
        "  Croix      = centre de chaque bbox GT/YOLO"
    )


if __name__ == "__main__":
    main()

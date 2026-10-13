"""
Projette les points LiDAR dans l'image RGB pour verifier la calibration.
Genere une image overlay coloree par distance.
"""
import numpy as np
import cv2
import json
import os
import sys

DATASET_DIR = "dataset"


def load_calib(dataset_dir):
    with open(os.path.join(dataset_dir, "calib.json")) as f:
        calib = json.load(f)
    K = np.array(calib["K"])
    lidar_to_camera = np.array(calib["lidar_to_camera"])
    return K, lidar_to_camera, calib["img_width"], calib["img_height"]


def project_lidar_to_image(points, K, lidar_to_camera, img_w, img_h):
    """
    points: (N, 4) x, y, z, intensity dans le repere LiDAR CARLA
    Retourne les points 2D valides + leur profondeur (pour la couleur)
    """
    xyz = points[:, :3]
    n = xyz.shape[0]

    # Repere LiDAR CARLA : x-avant, y-droite, z-haut (convention UE4 gauche)
    # Repere camera CARLA (avant transformation standard) : meme convention
    # On passe en coordonnees homogenes
    xyz_hom = np.hstack([xyz, np.ones((n, 1))])  # (N, 4)

    # Application de la transformation extrinseque lidar -> camera
    points_cam = (lidar_to_camera @ xyz_hom.T).T  # (N, 4)

    # Repere CARLA (x-avant, y-droite, z-haut) -> repere camera standard (x-droite, y-bas, z-avant)
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    points_std = np.stack([y, -z, x], axis=1)  # (N, 3) repere "camera standard"

    # On ne garde que les points devant la camera
    valid = points_std[:, 2] > 0.1
    points_std = points_std[valid]
    depths = points_std[:, 2].copy()

    # Projection via K
    points_2d_hom = (K @ points_std.T).T
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]

    # Filtrer ce qui tombe dans l'image
    in_image = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_w) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_h)
    )
    return points_2d[in_image], depths[in_image]


def make_overlay(img, points_2d, depths, max_depth=50.0):
    overlay = img.copy()
    depths_clipped = np.clip(depths, 0, max_depth)
    norm = (depths_clipped / max_depth * 255).astype(np.uint8)
    colors = cv2.applyColorMap(norm.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)

    for (x, y), color in zip(points_2d, colors):
        cv2.circle(overlay, (int(x), int(y)), 2, color.tolist(), -1)
    return overlay


def main():
    K, lidar_to_camera, img_w, img_h = load_calib(DATASET_DIR)

    frame_id = sys.argv[1] if len(sys.argv) > 1 else "0010"
    frame_dir = os.path.join(DATASET_DIR, f"frame_{frame_id}")

    img = cv2.imread(os.path.join(frame_dir, "rgb.png"))
    points = np.load(os.path.join(frame_dir, "lidar.npy"))

    points_2d, depths = project_lidar_to_image(points, K, lidar_to_camera, img_w, img_h)
    print(f"Frame {frame_id} : {points.shape[0]} points LiDAR total, "
          f"{points_2d.shape[0]} projetes dans l'image")

    overlay = make_overlay(img, points_2d, depths)

    out_path = os.path.join("results", f"overlay_{frame_id}.png")
    os.makedirs("results", exist_ok=True)
    cv2.imwrite(out_path, overlay)
    print(f"Overlay sauvegarde : {out_path}")

    cv2.imshow("LiDAR-Camera overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
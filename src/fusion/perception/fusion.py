"""
Fusion tardive caméra + LiDAR. Fonctions extraites à l'identique de scripts/09_fusion_depth.py
(aucun changement de comportement). N'importe pas carla : testable hors simulation.
"""
import numpy as np

from .. import geometry as geo

GROUND_MARGIN = 0.30     # m au-dessus du sol (repère LiDAR : le capteur est à lidar_height du sol)
INNER = 0.6              # fraction centrale de la boîte où l'on cherche les points
MONO_HEIGHT = 1.5        # m, hauteur de véhicule supposée pour la baseline caméra seule


def box_iou(a, b):
    """IoU (N,M) et IoA = intersection / aire de a (N,M)."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)
    ioa = inter / np.maximum(area_a[:, None], 1e-9)
    return iou, ioa


def boxes_of(objs):
    """Boîtes [x0,y0,x1,y1] (bord exclusif) des objets d'un labels.json."""
    return np.array([[o["box2d"][0], o["box2d"][1], o["box2d"][2] + 1, o["box2d"][3] + 1]
                     for o in objs], float).reshape(-1, 4)


def project_lidar(pts, T_world_lidar, T_world_cam, K, width, height, lidar_height,
                  ground_margin=GROUND_MARGIN):
    """Points LiDAR (N,4+, repère LiDAR) -> (uv, depth) dans l'image, sol retiré."""
    pc = geo.transform_points(geo.lidar_to_camera_matrix(T_world_lidar, T_world_cam), pts[:, :3])
    uv, depth, src = geo.project(pc, K, width, height)
    keep = pts[src, 2] > -lidar_height + ground_margin
    return uv[keep], depth[keep]


def lidar_in_box(uv, depth, box, inner=INNER):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) / 2 * inner, (box[3] - box[1]) / 2 * inner
    m = (np.abs(uv[:, 0] - cx) <= hw) & (np.abs(uv[:, 1] - cy) <= hh)
    return depth[m]


def front_cluster(d, gap=1.0, min_pts=3):
    """Premier groupe (le plus proche) d'au moins min_pts profondeurs, séparé du suivant par > gap m.
    Retourne (profondeur médiane ou None, nombre de points)."""
    if len(d) < min_pts:
        return None, 0
    d = np.sort(d)
    for grp in np.split(d, np.flatnonzero(np.diff(d) > gap) + 1):
        if len(grp) >= min_pts:
            return float(np.median(grp)), len(grp)
    return None, 0


def fuse_box(uv, depth, box):
    """Distance LiDAR (ou None) et nombre de points du premier groupe pour une boîte détectée."""
    return front_cluster(lidar_in_box(uv, depth, box))


def mono_depth(K, box, height=MONO_HEIGHT):
    """Baseline caméra seule : profondeur depuis la hauteur de boîte (véhicule supposé de `height` m)."""
    return float(K[0, 0] * height / max(box[3] - box[1], 1.0))

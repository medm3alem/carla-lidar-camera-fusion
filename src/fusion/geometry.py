"""
Repère CARLA / UE4 : x avant, y droite, z haut.
Repère caméra "vision" : x droite, y bas, z avant (profondeur).
"""
import numpy as np

_CARLA_TO_CV = np.array([[0, 1, 0],
                         [0, 0, -1],
                         [1, 0, 0]], dtype=float)


def intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    f = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return np.array([[f, 0, width / 2.0],
                     [0, f, height / 2.0],
                     [0, 0, 1.0]])


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """T : matrice 4x4, pts : (N,3). Retourne (N,3)."""
    pts_h = np.hstack([pts[:, :3], np.ones((len(pts), 1))])
    return (T @ pts_h.T).T[:, :3]


def lidar_to_camera_matrix(T_world_lidar: np.ndarray,
                           T_world_cam: np.ndarray) -> np.ndarray:
    """Matrice 4x4 qui amène un point du repère LiDAR au repère caméra (CARLA)."""
    return np.linalg.inv(T_world_cam) @ T_world_lidar


def project(pts_cam_carla: np.ndarray, K: np.ndarray,
            width: int, height: int, min_depth: float = 0.1):
    """
    pts_cam_carla : (N,3) dans le repère caméra CARLA.
    Retourne uv (M,2), depth (M,), idx (M,) = indices d'origine des points
    devant la caméra ET dans l'image.
    """
    cv = pts_cam_carla @ _CARLA_TO_CV.T
    depth = cv[:, 2]
    in_front = depth > min_depth
    uvw = (K @ cv[in_front].T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < width) &
              (uv[:, 1] >= 0) & (uv[:, 1] < height))
    idx = np.flatnonzero(in_front)[inside]
    return uv[inside], depth[in_front][inside], idx


def nearest_face_depth(vertices_cam_carla: np.ndarray) -> float:
    """Profondeur de la face la plus proche d'une boîte 3D (min sur les 8 sommets).
    C'est la grandeur comparable à une mesure LiDAR, contrairement à la profondeur du centre."""
    return float((vertices_cam_carla @ _CARLA_TO_CV.T)[:, 2].min())


def project_all(pts_cam_carla: np.ndarray, K: np.ndarray):
    """Projection sans filtrage. Retourne uv (N,2) et depth (N,). A filtrer par l'appelant."""
    cv = pts_cam_carla @ _CARLA_TO_CV.T
    depth = cv[:, 2]
    uvw = (K @ cv.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, depth
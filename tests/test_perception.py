"""Tests unitaires de src/fusion/perception (sans CARLA, sans GPU). Lancer : python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fusion import geometry as geo
from fusion.perception import box_iou, front_cluster, fuse_box, ground_depth, mono_depth, project_lidar


def test_iou_identique_et_disjoint():
    a = [[0, 0, 10, 10]]
    assert np.isclose(box_iou(a, a)[0][0, 0], 1.0)
    assert box_iou(a, [[20, 20, 30, 30]])[0][0, 0] == 0.0


def test_front_cluster_premier_groupe():
    d, n = front_cluster(np.array([10.0, 10.2, 10.1, 25.0, 25.1, 25.2, 25.3]))
    assert n == 3 and abs(d - 10.1) < 1e-9


def test_front_cluster_trop_peu_de_points():
    assert front_cluster(np.array([5.0, 5.1])) == (None, 0)


def test_projection_et_fusion_point_devant():
    # LiDAR et caméra confondus (identité) : un point à 20 m droit devant se projette au centre.
    K = geo.intrinsics(800, 600, 90.0)
    I = np.eye(4)
    pts = np.array([[20.0, 0.0, 0.0, 1.0]] * 5)     # repère CARLA : x avant
    uv, depth = project_lidar(pts, I, I, K, 800, 600, lidar_height=2.4)
    assert len(uv) == 5 and np.allclose(depth, 20.0)
    assert np.allclose(uv[0], [400, 300])
    d, n = fuse_box(uv, depth, np.array([380.0, 280.0, 420.0, 320.0]))
    assert n == 5 and abs(d - 20.0) < 1e-6


def test_sol_retire():
    K = geo.intrinsics(800, 600, 90.0)
    I = np.eye(4)
    pts = np.array([[20.0, 0.0, -2.4, 1.0]] * 5)    # au niveau du sol
    uv, depth = project_lidar(pts, I, I, K, 800, 600, lidar_height=2.4)
    assert len(uv) == 0


def test_mono_depth():
    K = geo.intrinsics(800, 600, 90.0)
    assert abs(mono_depth(K, [0, 0, 10, 30]) - 400 * 1.5 / 30) < 1e-9


def test_ground_depth_geometrie():
    # caméra à 2,4 m, f = 400 px : un point de la route à 20 m se projette à v = cy + f*h/z = 300 + 48 px
    K = geo.intrinsics(800, 600, 90.0)
    box = [380, 250, 420, 348]
    assert abs(ground_depth(K, box, 2.4) - 20.0) < 1e-6
    assert ground_depth(K, [0, 0, 10, 299], 2.4) is None          # bord bas au-dessus de l'horizon : pas de sol

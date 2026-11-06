"""fuse_box_lateral : même profondeur que fuse_box, latéral issu des points (robuste aux boîtes tronquées)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fusion.perception import fuse_box, fuse_box_lateral

K = np.array([[500.0, 0, 400.0], [0, 500.0, 300.0], [0, 0, 1.0]])


def points(lat_m, depth_m, n=60, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.uniform(depth_m[0], depth_m[1], n)
    lat = rng.uniform(lat_m[0], lat_m[1], n)
    uv = np.stack([400.0 + 500.0 * lat / d, rng.uniform(280, 320, n)], 1)
    return uv, d


def test_meme_profondeur_que_fuse_box():
    uv, d = points((-0.8, 0.8), (20.0, 20.5))
    box = [300, 250, 500, 350]
    d1, n1 = fuse_box(uv, d, box)
    d2, n2, _ = fuse_box_lateral(uv, d, box, K)
    assert d1 == d2 and n1 == n2


def test_vehicule_voisin_vu_de_cote_reste_hors_du_couloir():
    """Véhicule à 2,6-4,4 m sur le côté, boîte tronquée au bord droit (u max = 800 px) : le centre de boîte
    donnerait un latéral trop faible, la médiane des points reste proche de 3,5 m."""
    uv, d = points((2.6, 4.4), (5.0, 7.0))
    keep = uv[:, 0] < 800
    uv, d = uv[keep], d[keep]
    box = [float(uv[:, 0].min()) - 5, 250.0, 800.0, 350.0]
    d_l, n, lat = fuse_box_lateral(uv, d, box, K)
    assert d_l is not None and 2.6 <= lat <= 4.4


def test_pas_assez_de_points():
    uv, d = points((0.0, 0.1), (10.0, 10.5), n=2)
    assert fuse_box_lateral(uv, d, [300, 250, 500, 350], K) == (None, 0, None)

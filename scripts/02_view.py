"""Affiche des frames du dataset v4 : boîtes (vert = gardée, orange = ignorée) + points LiDAR.
Usage : python scripts\02_view.py data\v4_test\clear\ep000 [idx ...]"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fusion.config import CFG
from fusion import geometry as geo


def main():
    ep = Path(sys.argv[1])
    idxs = [int(a) for a in sys.argv[2:]] or [0, 20, 40]
    lab = json.loads((ep / "labels.json").read_text())
    K = geo.intrinsics(CFG.img_w, CFG.img_h, CFG.fov)
    out = ROOT / "runs" / "view"
    out.mkdir(parents=True, exist_ok=True)
    for i in idxs:
        fr = next((f for f in lab["frames"] if f["idx"] == i), None)
        if fr is None:
            print(f"frame {i} absente ({len(lab['frames'])} frames dans l'épisode)")
            continue
        img = cv2.imread(str(ep / f"{i:03d}_rgb.png"))
        pts = np.load(ep / f"{i:03d}_lidar.npy")
        T_wc, T_wl = np.array(fr["T_world_cam"]), np.array(fr["T_world_lidar"])
        pc = geo.transform_points(geo.lidar_to_camera_matrix(T_wl, T_wc), pts[:, :3])
        uv, d, _ = geo.project(pc, K, CFG.img_w, CFG.img_h)
        col = cv2.applyColorMap(
            (np.clip(d, 0, CFG.lidar_range) / CFG.lidar_range * 255).astype(np.uint8).reshape(-1, 1),
            cv2.COLORMAP_JET).reshape(-1, 3)
        for (x, y), c in zip(uv.astype(int), col):
            cv2.circle(img, (int(x), int(y)), 1, c.tolist(), -1)
        for o in fr["objects"]:
            x0, y0, x1, y1 = o["box2d"]
            c = (0, 165, 255) if o["ignore"] else (0, 255, 0)
            cv2.rectangle(img, (x0, y0), (x1, y1), c, 1)
            cv2.putText(img, f"{o['distance']:.0f}m {o['n_lidar']}pts", (x0, max(y0 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
        dst = out / f"{ep.parent.name}_{ep.name}_{i:03d}.png"
        cv2.imwrite(str(dst), img)
        print(f"{dst}  ({len(fr['objects'])} objets)")


if __name__ == "__main__":
    main()
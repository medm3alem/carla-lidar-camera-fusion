r"""
Convertit le dataset v4 au format YOLO (une seule classe : vehicle).
- labels : boîtes serrées des véhicules NON ignorés
- images : liens physiques vers les PNG (pas de copie). Exception en train : si la frame contient
  un véhicule "ignoré" (trop caché), on écrit une copie où sa zone est grisée ; les pixels des
  véhicules gardés sont restaurés par-dessus.
Usage : python scripts\04_make_yolo.py [--src data\v4] [--dst data\yolo_v4]
"""
import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fusion.config import CFG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "v4"))
    ap.add_argument("--dst", default=str(ROOT / "data" / "yolo_v4"))
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    W, H = CFG.img_w, CFG.img_h
    stats = defaultdict(lambda: [0, 0])      # (split, météo) -> [images, boîtes]

    for lab_p in sorted(src.glob("*/ep*/labels.json")):
        lab = json.loads(lab_p.read_text())
        split, weather, ep = lab["split"], lab["weather"], lab["episode"]
        img_dir = dst / "images" / split
        lbl_dir = dst / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for fr in lab["frames"]:
            k = fr["idx"]
            name = f"{weather}_ep{ep:03d}_{k:03d}"
            src_img = lab_p.parent / f"{k:03d}_rgb.png"
            dst_img = img_dir / f"{name}.png"
            kept = [o for o in fr["objects"] if not o["ignore"]]
            ignored = [o for o in fr["objects"] if o["ignore"]]

            if dst_img.exists():
                dst_img.unlink()
            if split == "train" and ignored:
                img = cv2.imread(str(src_img))
                orig = img.copy()
                for o in ignored:
                    x0, y0, x1, y1 = o["box2d"]
                    img[y0:y1 + 1, x0:x1 + 1] = 114
                for o in kept:
                    x0, y0, x1, y1 = o["box2d"]
                    img[y0:y1 + 1, x0:x1 + 1] = orig[y0:y1 + 1, x0:x1 + 1]
                cv2.imwrite(str(dst_img), img)
            else:
                try:
                    os.link(src_img, dst_img)
                except OSError:
                    shutil.copy2(src_img, dst_img)

            lines = []
            for o in kept:
                x0, y0, x1, y1 = o["box2d"]          # pixels inclusifs
                lines.append("0 {:.6f} {:.6f} {:.6f} {:.6f}".format(
                    (x0 + x1 + 1) / 2 / W, (y0 + y1 + 1) / 2 / H,
                    (x1 - x0 + 1) / W, (y1 - y0 + 1) / H))
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines))
            stats[(split, weather)][0] += 1
            stats[(split, weather)][1] += len(lines)

    (dst / "dataset.yaml").write_text(
        f"path: {dst.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnames:\n  0: vehicle\n")
    for (sp, w), (n_img, n_box) in sorted(stats.items()):
        print(f"{sp:<6}{w:<6}{n_img:>6} images {n_box:>7} boîtes")


if __name__ == "__main__":
    main()
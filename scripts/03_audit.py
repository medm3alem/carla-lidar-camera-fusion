r"""
Audit du dataset v4 : volumes par split et météo, densité LiDAR par distance,
appariement entre météos, contraste image.
Usage : python scripts\03_audit.py [data\v4]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = [(0, 20), (20, 40), (40, 60)]
MIN_PTS = 5      # seuil indicatif : en dessous, une profondeur LiDAR est peu fiable


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "v4"
    weathers = sorted(p.name for p in root.iterdir() if p.is_dir())
    stats = defaultdict(lambda: defaultdict(int))
    per_ep = defaultdict(dict)                 # ep -> météo -> (frames, objets gardés)
    pts = defaultdict(list)                    # (météo, bin) -> n_lidar des objets gardés
    vis, contrast = [], defaultdict(list)

    for w in weathers:
        for ep_dir in sorted((root / w).glob("ep*")):
            lab_p = ep_dir / "labels.json"
            if not lab_p.exists():
                print(f"incomplet : {ep_dir}")
                continue
            lab = json.loads(lab_p.read_text())
            frames = lab["frames"]
            s = stats[(lab["split"], w)]
            s["ep"] += 1
            s["frames"] += len(frames)
            kept = 0
            for fr in frames:
                for o in fr["objects"]:
                    if o["ignore"]:
                        s["ignores"] += 1
                        continue
                    s["objets"] += 1
                    kept += 1
                    if o["visibility"] is not None:
                        vis.append(o["visibility"])
                    for lo, hi in BINS:
                        if lo <= o["distance"] < hi:
                            pts[(w, (lo, hi))].append(o["n_lidar"])
            per_ep[lab["episode"]][w] = (len(frames), kept)
            if frames:
                f = frames[len(frames) // 2]["idx"]
                g = cv2.imread(str(ep_dir / f"{f:03d}_rgb.png"), cv2.IMREAD_GRAYSCALE)
                if g is not None:
                    contrast[w].append(float(g.std()))

    print(f"\n{'split':<6}{'météo':<7}{'ép.':>4}{'frames':>8}{'objets':>8}{'ignorés':>9}")
    for (sp, w), s in sorted(stats.items()):
        print(f"{sp:<6}{w:<7}{s['ep']:>4}{s['frames']:>8}{s['objets']:>8}{s['ignores']:>9}")

    print(f"\nPoints LiDAR par véhicule gardé (médiane, part avec >= {MIN_PTS} points) :")
    for w in weathers:
        row = []
        for b in BINS:
            a = np.array(pts[(w, b)])
            row.append(f"{b[0]}-{b[1]} m : {np.median(a):.0f}, {100 * (a >= MIN_PTS).mean():.0f} %"
                       if len(a) else f"{b[0]}-{b[1]} m : -")
        print(f"  {w:<6} " + " | ".join(row))

    print("\nVisibilité (percentiles 5/25/50/75/95) :", np.percentile(vis, [5, 25, 50, 75, 95]).round(2))
    print("Contraste moyen par météo :", {w: round(float(np.mean(c)), 1) for w, c in contrast.items()})

    diff = {ep for ep, d in per_ep.items() if len(set(d.values())) > 1}
    print(f"\nÉpisodes dont frames/objets diffèrent selon la météo : {len(diff)}/{len(per_ep)}")
    for ep in sorted(diff):
        print(f"  ep{ep:03d} : " + ", ".join(f"{w} {f}f/{k}obj" for w, (f, k) in per_ep[ep].items()))

    same = tot = 0
    for ep, d in sorted(per_ep.items()):
        if ep in diff or len(d) < 2:
            continue
        k = min(30, min(f for f, _ in d.values()) - 1)
        if k < 0:
            continue
        ws = list(d)
        a = np.load(root / ws[0] / f"ep{ep:03d}" / f"{k:03d}_lidar.npy")
        for w in ws[1:]:
            b = np.load(root / w / f"ep{ep:03d}" / f"{k:03d}_lidar.npy")
            tot += 1
            same += int(a.shape == b.shape and np.allclose(a, b))
    print(f"\nLiDAR identique entre météos (frame ~30, épisodes appariés) : {same}/{tot} comparaisons")


if __name__ == "__main__":
    main()
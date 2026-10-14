r"""Recall par nouveauté de la frame de test (distance à l'entraînement, même cap).
Usage : python scripts\08_novelty_recall.py runs\eval_v4_n_test.csv [data\v4]"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = [(0, 20), (20, 40), (40, 60)]
GROUPS = [("familière (<5 m)", 0, 5), ("intermédiaire (5-20 m)", 5, 20), ("nouvelle (>=20 m)", 20, 1e9)]


def poses(split, root):
    out = {}
    for p in sorted(root.glob("clear/ep*/labels.json")):      # la position ne dépend pas de la météo
        lab = json.loads(p.read_text())
        if lab["split"] != split:
            continue
        for fr in lab["frames"]:
            T = np.array(fr["T_world_cam"])
            out[(lab["episode"], fr["idx"])] = (T[:3, 3], np.degrees(np.arctan2(T[1, 0], T[0, 0])))
    return out


csv_path = Path(sys.argv[1])
root = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "v4"
train, test = poses("train", root), poses("test", root)
Ptr = np.array([p for p, _ in train.values()])
Ytr = np.array([y for _, y in train.values()])

novelty = {}
for k, (p, y) in test.items():
    ok = np.abs((Ytr - y + 180) % 360 - 180) < 30
    novelty[k] = np.linalg.norm(Ptr[ok] - p, axis=1).min() if ok.any() else np.inf

n_frames = defaultdict(int)
for d in novelty.values():
    for g, lo, hi in GROUPS:
        n_frames[g] += lo <= d < hi

stat = defaultdict(lambda: [0, 0])      # (groupe, météo, bin) -> [détectés, total]
for r in csv.DictReader(open(csv_path)):
    k = (int(r["episode"]), int(r["idx"]))
    d = novelty.get(k)
    if d is None:
        continue
    for g, lo, hi in GROUPS:
        if lo <= d < hi:
            for b in BINS:
                if b[0] <= float(r["distance"]) < b[1]:
                    stat[(g, r["weather"], b)][0] += int(r["detected"])
                    stat[(g, r["weather"], b)][1] += 1

for g, _, _ in GROUPS:
    print(f"\n{g} : {n_frames[g]} frames de test (même frame comptée une fois par météo ci-dessous)")
    for w in ("clear", "rain", "fog"):
        cells = []
        for b in BINS:
            n, t = stat[(g, w, b)]
            cells.append(f"{b[0]}-{b[1]} m : {n:>4}/{t:<4} {100 * n / t:5.1f} %" if t else f"{b[0]}-{b[1]} m :    -")
        print(f"  {w:<6}" + " | ".join(cells))
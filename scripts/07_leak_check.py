r"""Distance de chaque frame de test à la frame d'entraînement la plus proche (position + cap)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(split, root):
    P, Y = [], []
    for p in sorted(root.glob("clear/ep*/labels.json")):     # la position ne dépend pas de la météo
        lab = json.loads(p.read_text())
        if lab["split"] != split:
            continue
        for fr in lab["frames"]:
            T = np.array(fr["T_world_cam"])
            P.append(T[:3, 3])
            Y.append(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
    return np.array(P), np.array(Y)


root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "v4"
Ptr, Ytr = load("train", root)
Pte, Yte = load("test", root)
d = np.empty(len(Pte))
for i, (p, y) in enumerate(zip(Pte, Yte)):
    ok = np.abs((Ytr - y + 180) % 360 - 180) < 30            # même cap, à ±30°
    d[i] = np.linalg.norm(Ptr[ok] - p, axis=1).min() if ok.any() else np.inf
print(f"{len(Pte)} frames de test, {len(Ptr)} frames d'entraînement")
for t in (5, 10, 20, 50):
    print(f"  à moins de {t:>2} m d'une frame d'entraînement (même cap) : {100 * (d < t).mean():.0f} %")
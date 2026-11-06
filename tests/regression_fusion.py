"""
Non-régression : compare un CSV produit par 09_fusion_depth.py au CSV de référence versionné.
Usage :
  python scripts\09_fusion_depth.py --data data\v4_town03 --tag regress
  python tests\regression_fusion.py runs\fusion_regress_test.csv reference\fusion_town03_test.csv
"""
import csv
import sys

import numpy as np


def load(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def main(new_p, ref_p, tol=1e-6):
    new, ref = load(new_p), load(ref_p)
    if len(new) != len(ref):
        print(f"ECHEC : {len(new)} détections vs {len(ref)} en référence")
        return 1
    bad = 0
    for i, (a, b) in enumerate(zip(new, ref)):
        for k in ("weather", "episode", "idx", "status"):
            if a[k] != b[k]:
                bad += 1
                print(f"ligne {i}, champ {k} : {a[k]!r} != {b[k]!r}")
        for k in ("conf", "gt_dist", "gt_face", "lidar_depth", "n_cluster", "mono_depth"):
            x, y = a[k], b[k]
            if (x == "") != (y == ""):
                bad += 1
                print(f"ligne {i}, champ {k} : {x!r} != {y!r}")
            elif x != "" and not np.isclose(float(x), float(y), atol=tol, rtol=0):
                bad += 1
                print(f"ligne {i}, champ {k} : {x} != {y}")
        if bad > 20:
            break
    print("OK : identique à la référence" if not bad else f"ECHEC : {bad} écarts")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))

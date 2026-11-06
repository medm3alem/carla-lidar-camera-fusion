r"""
Balayage du banc cinématique AEB (sans CARLA, sans GPU) : taux de collision, vitesse d'impact,
fausses alertes, selon vitesse, cadence de perception et scénario. Résultats = MODÈLE CINÉMATIQUE
(points matériels, perception émulée), pas CARLA : ils valident la logique et orientent les seuils.
Usage : python scripts\10_aeb_sweep.py [--seeds 20]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fusion.adas import simulate


def row(label, runs):
    coll = np.mean([r.collided for r in runs])
    imp = np.mean([r.impact_speed * 3.6 for r in runs if r.collided]) if any(r.collided for r in runs) else 0.0
    gap = np.median([r.min_gap for r in runs])
    fb = np.mean([r.false_brake for r in runs])
    print(f"{label:<44}{100 * coll:>8.0f} %{imp:>10.1f}{gap:>10.1f}{100 * fb:>9.0f} %")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    seeds = range(args.seeds)
    print(f"{'scénario':<44}{'collision':>10}{'impact km/h':>12}{'gap min m':>10}{'faux frein':>10}")
    print("--- A. véhicule arrêté à 60 m (perception : bruit 0,3 m, 10 % de pertes)")
    for fps in (5, 20):
        for v in (30, 50, 70, 90):
            row(f"ego {v} km/h, {fps} Hz",
                [simulate(v0_kmh=v, gap0=60, fps=fps, dropout=0.1, seed=s) for s in seeds])
    print("--- B. véhicule devant à même vitesse qui freine fort (6 m/s2) à t = 2 s, écart 20 m")
    for fps in (5, 20):
        for v in (30, 50, 70):
            row(f"ego et lead {v} km/h, {fps} Hz",
                [simulate(v0_kmh=v, gap0=20, lead_speed_kmh=v, lead_brake_t=2.0, fps=fps,
                          dropout=0.1, seed=s) for s in seeds])
    print("--- C. non dangereux (faux freinages attendus : 0 %)")
    for label, kw in [("suivi à même vitesse, 15 m", dict(v0_kmh=50, lead_speed_kmh=50, gap0=15)),
                      ("lead plus rapide, 10 m", dict(v0_kmh=50, lead_speed_kmh=60, gap0=10)),
                      ("obstacle hors couloir (3 m latéral)", dict(v0_kmh=50, gap0=50, lateral=3.0))]:
        row(label + ", 20 Hz", [simulate(dangerous=False, fps=20, t_max=10, dropout=0.1, seed=s, **kw)
                                for s in seeds])


if __name__ == "__main__":
    main()

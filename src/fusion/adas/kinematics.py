"""
Banc de validation cinématique 1D (sans CARLA) : un ego freiné par l'AEB approche un véhicule devant.
Ce n'est PAS une simulation de conduite : points matériels, freinage à décélération bornée avec délai
d'actionnement, perception émulée (mesure de distance bruitée à la cadence de la caméra).
Sert à vérifier la logique du tracker et de l'AEB et à régler les seuils avant la boucle fermée CARLA.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .aeb import AEB, AEBConfig
from .tracker import Tracker


@dataclass
class SimResult:
    collided: bool
    impact_speed: float          # m/s (vitesse relative à l'impact, 0 si pas de collision)
    min_gap: float               # m
    t_warning: Optional[float]   # s depuis le début
    t_brake: Optional[float]
    false_brake: bool            # freinage déclenché alors que le scénario n'est pas dangereux
    final_speed: float


def simulate(v0_kmh=50.0, gap0=50.0, lead_speed_kmh=0.0, lead_brake_t=None, lead_decel=6.0,
             fps=20.0, sigma=0.3, dropout=0.0, act_delay=0.2, a_max=8.0, lateral=0.0,
             detect_range=60.0, cfg: AEBConfig = AEBConfig(), seed=0, t_max=20.0, dangerous=True):
    rng = np.random.default_rng(seed)
    v, vl = v0_kmh / 3.6, lead_speed_kmh / 3.6
    gap, t = gap0, 0.0
    dt_sim, dt_frame = 0.005, 1.0 / fps
    next_frame = 0.0
    tracker, aeb = Tracker(dt_frame), AEB(cfg, dt_frame)
    brake_cmd, applied = 0.0, 0.0
    pending = []                 # (t_effet, consigne)
    t_warn = t_brake = None
    min_gap, braked, collided, impact = gap0, False, False, 0.0

    while t < t_max and v > 0.01 or (t < t_max and vl > 0.01 and gap < detect_range):
        if t >= next_frame:
            next_frame += dt_frame
            meas = []
            if gap < detect_range and rng.random() >= dropout:
                meas = [(gap + rng.normal(0, sigma), lateral + rng.normal(0, 0.1))]
            tracks = tracker.step(meas)
            out = aeb.step(tracks, v)
            if out.warning and t_warn is None:
                t_warn = t
            if out.brake > 0 and t_brake is None:
                t_brake = t
            braked |= out.brake > 0
            pending.append((t + act_delay, out.brake))
        while pending and pending[0][0] <= t:
            applied = pending.pop(0)[1]
        v = max(0.0, v - applied * a_max * dt_sim)
        if lead_brake_t is not None and t >= lead_brake_t:
            vl = max(0.0, vl - lead_decel * dt_sim)
        gap += (vl - v) * dt_sim
        min_gap = min(min_gap, gap)
        t += dt_sim
        if gap <= 0 and abs(lateral) < 2.0:       # obstacle hors couloir : pas de collision
            collided, impact = True, v - vl
            break
        if v <= 0.01 and vl <= 0.01:
            break
        if v <= vl and gap > detect_range:
            break
    return SimResult(collided, max(impact, 0.0), min_gap, t_warn, t_brake,
                     braked and not dangerous, v)

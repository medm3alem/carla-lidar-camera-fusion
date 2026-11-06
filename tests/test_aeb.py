"""Tests du tracker, de l'AEB et du banc cinématique (sans CARLA). Lancer : python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fusion.adas import (AEB, AEBConfig, EMERGENCY_BRAKING, NORMAL, PARTIAL_BRAKING, Tracker, simulate)


def run_tracker(ranges, dt=0.05, y=0.0):
    tr = Tracker(dt)
    out = []
    for r in ranges:
        out = tr.step([(r, y)])
    return tr, out


def test_tracker_estime_la_vitesse_relative():
    dt, v = 0.05, 10.0
    ranges = [50 - v * dt * k for k in range(40)]
    _, tracks = run_tracker(ranges, dt)
    assert len(tracks) == 1
    assert abs(tracks[0].closing_speed - v) < 0.5


def test_tracker_deux_objets_deux_pistes():
    tr = Tracker(0.05)
    for _ in range(5):
        tracks = tr.step([(20.0, 0.0), (40.0, 3.0)])
    assert len(tracks) == 2


def test_tracker_supprime_piste_perdue():
    tr = Tracker(0.05)
    for _ in range(4):
        tr.step([(20.0, 0.0)])
    for _ in range(10):
        tracks = tr.step([])
    assert tracks == [] and tr.tracks == []


def test_aeb_normal_sans_piste():
    aeb = AEB(dt=0.05)
    assert aeb.step([], 14.0).state == NORMAL


def test_aeb_urgence_obstacle_arrete_proche():
    # ego 14 m/s, obstacle arrêté à 12 m : TTC < 1 s, décélération requise infinie
    dt = 0.05
    tr, aeb = Tracker(dt), AEB(dt=dt)
    r, out = 25.0, None
    for _ in range(40):
        tracks = tr.step([(r, 0.0)])
        out = aeb.step(tracks, 14.0)
        r -= 14.0 * dt
        if out.state == EMERGENCY_BRAKING:
            break
    assert out.state == EMERGENCY_BRAKING and out.brake == 1.0 and r > 0


def test_aeb_ignore_hors_couloir():
    dt = 0.05
    tr, aeb = Tracker(dt), AEB(dt=dt)
    r = 30.0
    for _ in range(30):
        out = aeb.step(tr.step([(r, 3.0)]), 14.0)
        r -= 14.0 * dt
    assert out.state == NORMAL


def test_banc_pas_de_collision_50kmh_arrete_a_60m():
    # décélération max 5 m/s^2 : à 20 Hz aucune collision ; à 5 Hz la marge est mince, on exige un
    # impact résiduel < 5 km/h (un tirage sur 10 touche à ~1 km/h)
    for seed in range(10):
        assert not simulate(v0_kmh=50, gap0=60, fps=20, dropout=0.1, seed=seed).collided
        r = simulate(v0_kmh=50, gap0=60, fps=5, dropout=0.1, seed=seed)
        assert r.impact_speed * 3.6 < 5.0


def test_banc_pas_de_faux_freinage_suivi_stable():
    for seed in range(20):
        r = simulate(v0_kmh=50, lead_speed_kmh=50, gap0=15, fps=20, dropout=0.1, seed=seed,
                     t_max=10, dangerous=False)
        assert not r.false_brake


def test_aeb_garde_le_frein_a_tres_basse_vitesse_pres_d_un_obstacle():
    # régression : relâchement à ~1 m/s à 11 m d'un obstacle arrêté
    from fusion.adas import PARTIAL_BRAKING
    dt = 0.05
    tr, aeb = Tracker(dt), AEB(dt=dt)
    r, v, out = 40.0, 13.9, None
    for _ in range(300):
        out = aeb.step(tr.step([(r, 0.0)]), v)
        v = max(1.0, v - (out.brake * 4.5) * dt)      # l'ego ralentit mais reste à ~1 m/s
        r = max(r - v * dt, 5.0)
    assert out.state >= PARTIAL_BRAKING and r < 15.0


def test_tracker_ca_estime_une_acceleration_relative():
    dt, g, v = 0.05, 40.0, 0.0                   # écart croissant : cible qui freine moins vite que l'ego... ici cible qui accélère
    tr = Tracker(dt, model="ca", jerk=2.0)
    x, vx = 30.0, -10.0
    for _ in range(80):
        vx += 3.0 * dt                           # accélération relative +3 m/s^2
        x += vx * dt
        tracks = tr.step([(x, 0.0)])
    assert abs(tracks[0].rel_accel - 3.0) < 1.0


def test_banc_cible_qui_freine_avec_anticipation():
    from dataclasses import replace
    cfg = replace(AEBConfig(), use_lead_decel=True)
    for seed in range(10):
        r = simulate(v0_kmh=50, gap0=30, lead_speed_kmh=50, lead_brake_t=2.0, lead_decel=5.0, fps=20,
                     dropout=0.1, seed=seed, cfg=cfg, tracker_model="ca", jerk=2.0)
        assert not r.collided


class _Piste:
    def __init__(self, rng, closing):
        self.id, self.range, self.lateral, self.closing_speed, self.rel_accel = 1, rng, 0.0, closing, 0.0


def _tient_le_frein(hysteresis):
    aeb = AEB(AEBConfig(a_warn=1.5, a_partial=2.5, a_emergency=3.5, brake_partial=0.6, hysteresis=hysteresis), dt=0.05)
    for _ in range(4):                                    # a_req = 2,63 m/s^2 : freinage partiel
        out = aeb.step([_Piste(24.0, 10.0)], 12.0)
    assert out.state == PARTIAL_BRAKING
    for _ in range(60):                                   # le freinage réussit : a_req ~ 2,4, sous le seuil 2,5
        out = aeb.step([_Piste(17.6, 8.0)], 8.0)          # obstacle à plus de hold_range
    return out.state


def test_aeb_hysteresis_ne_relache_pas_quand_a_req_oscille_sous_le_seuil():
    assert _tient_le_frein(0.7) == PARTIAL_BRAKING
    assert _tient_le_frein(1.0) < PARTIAL_BRAKING         # sans hystérésis : le frein est relâché (cas CARLA 50 km/h)


def test_aeb_relache_quand_la_menace_disparait():
    aeb = AEB(dt=0.05)
    for _ in range(4):
        aeb.step([_Piste(24.0, 10.0)], 12.0)
    for _ in range(40):
        out = aeb.step([_Piste(40.0, 3.0)], 6.0)          # a_req ~ 0,1 : menace levée
    assert out.state == NORMAL


# Mesures (distance, latéral) relevées à 5 Hz pendant le dépassement d'un véhicule arrêté dans la voie voisine
# (CARLA, 50 km/h, graine 43). Deux détections voisines du même véhicule (11,4 m / 3,8 m puis 10,3 m / 2,7 m) créaient
# une piste dont la vitesse latérale initiale (-5 m/s) l'extrapolait dans le couloir : freinage d'urgence parasite.
DEPASSEMENT = [[(15.8, 3.2), (36.1, 6.4)], [(13.9, 3.2), (36.1, 6.4)], [(11.1, 3.1), (11.4, 3.8), (25.0, 6.0)],
               [(8.6, 3.0), (10.3, 2.7)], [(5.8, 2.9), (18.9, 5.7)], [(3.2, 2.7), (13.1, 4.7), (5.0, 3.2)],
               [(2.5, 2.5), (7.0, 5.8)], [(19.4, 16.0)]]


def _depassement(vy_init_std):
    tr = Tracker(0.2, model="ca", jerk=2.0, vy_init_std=vy_init_std)
    aeb = AEB(AEBConfig(use_lead_decel=True, t_latency=0.85, escalate_max_misses=None), 0.2)   # sans la garde
    return max(aeb.step(tr.step(m), 13.88).state for m in DEPASSEMENT)


def test_depassement_a_5hz_ne_declenche_pas_le_freinage():
    assert _depassement(2.0) == NORMAL
    assert _depassement(8.0) == EMERGENCY_BRAKING     # ancien a priori : reproduit l'événement CARLA


def _depassement_garde(garde):
    tr = Tracker(0.2, model="ca", jerk=2.0, vy_init_std=8.0)       # ancien a priori : piste fantôme qui dérive
    cfg = AEBConfig(use_lead_decel=True, t_latency=0.85, escalate_max_misses=2 if garde else None)
    aeb = AEB(cfg, 0.2)
    return max(aeb.step(tr.step(m), 13.88).state for m in DEPASSEMENT)


def test_garde_piste_en_roue_libre():
    assert _depassement_garde(False) == EMERGENCY_BRAKING
    assert _depassement_garde(True) == NORMAL


def test_garde_ne_retarde_pas_une_cible_reelle():
    # cible arrêtée à 50 m, ego à 13,9 m/s, une mesure perdue sur 4 : doit quand même freiner jusqu'à l'arrêt
    tr, aeb, x, st = Tracker(0.05, model="ca"), AEB(AEBConfig(), 0.05), 50.0, 0
    for k in range(120):
        x -= 13.88 * 0.05
        st = max(st, aeb.step(tr.step([] if k % 4 == 3 else [(x, 0.0)]), 13.88).state)
    assert st == EMERGENCY_BRAKING

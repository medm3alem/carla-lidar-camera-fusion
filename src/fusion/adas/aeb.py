"""
Freinage d'urgence automatique (AEB) : machine d'états pilotée par le temps avant collision (TTC).
Entrées : pistes confirmées du tracker, vitesse de l'ego. Sorties : état, consigne de frein [0,1], alerte.

  NORMAL -> WARNING (TTC < ttc_warn) -> PARTIAL_BRAKING (TTC < ttc_partial) -> EMERGENCY_BRAKING
Escalade : le niveau doit être demandé sur `confirm_frames` images consécutives (anti-fausse alarme).
Désescalade : après `release_hold_s` sans demande, et seulement si l'ego roule encore ou si plus aucun
obstacle n'est à moins de `hold_range` m (on garde le frein à l'arrêt derrière un obstacle).
Critère principal : décélération requise pour s'arrêter avant l'obstacle, latence comprise,
  a_req = v_c^2 / (2 (x - v_c * t_latence - marge))     (infinie si la distance est déjà insuffisante)
Un seuil de TTC fixe ne convient pas à toutes les vitesses (essai : il freine trop tard à 50 km/h
avec un délai d'actionnement de 0,2 s) ; le TTC ne sert donc qu'à l'alerte.
Les seuils sont des valeurs de départ à régler par balayage, pas des valeurs homologuées.
"""
from collections import deque
from dataclasses import dataclass
from typing import Optional

NORMAL, WARNING, PARTIAL_BRAKING, EMERGENCY_BRAKING = 0, 1, 2, 3
NAMES = ["NORMAL", "WARNING", "PARTIAL_BRAKING", "EMERGENCY_BRAKING"]


@dataclass(frozen=True)
class AEBConfig:
    corridor_half_width: float = 1.5     # m, demi-largeur du couloir de trajectoire
    max_range: float = 60.0
    min_closing: float = 0.5             # m/s : en dessous, pas de menace
    ttc_warn: float = 2.6                # alerte seulement
    a_warn: float = 2.0                  # m/s^2 : décélération requise -> alerte
    a_partial: float = 4.0               # -> freinage partiel
    a_emergency: float = 6.0             # -> freinage d'urgence (a_max supposé ~8 m/s^2)
    t_latency: float = 0.4               # s : actionnement + traitement
    stop_margin: float = 1.0             # m : distance d'arrêt visée devant l'obstacle
    brake_partial: float = 0.4
    brake_emergency: float = 1.0
    confirm_frames: int = 2
    release_hold_s: float = 1.0
    hold_range: float = 15.0
    stopped_speed: float = 0.5


@dataclass
class AEBOutput:
    state: int
    brake: float
    warning: bool
    target_id: Optional[int]
    ttc: Optional[float]
    a_req: float = 0.0

    @property
    def state_name(self):
        return NAMES[self.state]


class AEB:
    def __init__(self, cfg: AEBConfig = AEBConfig(), dt: float = 0.05):
        self.cfg = cfg
        self.dt = dt
        self.state = NORMAL
        self.hist = deque(maxlen=cfg.confirm_frames)
        self.below_time = 0.0

    def required_decel(self, rng, closing):
        """Décélération (m/s^2) nécessaire pour s'arrêter à stop_margin de l'obstacle, latence comprise."""
        c = self.cfg
        room = rng - closing * c.t_latency - c.stop_margin
        return float("inf") if room <= 0 else closing ** 2 / (2.0 * room)

    def _raw_level(self, ttc, a_req):
        c = self.cfg
        if ttc is None:
            return NORMAL
        if a_req >= c.a_emergency:
            return EMERGENCY_BRAKING
        if a_req >= c.a_partial:
            return PARTIAL_BRAKING
        if a_req >= c.a_warn or ttc < c.ttc_warn:
            return WARNING
        return NORMAL

    def step(self, tracks, ego_speed):
        c = self.cfg
        in_path = [t for t in tracks if 0 < t.range < c.max_range and abs(t.lateral) < c.corridor_half_width]
        threats = [t for t in in_path if t.closing_speed > c.min_closing]
        target, ttc, a_req, raw = None, None, 0.0, NORMAL
        for t in threats:
            v = t.range / t.closing_speed
            a = self.required_decel(t.range, t.closing_speed)
            lvl = self._raw_level(v, a)
            if target is None or lvl > raw or (lvl == raw and v < ttc):
                target, ttc, a_req, raw = t, v, a, lvl

        self.hist.append(raw)
        if len(self.hist) == self.hist.maxlen and min(self.hist) > self.state:
            self.state = min(self.hist)
            self.below_time = 0.0
        elif raw < self.state:
            obstacle_close = any(t.range < c.hold_range for t in in_path)
            if ego_speed > c.stopped_speed or not obstacle_close:
                self.below_time += self.dt
                if self.below_time >= c.release_hold_s:
                    self.state = raw
                    self.below_time = 0.0
            else:
                self.below_time = 0.0
        else:
            self.below_time = 0.0

        brake = {PARTIAL_BRAKING: c.brake_partial, EMERGENCY_BRAKING: c.brake_emergency}.get(self.state, 0.0)
        return AEBOutput(self.state, brake, self.state >= WARNING,
                         target.id if target else None, ttc, a_req)

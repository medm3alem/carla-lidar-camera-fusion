"""
Freinage d'urgence automatique (AEB) : machine d'états pilotée par le temps avant collision (TTC).
Entrées : pistes confirmées du tracker, vitesse de l'ego. Sorties : état, consigne de frein [0,1], alerte.

  NORMAL -> WARNING (TTC < ttc_warn) -> PARTIAL_BRAKING (TTC < ttc_partial) -> EMERGENCY_BRAKING
Escalade : le niveau doit être demandé sur `confirm_frames` images consécutives (anti-fausse alarme).
Désescalade : après `release_hold_s` sans demande, et seulement si plus aucun obstacle du couloir n'est à
moins de `hold_range` m sans s'éloigner (on garde le frein tant qu'un obstacle proche n'ouvre pas la distance,
à l'arrêt comme à très basse vitesse : un essai a montré qu'un relâchement à 1 m/s, avec un conducteur qui
ré-accélère, ramène l'ego à 0,5 m de l'obstacle).
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
    # Seuils calibrés sur la décélération mesurée de la voiture CARLA (4 à 6 m/s^2 à frein 100 %, et non 8).
    # Voir AEBConfig.legacy() pour les valeurs initiales (calées sur 8 m/s^2), gardées pour comparaison.
    a_warn: float = 1.5                  # m/s^2 : décélération requise -> alerte
    a_partial: float = 2.5               # -> freinage partiel
    a_emergency: float = 3.5             # -> freinage d'urgence
    t_latency: float = 0.4               # s : actionnement + traitement
    stop_margin: float = 1.0             # m : distance d'arrêt visée devant l'obstacle
    brake_partial: float = 0.6
    brake_emergency: float = 1.0
    confirm_frames: int = 2
    release_hold_s: float = 1.0
    hold_range: float = 15.0
    # Hystérésis de désescalade : un niveau n'est quitté que si la décélération requise tombe sous
    # `hysteresis` x son seuil (sinon un freinage qui réussit fait osciller a_req autour du seuil, relâche
    # le frein à 16 m d'un obstacle fixe à 7,5 m/s, et il faut ensuite freiner plus fort : essai CARLA, 50 km/h).
    hysteresis: float = 0.7
    stopped_speed: float = 0.5
    # Anticipation d'une cible qui freine (nécessite Tracker(model="ca")) : désactivée par défaut
    use_lead_decel: bool = False
    lead_decel_min: float = 3.0          # m/s^2 : ~4 écarts-types du bruit d'estimation (saccade 2 m/s^3)
    lead_decel_frames: int = 3           # images consécutives sous le seuil avant d'anticiper
    lead_decel_cap: float = 8.0          # m/s^2 : plafond de la décélération supposée de la cible

    @classmethod
    def legacy(cls):
        """Seuils de la première version (a_max supposé 8 m/s^2) : référence de comparaison."""
        return cls(a_warn=2.0, a_partial=4.0, a_emergency=6.0, brake_partial=0.4)


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
        self._prev_v = None
        self.ego_accel = 0.0
        self._brake_cnt = {}                              # id de piste -> images consécutives de freinage de la cible

    def required_decel(self, rng, closing):
        """Décélération (m/s^2) nécessaire pour s'arrêter à stop_margin de l'obstacle, latence comprise."""
        c = self.cfg
        room = rng - closing * c.t_latency - c.stop_margin
        return float("inf") if room <= 0 else closing ** 2 / (2.0 * room)

    def _raw_level(self, ttc, a_req, scale=1.0):
        """Niveau demandé ; `scale` < 1 abaisse les seuils (niveau de maintien, pour l'hystérésis)."""
        c = self.cfg
        if ttc is None:
            return NORMAL
        if a_req >= c.a_emergency * scale:
            return EMERGENCY_BRAKING
        if a_req >= c.a_partial * scale:
            return PARTIAL_BRAKING
        if a_req >= c.a_warn * scale or ttc < c.ttc_warn:
            return WARNING
        return NORMAL

    def required_decel_lead_stop(self, rng, ve, vl, al):
        """Décélération pour s'arrêter avant le point où la cible (qui freine à `al` < 0) va s'arrêter."""
        c = self.cfg
        d_l = vl * vl / (2.0 * min(abs(al), c.lead_decel_cap))
        room = rng + d_l - ve * c.t_latency - c.stop_margin
        return float("inf") if room <= 0 else ve * ve / (2.0 * room)

    def step(self, tracks, ego_speed):
        c = self.cfg
        if self._prev_v is not None:                      # accélération de l'ego, filtrée (EMA)
            self.ego_accel += 0.3 * ((ego_speed - self._prev_v) / self.dt - self.ego_accel)
        self._prev_v = ego_speed
        in_path = [t for t in tracks if 0 < t.range < c.max_range and abs(t.lateral) < c.corridor_half_width]
        threats = [t for t in in_path if t.closing_speed > c.min_closing]
        target, ttc, a_req, raw, hold = None, None, 0.0, NORMAL, NORMAL
        for t in threats:
            v = t.range / t.closing_speed
            a = self.required_decel(t.range, t.closing_speed)
            if c.use_lead_decel:
                al = t.rel_accel + self.ego_accel         # accélération absolue de la cible
                vl = max(ego_speed - t.closing_speed, 0.0)
                n = self._brake_cnt.get(t.id, 0) + 1 if (al < -c.lead_decel_min and vl > 0.5) else 0
                self._brake_cnt[t.id] = n
                if n >= c.lead_decel_frames:
                    a = max(a, self.required_decel_lead_stop(t.range, ego_speed, vl, al))
            lvl = self._raw_level(v, a)
            hold = max(hold, self._raw_level(v, a, c.hysteresis))
            if target is None or lvl > raw or (lvl == raw and v < ttc):
                target, ttc, a_req, raw = t, v, a, lvl

        self.hist.append(raw)
        if len(self.hist) == self.hist.maxlen and min(self.hist) > self.state:
            self.state = min(self.hist)
            self.below_time = 0.0
        elif hold < self.state:
            obstacle_close = any(t.range < c.hold_range and t.closing_speed > -c.min_closing for t in in_path)
            if not obstacle_close:
                self.below_time += self.dt
                if self.below_time >= c.release_hold_s:
                    self.state = hold
                    self.below_time = 0.0
            else:
                self.below_time = 0.0
        else:
            self.below_time = 0.0

        brake = {PARTIAL_BRAKING: c.brake_partial, EMERGENCY_BRAKING: c.brake_emergency}.get(self.state, 0.0)
        return AEBOutput(self.state, brake, self.state >= WARNING,
                         target.id if target else None, ttc, a_req)

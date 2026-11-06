"""
Suivi multi-objets dans le repère du véhicule (x = distance devant, y = latéral, droite positive).
Filtre de Kalman à vitesse constante sur (x, y, vx, vy). La vitesse est RELATIVE à l'ego :
un objet à l'arrêt devant un ego qui roule à 14 m/s donne vx = -14 m/s.
Les mesures viennent du LiDAR (distance de la face avant) ; une détection sans mesure LiDAR
ne crée pas de piste (la preuve physique est exigée).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    id: int
    x: np.ndarray                  # [x, y, vx, vy]
    P: np.ndarray
    hits: int = 1
    misses: int = 0
    age: int = 1
    confirmed: bool = False

    @property
    def range(self):
        return float(self.x[0])

    @property
    def lateral(self):
        return float(self.x[1])

    @property
    def closing_speed(self):
        """m/s, positif si l'objet se rapproche."""
        return float(-self.x[2])


class Tracker:
    def __init__(self, dt, r_pos=0.3, sigma_a=6.0, v_init_std=8.0,
                 confirm_hits=2, max_misses=3):
        self.dt = dt
        self.R = np.diag([r_pos ** 2, r_pos ** 2])
        self.sigma_a = sigma_a
        self.v_init_std = v_init_std
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.gate_x = 2.0 + 25.0 * dt          # m : tient compte du déplacement relatif entre 2 images
        self.gate_y = 2.0
        self.tracks: list[Track] = []
        self._next_id = 1
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)

    def _F_Q(self):
        dt = self.dt
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.sigma_a ** 2
        Q = np.zeros((4, 4))
        for i in (0, 1):
            Q[i, i] = q * dt ** 4 / 4
            Q[i, i + 2] = Q[i + 2, i] = q * dt ** 3 / 2
            Q[i + 2, i + 2] = q * dt ** 2
        return F, Q

    def step(self, measurements):
        """measurements : liste de (x, y). Retourne les pistes confirmées."""
        F, Q = self._F_Q()
        for t in self.tracks:
            t.x = F @ t.x
            t.P = F @ t.P @ F.T + Q
            t.age += 1

        pairs = []
        for ti, t in enumerate(self.tracks):
            for mi, (mx, my) in enumerate(measurements):
                dx = abs(mx - t.x[0]) / self.gate_x
                dy = abs(my - t.x[1]) / self.gate_y
                if dx < 1 and dy < 1:
                    pairs.append((dx * dx + dy * dy, ti, mi))
        used_t, used_m = set(), set()
        for _, ti, mi in sorted(pairs):
            if ti in used_t or mi in used_m:
                continue
            used_t.add(ti)
            used_m.add(mi)
            t = self.tracks[ti]
            z = np.array(measurements[mi], float)
            S = self.H @ t.P @ self.H.T + self.R
            K = t.P @ self.H.T @ np.linalg.inv(S)
            t.x = t.x + K @ (z - self.H @ t.x)
            t.P = (np.eye(4) - K @ self.H) @ t.P
            t.hits += 1
            t.misses = 0
            if t.hits >= self.confirm_hits:
                t.confirmed = True

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        for mi, (mx, my) in enumerate(measurements):
            if mi not in used_m:
                P = np.diag([self.R[0, 0], self.R[1, 1], self.v_init_std ** 2, self.v_init_std ** 2])
                self.tracks.append(Track(self._next_id, np.array([mx, my, 0.0, 0.0]), P))
                self._next_id += 1
        return [t for t in self.tracks if t.confirmed]

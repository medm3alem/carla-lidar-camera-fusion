"""
Injection de défauts de perception pour tester la robustesse de la chaîne tracker + AEB, sans CARLA.
Chaque appel reçoit les mesures (distance_pare-chocs, latéral) d'une image et renvoie les mesures dégradées :
  - fantôme : fausse détection dans le couloir, fixe dans le monde (sa distance diminue de la distance
              parcourue par l'ego), qui dure `ghost_len` images consécutives ; démarre avec la probabilité
              `ghost_prob` à chaque image, à une distance tirée dans `ghost_range` ;
  - perte   : chaque mesure est supprimée avec la probabilité `drop_prob` (image sans détection) ;
  - bruit   : bruit gaussien d'écart-type `noise_sigma` (m) sur la distance.
Ce sont des modèles de défauts simples choisis pour le test, pas une caractérisation du détecteur réel.
"""
import numpy as np


class FaultInjector:
    def __init__(self, ghost_prob=0.0, ghost_len=1, ghost_range=(10.0, 40.0), drop_prob=0.0,
                 noise_sigma=0.0, seed=0):
        self.ghost_prob, self.ghost_len, self.ghost_range = ghost_prob, max(int(ghost_len), 1), ghost_range
        self.drop_prob, self.noise_sigma = drop_prob, noise_sigma
        self.rng = np.random.default_rng(seed)
        self._ghost_left = 0
        self._ghost_range = 0.0
        self.n_ghost_frames = 0

    @property
    def active(self):
        return self.ghost_prob > 0 or self.drop_prob > 0 or self.noise_sigma > 0

    def apply(self, meas, ego_speed, dt):
        out = []
        for r, y in meas:
            if self.drop_prob > 0 and self.rng.random() < self.drop_prob:
                continue
            out.append((r + (self.rng.normal(0.0, self.noise_sigma) if self.noise_sigma > 0 else 0.0), y))
        if self._ghost_left == 0 and self.ghost_prob > 0 and self.rng.random() < self.ghost_prob:
            self._ghost_left = self.ghost_len
            self._ghost_range = float(self.rng.uniform(*self.ghost_range))
        if self._ghost_left > 0:
            if self._ghost_range > 1.0:
                out.append((self._ghost_range, 0.0))
                self.n_ghost_frames += 1
            self._ghost_left -= 1
            self._ghost_range -= ego_speed * dt
        return out

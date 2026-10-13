from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Serveur CARLA ---
    host: str = "localhost"
    port: int = 2000
    tm_port: int = 8000
    town: str = "Town10HD_Opt"

    # --- Simulation ---
    dt: float = 0.05                 # 20 Hz
    save_every: int = 4              # 1 frame sur 4 -> 5 Hz
    seed: int = 42

    # --- Épisodes ---
    n_episodes: int = 60
    target_frames: int = 60            # frames sauvegardées par épisode (5 Hz, soit 12 s de conduite)
    max_episode_seconds: float = 40.0  # arrêt même si la cible n'est pas atteinte (ego bloqué)
    warmup_seconds: float = 3.0      # laisser le trafic se mettre en place
    n_npc: int = 70
    min_ego_speed: float = 0.5       # m/s ; en dessous, la frame est ignorée (doublons à l'arrêt)
    weathers: tuple = ("clear", "rain", "fog")

    # --- Caméra RGB / segmentation d'instance / profondeur (même pose) ---
    img_w: int = 800
    img_h: int = 600
    fov: float = 90.0
    cam_pos: tuple = (1.5, 0.0, 2.4)     # x avant, y droite, z haut (repère CARLA)

    # --- LiDAR (à ajuster selon le rendu) ---
    lidar_pos: tuple = (0.0, 0.0, 2.4)
    lidar_channels: int = 64
    lidar_points_per_second: int = 1_200_000
    lidar_range: float = 70.0
    lidar_upper_fov: float = 5.0
    lidar_lower_fov: float = -20.0
    lidar_dropoff: float = 0.0       # CARLA supprime 45 % des points par défaut

    # --- Étiquettes ---
    max_label_distance: float = 60.0
    min_visibility: float = 0.3      # en dessous : zone "à ignorer"

    # --- Découpage par épisode ---
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    def split_of(self, episode_id: int) -> str:
        n_test = round(self.n_episodes * self.test_fraction)
        n_val = round(self.n_episodes * self.val_fraction)
        if episode_id >= self.n_episodes - n_test:
            return "test"
        if episode_id >= self.n_episodes - n_test - n_val:
            return "val"
        return "train"


CFG = Config()
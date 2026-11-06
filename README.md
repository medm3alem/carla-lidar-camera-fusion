# carla-lidar-camera-fusion

Fusion tardive caméra + LiDAR sous CARLA 0.9.16 (Windows, Python 3.12) : collecte synchronisée,
étiquetage automatique, YOLO fine-tuné, estimation de distance et rejet de fausses alarmes par LiDAR,
puis suivi (Kalman) et freinage d'urgence automatique (AEB) validé en boucle fermée.

## Pipeline (scripts/)
| Script | Rôle |
|---|---|
| 00_rig_test | test du banc de capteurs |
| 01_collect | collecte (épisodes x météos clear/rain/fog), reprise possible |
| 02_view / 03_audit | visualisation et audit du dataset |
| 04_make_yolo | conversion au format YOLO (1 classe : vehicle) |
| 05_eval_yolo | évaluation par distance et météo |
| 06_train_yolo | entraînement |
| 07_leak_check / 08_novelty_recall | contrôle de fuite train/test, recall par nouveauté |
| 09_fusion_depth | fusion LiDAR : profondeur et rejet de fausses alarmes |
| 10_aeb_sweep | balayage de l'AEB sur un modèle cinématique (sans CARLA) |
| 11_aeb_closed_loop | un essai AEB en boucle fermée dans CARLA (YOLO -> LiDAR -> tracker -> AEB -> frein) |
| 12_aeb_campaign | campagne d'essais (vitesses x météos x graines), tableaux agrégés |

## Code (src/fusion/)
`config.py` (paramètres), `geometry.py` (repères, projection), `rig.py` (capteurs CARLA, étiquettes),
`perception/` (fusion caméra-LiDAR), `adas/` (tracker de Kalman, AEB, banc cinématique).
`backend/` : prototype d'API FastAPI + page web pour lancer et suivre un run (en cours, voir plus bas).

## AEB : méthode
- **Perception** : boîtes YOLO, points LiDAR (sol retiré) projetés dans la boîte, premier groupe de profondeur ;
  pas de profondeur LiDAR = pas de piste.
- **Tracker** : Kalman dans le repère véhicule, vitesse et accélération relatives (modèle `ca`).
- **AEB** : machine d'états NORMAL / WARNING / PARTIAL / EMERGENCY pilotée par la décélération requise
  `a_req = v² / (2 (x - v·t_latence - marge))`, confirmation sur 2 images, hystérésis à la désescalade,
  anticipation optionnelle d'une cible qui freine. Seuils calibrés sur la décélération mesurée de la voiture
  CARLA (≈ 4 m/s² à 70 km/h à frein 100 %), pas sur des valeurs homologuées.

## AEB : résultats mesurés (CARLA, 20 Hz, Town10HD_Opt, météo claire)
Banc de test maison : conducteur et régulateur de l'ego écrits pour l'essai, pas de trafic. 2 à 4 graines par cas.

| Scénario | Résultat |
|---|---|
| Cible arrêtée, 30 / 50 / 70 km/h (4 graines) | 0 collision ; écart minimal médian 11,9 / 9,2 / 4,3 m |
| Cible qui freine, 50 km/h, écart 20 et 25 m (2 graines) | 0 collision (sans anticipation : 2/2 collisions, impact 23 km/h à 20 m) |
| Cible hors couloir, suivi à même vitesse, approche normale | 0 freinage parasite observé (alerte seule en approche à 20 et 18 m) |

Les CSV figés sont dans `reference/` (`aeb_*_v3_*.csv`). Ils sont reproductibles avec
`python scripts\12_aeb_campaign.py --scenario stopped --speeds 30 50 70 --weathers clear --seeds 42 43 44 45`.

## Reproduire
```
python -m venv venv ; venv\Scripts\activate
pip install -r requirements.txt   # + wheel CARLA, voir requirements.txt
python scripts\01_collect.py      # serveur CARLA lancé
python -m pytest tests            # tests sans CARLA
```
Les données (data/) et les runs (runs/) ne sont pas versionnés ; les résultats de référence
sont dans reference/.

## Limites connues
- Simulation uniquement, une seule ville (Town10HD_Opt), peu de graines : ces essais montrent un comportement,
  ils ne constituent pas une validation statistique ni une homologation.
- Le LiDAR CARLA n'est pas dégradé par pluie/brouillard : l'axe météo ne teste que la caméra.
- Modèles de véhicules identiques entre train et test ; baseline caméra seule naïve (hauteur fixe 1,5 m).
- L'AEB est validé à 20 Hz ; le Worker web actuel tourne à 5 Hz, où le banc cinématique donne des collisions
  à haute vitesse. Pas encore de test avec bruit capteur ou fausses détections injectées.
- Les freinages parasites ne sont mesurés que sur trois scénarios simples, pas sur du trafic réel.

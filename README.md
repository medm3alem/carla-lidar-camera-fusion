# carla-lidar-camera-fusion

Fusion tardive caméra + LiDAR sous CARLA 0.9.16 (Windows, Python 3.12) : collecte synchronisée,
étiquetage automatique, YOLO fine-tuné, estimation de distance et rejet de fausses alarmes par LiDAR.

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

## Code (src/fusion/)
`config.py` (paramètres), `geometry.py` (repères, projection), `rig.py` (capteurs CARLA, étiquettes).

## Reproduire
```
python -m venv venv ; venv\Scripts\activate
pip install -r requirements.txt   # + wheel CARLA, voir requirements.txt
python scripts\01_collect.py      # serveur CARLA lancé
```
Les données (data/) et les runs (runs/) ne sont pas versionnés ; les résultats de référence
sont dans reference/.

## Limites connues
Simulation uniquement ; le LiDAR CARLA n'est pas dégradé par pluie/brouillard ; modèles de
véhicules identiques entre train et test ; baseline caméra seule naïve (hauteur fixe 1,5 m).

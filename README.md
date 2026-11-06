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
  CARLA (≈ 4 m/s² à 70 km/h à frein 100 %), pas sur des valeurs homologuées. Jeu par défaut : alerte / partiel /
  urgence à 2,0 / 3,0 / 4,0 m/s² de décélération requise, frein partiel à 70 %. `AEBConfig.conservative()` garde
  le jeu précédent (1,5 / 2,5 / 3,5, 60 %), qui freine ~0,7 s plus tôt à 50 km/h.

## AEB : résultats mesurés (CARLA, 20 Hz, Town10HD_Opt, météo claire)
Banc de test maison : conducteur et régulateur de l'ego écrits pour l'essai, pas de trafic. 2 à 4 graines par cas.

| Scénario | Résultat |
|---|---|
| Cible arrêtée, 30 / 50 / 70 km/h (3 graines, écart de départ 50 m) | 0 collision ; écart minimal médian 9,9 / 7,6 / 3,3 m ; TTC au premier freinage 1,8 / 2,5 / 2,4 s |
| Idem avec le jeu conservateur | 0 collision ; 11,9 / 9,2 / 4,3 m ; TTC 2,1 / 3,2 / 2,4 s |
| Cible qui freine (1 s après le départ), 50 km/h, écart 20 m (2 graines) | 0 collision, écart minimal 8,6 m (sans anticipation de la décélération de la cible : 2/2 collisions, impact 23 km/h) |
| 70 km/h avec plus de place (écart de départ 60 m, 3 graines) | 0 collision, écart minimal médian 13,2 m |
| Cible hors couloir, suivi à même vitesse, approche normale | 0 freinage parasite observé (alerte seule en approche à 20 et 18 m) |

À 70 km/h avec 50 m de départ, la distance est déjà inférieure à la distance d'arrêt : le freinage part dès la
première détection quel que soit le seuil, donc cette ligne ne départage pas les jeux de seuils.

**Fréquence de perception** (`--perception-hz`, cible arrêtée, écart de départ 60 m, 2 graines, écart minimal médian) :
20 Hz : 7,6 m à 50 km/h pour 50 m de départ (13,2 m à 70 km/h pour 60 m) ; 10 Hz : 8,0 / 11,5 m (50 / 70 km/h) ;
5 Hz : 5,3 / 6,5 m. Sans correction de la latence supposée, à 5 Hz le même jeu provoquait une collision
(13 km/h) sur 1 des 2 essais à 50 km/h. L'AEB ajoute donc ≈ 3 périodes de calcul à la latence quand la perception est sous
20 Hz (maintien du frein + confirmation) ; c'est une hypothèse de modèle, dont les effets sont validés
sur la cible arrêtée seulement, pas sur les scénarios non dangereux à 5 et 10 Hz.

Les CSV figés sont dans `reference/` (`aeb_*_v3_*.csv`). Ils sont reproductibles avec
`python scripts\12_aeb_campaign.py --scenario stopped --speeds 30 50 70 --weathers clear --seeds 42 43 44 45`.

## Studio web (backend/)
```
python -m uvicorn app.main:app --app-dir backend --port 8000     # serveur CARLA lancé ; puis http://localhost:8000
```
Deux types de run : **observation** (trafic, YOLO + distances LiDAR) et **scénario AEB** en boucle fermée
(`scripts/11_aeb_closed_loop.py` lancé en sous-processus). Vue vidéo en direct (images à 10 Hz, rejouées avec
une petite mémoire tampon), état de l'AEB, décélération requise, écart ; à la fin, bilan et courbes
(`runs/studio/<run>/log.csv`). Options : AEB désactivé (témoin), fréquence de perception 20 / 10 / 5 Hz,
injection de défauts (fausses détections de durée réglable, pertes de mesure, bruit de distance).
Un seul run à la fois (une instance CARLA, un GPU). Tests de l'API sans CARLA : `python -m pytest tests`.

### Robustesse aux fausses détections
Test de contrainte (pas un taux réaliste) : à 50 km/h, route dégagée, une fausse détection dans le couloir (avec
mesure LiDAR plausible) démarre avec une probabilité de 0,10 par image, soit environ 8 événements par essai de 4 s.
Fréquence des essais avec au moins un freinage parasite, prédite par le modèle hors CARLA (`tests/test_faults.py`,
200 simulations) puis mesurée en boucle fermée dans CARLA (6 graines), avec confirmation de piste sur 2 mesures :

| Durée de la fausse détection | Modèle hors CARLA | CARLA (6 essais) |
|---|---|---|
| 1 image | 14 % | 1/6 |
| 2 images | 49 % | 3/6 |
| 3 images | 98,5 % | 6/6 |

Le modèle hors CARLA prévoit bien les mesures. La confirmation courte (2 mesures du tracker + 2 images de l'AEB)
est donc une faiblesse réelle. Modèle de défauts simple : les statistiques de persistance des fausses détections du
détecteur réel ne sont pas mesurées.

**Correction** : la piste n'est confirmée qu'après 0,15 s de persistance (`--confirm-s`, défaut 0,15 : 4 mesures à
20 Hz, 3 à 10 Hz, 2 à 5 Hz ; 0 = ancien comportement). En CARLA, mêmes conditions, 6 graines : **0/6 freinage** pour des
fausses détections de 1, 2 et 3 images (le modèle hors CARLA en prévoit 0 %, 17 % et 20 %, soit 0 à 1 essai sur 6 :
ce résultat est compatible avec lui, il ne le prouve pas). Une fausse détection de 4 images ou plus freine toujours
(100 % prédit, non mesuré en CARLA). Pas de régression sur la cible arrêtée à 20 Hz (mêmes écarts minimaux et mêmes
distances de premier freinage) ; le retard ajouté si l'obstacle apparaît à 20-25 m est d'environ 0,7 m à 50 km/h
(modèle hors CARLA).

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
- L'AEB est validé à 20 Hz. À 5 Hz (Worker d'observation web, sans AEB), la marge est mince à 70 km/h sur cible
  arrêtée : écart minimal ≈ 2,4 m en 16 lancements sans collision, mais 1 collision (impact 16 km/h) observée une fois
  sur la graine 42 dans un autre lot, non reproduite (cause non élucidée : variabilité du simulateur ou marge). Ne pas
  revendiquer 0 collision à 5 Hz et 70 km/h.
- Fausses détections : seul un modèle de défauts simple est testé (voir plus haut) ; pas de bruit capteur réaliste.
- Les freinages parasites ne sont mesurés que sur trois scénarios simples, pas sur du trafic réel.
- Cas corrigé : à 5 Hz, en passant à côté d'une cible arrêtée dans la voie voisine (50 km/h, graine 43), un freinage
  d'urgence parasite apparaissait dans 4 lancements sur 6. Cause trouvée en rejouant le journal : une fausse piste née
  de deux détections proches recevait une vitesse latérale initiale d'environ -5 m/s (a priori trop large, 8 m/s), et
  dérivait dans le couloir pendant 3 images. L'a priori latéral est ramené à 2 m/s (`vy_init_std`) ; test de
  non-régression `test_depassement_a_5hz_...`. En CARLA : 0 freinage sur 8 lancements de la graine 43, et 0/18 sur
  les graines 42 à 47 à 30 / 50 / 70 km/h. Mesure latérale par points LiDAR (`--lateral-source lidar`, défaut) :
  n'a pas résolu le problème à elle seule.

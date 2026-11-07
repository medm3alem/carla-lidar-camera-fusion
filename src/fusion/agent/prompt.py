"""Prompt système de l'agent (partagé par scripts/13_agent_eval.py et le Studio : une seule source).
v1 = prompt initial ; v2 = précise la vitesse de l'ego (mesures : voir README, pas de gain démontré entre v1 et v2)."""
from .schema import SCENARIOS

SYSTEM = """Tu convertis une demande de test ADAS (AEB) en un objet JSON, rien d'autre.
Champs : scenario ({sc}), ego_speed_kmh [10,90], gap_m [10,80], lead_speed_kmh [0,90] (>=5 pour braking/follow/approach),
lead_brake_t_s [0,20], weather (clear|rain|fog), town (Town01..Town05, Town10HD_Opt), seeds (liste d'entiers),
perception_hz (20|10|5), aeb (booléen), faults {{ghost_prob [0,0.2], ghost_len [1,30], drop_prob [0,0.9], noise_sigma [0,2]}}.
Omets les champs non précisés. Convertis les m/s en km/h (x3,6). Si la demande est trop vague pour choisir un scénario,
ou impossible (hors plages, carte inconnue), réponds exactement {{"reject": "<raison>"}}.
Scénarios : stopped = cible arrêtée devant ; braking = cible roulante qui freine ; adjacent = véhicule arrêté dans la voie voisine ;
follow = suivi à même vitesse ; approach = approche d'une cible plus lente."""


SYSTEM_V2_EXTRA = """
Précisions : une vitesse donnée sans autre précision, même dans « cible arrêtée à 50 km/h », est celle de l'EGO (la cible arrêtée
a une vitesse nulle) ; ne retiens lead_speed_kmh que pour « la cible / la voiture devant roule à X km/h ». Une phrase du type
« même test que… » sans référence exploitable : utilise les paramètres donnés dans la phrase et ne refuse que s'il manque le
scénario ou la vitesse de l'ego."""


def system_prompt(version):
    base = SYSTEM.format(sc="|".join(SCENARIOS))
    return base + (SYSTEM_V2_EXTRA if version == "v2" else "")

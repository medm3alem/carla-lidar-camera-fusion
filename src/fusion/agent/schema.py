"""
Schéma JSON d'un scénario d'essai AEB, validation et faisabilité (sans CARLA, sans LLM).
Un scénario écrit en langage naturel est converti en ce JSON, puis envoyé à l'API du Studio (/api/runs ou /api/campaigns).

Champs : scenario (obligatoire), ego_speed_kmh (obligatoire), gap_m, lead_speed_kmh, lead_brake_t_s, weather, town,
seeds, perception_hz, aeb, faults{ghost_prob, ghost_len, drop_prob, noise_sigma}.
Les champs absents prennent les valeurs par défaut validées en CARLA (DEFAULTS). Les plages sont celles de l'API.
"""
from copy import deepcopy

SCENARIOS = ("stopped", "braking", "adjacent", "follow", "approach")
WEATHERS = ("clear", "rain", "fog")
TOWNS = ("Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD_Opt")
# Plus long tronçon droit sans carrefour mesuré par scripts/11_aeb_closed_loop.py (m). Autres cartes : inconnu.
STRAIGHT_M = {"Town10HD_Opt": 95.0, "Town03": 80.0}

RANGES = {"ego_speed_kmh": (10, 90), "gap_m": (10, 80), "lead_speed_kmh": (0, 90), "lead_brake_t_s": (0, 20)}
FAULT_RANGES = {"ghost_prob": (0.0, 0.2), "ghost_len": (1, 30), "drop_prob": (0.0, 0.9), "noise_sigma": (0.0, 2.0)}
KEYS = {"scenario", "ego_speed_kmh", "gap_m", "lead_speed_kmh", "lead_brake_t_s", "weather", "town", "seeds",
        "perception_hz", "aeb", "faults"}

# Valeurs par défaut par scénario : celles validées en CARLA (Town10HD_Opt et Town03). lead_speed None = vitesse de l'ego.
DEFAULTS = {
    "stopped":  {"gap_m": 50.0, "lead_speed_kmh": 0.0, "lead_brake_t_s": 3.0},
    "braking":  {"gap_m": 45.0, "lead_speed_kmh": 30.0, "lead_brake_t_s": 1.0},
    "adjacent": {"gap_m": 50.0, "lead_speed_kmh": 0.0, "lead_brake_t_s": 3.0},
    "follow":   {"gap_m": 15.0, "lead_speed_kmh": None, "lead_brake_t_s": 3.0},
    "approach": {"gap_m": 25.0, "lead_speed_kmh": 30.0, "lead_brake_t_s": 3.0},
}
COMMON = {"weather": "clear", "town": "Town10HD_Opt", "seeds": [42], "perception_hz": 20, "aeb": True}
FAULT_DEFAULT = {"ghost_prob": 0.0, "ghost_len": 1, "drop_prob": 0.0, "noise_sigma": 0.0}


def normalize(d):
    """Remplit les valeurs par défaut. Ne valide pas : appeler validate() sur le résultat."""
    out = deepcopy(d)
    sc = out.get("scenario")
    for k, v in {**COMMON, **DEFAULTS.get(sc, {})}.items():
        out.setdefault(k, deepcopy(v))
    if sc == "follow" and out.get("lead_speed_kmh") is None and isinstance(out.get("ego_speed_kmh"), (int, float)):
        out["lead_speed_kmh"] = out["ego_speed_kmh"]
    f = out.get("faults")
    out["faults"] = {**FAULT_DEFAULT, **(f if isinstance(f, dict) else {})}
    return out


def need_length_m(sc):
    """Route droite nécessaire (m), mêmes formules que scripts/11_aeb_closed_loop.py (follow_seconds = 4 s)."""
    gap, vl, bt = sc["gap_m"], (sc.get("lead_speed_kmh") or 0.0) / 3.6, sc["lead_brake_t_s"]
    kind = sc["scenario"]
    if kind == "braking":
        return gap + 5.0 + vl * bt + vl * vl / 8.0 + 10.0
    if kind in ("follow", "approach"):
        return gap + 5.0 + vl * 4.0 + 15.0
    return gap + 30.0                                     # stopped, adjacent


def _isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate(d):
    """Retourne la liste des erreurs de format/plage (vide = valide) sur un dict normalisé."""
    errs = []
    if not isinstance(d, dict):
        return ["pas un objet JSON"]
    for k in d:
        if k not in KEYS:
            errs.append(f"champ inconnu : {k}")
    if d.get("scenario") not in SCENARIOS:
        errs.append(f"scenario invalide : {d.get('scenario')!r}")
    for k, (lo, hi) in RANGES.items():
        v = d.get(k)
        if v is None and k == "lead_speed_kmh":
            continue
        if k not in d:
            errs.append(f"champ manquant : {k}")
        elif not _isnum(v) or not lo <= v <= hi:
            errs.append(f"{k} hors plage [{lo}, {hi}] : {v!r}")
    if d.get("scenario") in ("braking", "approach", "follow") and _isnum(d.get("lead_speed_kmh")) \
            and d["lead_speed_kmh"] < 5:
        errs.append(f"lead_speed_kmh doit être >= 5 pour {d['scenario']}")
    if d.get("weather") not in WEATHERS:
        errs.append(f"weather invalide : {d.get('weather')!r}")
    if d.get("town") not in TOWNS:
        errs.append(f"town invalide : {d.get('town')!r}")
    if d.get("perception_hz") not in (20, 10, 5):
        errs.append(f"perception_hz invalide : {d.get('perception_hz')!r}")
    if not isinstance(d.get("aeb"), bool):
        errs.append("aeb doit être un booléen")
    s = d.get("seeds")
    if not (isinstance(s, list) and s and all(isinstance(x, int) and not isinstance(x, bool) for x in s)):
        errs.append("seeds doit être une liste d'entiers non vide")
    f = d.get("faults", {})
    for k, v in f.items():
        if k not in FAULT_RANGES:
            errs.append(f"faults : champ inconnu : {k}")
        elif not _isnum(v) or not FAULT_RANGES[k][0] <= v <= FAULT_RANGES[k][1]:
            errs.append(f"faults.{k} hors plage {FAULT_RANGES[k]} : {v!r}")
    return errs


def feasibility(d):
    """Erreurs de faisabilité physique/géométrique d'un scénario valide (vide = réalisable)."""
    errs = []
    need = need_length_m(d)
    cap = STRAIGHT_M.get(d["town"])
    if cap is not None and need > cap:
        errs.append(f"route droite nécessaire {need:.0f} m > {cap:.0f} m disponibles sur {d['town']}")
    return errs


def check(d):
    """normalize + validate + feasibility. Retourne (scénario normalisé, erreurs de format, erreurs de faisabilité)."""
    n = normalize(d) if isinstance(d, dict) else d
    fmt = validate(n)
    return n, fmt, (feasibility(n) if not fmt else [])


def to_studio_payload(n, many=None):
    """Convertit un scénario normalisé en corps de requête pour l'API du Studio.
    Un seul essai (une seule graine) -> /api/runs ; sinon -> /api/campaigns (vitesse unique, graines multiples)."""
    f = n["faults"]
    if many is None:
        many = len(n["seeds"]) > 1
    common = {"scenario": n["scenario"], "gap": n["gap_m"], "perception_hz": n["perception_hz"],
              "ghost_prob": f["ghost_prob"], "ghost_len": f["ghost_len"], "drop_prob": f["drop_prob"],
              "noise_sigma": f["noise_sigma"], "lead_brake_t": n["lead_brake_t_s"]}
    if many:
        return "/api/campaigns", {**common, "speeds": [n["ego_speed_kmh"]], "weathers": [n["weather"]],
                                  "seeds": n["seeds"], "lead_speed": max(5.0, n["lead_speed_kmh"] or 30.0),
                                  "same_speed": n["scenario"] == "follow", "skip_baseline": n["aeb"]}
    return "/api/runs", {**common, "kind": "aeb", "town": n["town"], "weather": n["weather"], "seed": n["seeds"][0],
                         "v0": n["ego_speed_kmh"], "lead_speed": max(5.0, n["lead_speed_kmh"] or 5.0), "aeb": n["aeb"]}

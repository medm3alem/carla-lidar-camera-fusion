"""Baseline sans LLM : extraction par mots-clés et expressions régulières (français et anglais).
Sert de référence à battre pour les approches LLM, et à valider le banc d'évaluation. Volontairement simple."""
import re

_NUM = r"(\d+(?:[.,]\d+)?)"


def _f(x):
    return float(x.replace(",", "."))


def parse_text(text):
    t = text.lower()
    d = {}
    if re.search(r"arrêt|arret|immobile|stationn|à l'arrêt|stopped|stationary|parked", t):
        d["scenario"] = "stopped"
    if re.search(r"frein|brak", t) and not re.search(r"sans frein", t):
        d["scenario"] = "braking" if re.search(r"(cible|voiture|véhicule|vehicule|lead|car|target).{0,40}(frein|brak)|(frein|brak).{0,40}(cible|devant|lead)", t) else d.get("scenario", "braking")
    if re.search(r"voisine|adjacent|côté|cote|doublé|double|dépass|depass|overtak|next lane|neighbo", t):
        d["scenario"] = "adjacent"
    if re.search(r"suit|suivi|following|follow", t):
        d["scenario"] = "follow"
    if re.search(r"approche|approach|plus lent|slower|rattrap", t):
        d["scenario"] = "approach"
    m = re.search(_NUM + r"\s*km/?h", t)
    if m:
        d["ego_speed_kmh"] = _f(m.group(1))
    m = re.search(_NUM + r"\s*m/s", t)
    if m and "ego_speed_kmh" not in d:
        d["ego_speed_kmh"] = round(_f(m.group(1)) * 3.6, 1)
    kms = re.findall(_NUM + r"\s*km/?h", t)
    if len(kms) >= 2 and d.get("scenario") in ("braking", "follow", "approach"):
        d["lead_speed_kmh"] = _f(kms[1])
    m = re.search(r"(?:écart|ecart|distance|gap|à|a|at)\s*(?:de|of)?\s*" + _NUM + r"\s*(?:m(?![/\w])|mètres|metres|meters)", t)
    if m:
        d["gap_m"] = _f(m.group(1))
    m = re.search(_NUM + r"\s*hz", t)
    if m:
        d["perception_hz"] = int(_f(m.group(1)))
    if re.search(r"pluie|rain", t):
        d["weather"] = "rain"
    elif re.search(r"brouillard|fog", t):
        d["weather"] = "fog"
    elif re.search(r"clair|clear|soleil|sunny", t):
        d["weather"] = "clear"
    m = re.search(r"town\s*0?(\d+)", t)
    if m:
        d["town"] = "Town10HD_Opt" if m.group(1) == "10" else f"Town{int(m.group(1)):02d}"
    seeds = re.search(r"grain(?:e|es)?\s*([\d\s,et]+)|seeds?\s*([\d\s,and]+)", t)
    if seeds:
        nums = [int(x) for x in re.findall(r"\d+", seeds.group(1) or seeds.group(2))]
        if nums:
            d["seeds"] = nums
    if re.search(r"sans aeb|aeb (désactivé|desactive|off)|without aeb|aeb disabled|témoin|temoin", t):
        d["aeb"] = False
    faults = {}
    m = re.search(r"fausses? détections?.{0,30}?" + _NUM + r"\s*%|ghost.{0,30}?" + _NUM + r"\s*%", t)
    if m:
        faults["ghost_prob"] = _f(m.group(1) or m.group(2)) / 100.0
    m = re.search(r"pertes? de mesure.{0,20}?" + _NUM + r"\s*%|dropout.{0,20}?" + _NUM + r"\s*%", t)
    if m:
        faults["drop_prob"] = _f(m.group(1) or m.group(2)) / 100.0
    m = re.search(r"bruit.{0,30}?" + _NUM + r"\s*m\b|noise.{0,30}?" + _NUM + r"\s*m\b", t)
    if m:
        faults["noise_sigma"] = _f(m.group(1) or m.group(2))
    if faults:
        d["faults"] = faults
    return d if "scenario" in d else None

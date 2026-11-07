"""Évaluation d'un convertisseur langage naturel -> scénario JSON, et regroupement des erreurs (sans LLM)."""
import json
import re
from collections import Counter

from .schema import check

FIELDS = ["scenario", "ego_speed_kmh", "gap_m", "lead_speed_kmh", "lead_brake_t_s", "weather", "town", "seeds",
          "perception_hz", "aeb", "ghost_prob", "ghost_len", "drop_prob", "noise_sigma"]


def flat(n):
    f = n.get("faults", {})
    return {**{k: n.get(k) for k in FIELDS[:10]}, **{k: f.get(k) for k in FIELDS[10:]}}


def _eq(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) <= 0.5 if max(abs(a), abs(b)) >= 5 else abs(a - b) <= 1e-6
    return a == b


def parse(output):
    """Sortie d'un modèle (dict, ou texte contenant un objet JSON) -> (dict ou None, 'ok' | 'json_invalide' | 'rejet')."""
    if output is None:
        return None, "rejet"
    if isinstance(output, dict):
        return (None, "rejet") if output.get("reject") else (output, "ok")
    txt = re.sub(r"<think>.*?</think>", "", str(output), flags=re.S).strip()
    if txt.startswith("```"):
        txt = txt.strip("`").lstrip("json").strip()
    i, j = txt.find("{"), txt.rfind("}")
    try:
        d = json.loads(txt[i:j + 1])
    except ValueError:
        return None, "json_invalide"
    return (None, "rejet") if isinstance(d, dict) and d.get("reject") else (d, "ok")


def classify(item, output):
    """Retourne (correct: bool, causes: list[str]). Une cause = une catégorie d'erreur, pour le regroupement."""
    d, status = parse(output)
    gold = item.get("gold")
    if gold is None:                                       # le bon comportement est de refuser
        if status == "rejet":
            return True, []
        if status == "json_invalide":
            return False, ["json_invalide"]
        reason = item.get("reject_reason", "infaisable")
        if reason == "ambigu":                             # demande trop vague : seul un refus explicite est correct
            return False, ["ambigu_accepté"]
        n, fmt, infeasible = check(d)                      # infaisable : refus explicite, ou sortie que le validateur écarte
        return (True, []) if (fmt or infeasible) else (False, ["infaisable_accepté"])
    if status == "json_invalide":
        return False, ["json_invalide"]
    if status == "rejet":
        return False, ["rejet_à_tort"]
    n, fmt, infeasible = check(d)
    causes = []
    if fmt:
        for e in fmt:
            causes.append("champ_inconnu" if e.startswith("champ inconnu") or "faults : champ inconnu" in e else
                          "champ_manquant" if e.startswith("champ manquant") else
                          "valeur_invalide" if "invalide" in e else "hors_plage")
        return False, sorted(set(causes))
    g, p = flat(check(gold)[0]), flat(n)
    if g["scenario"] != p["scenario"]:                     # un mauvais scénario change aussi les valeurs par défaut :
        return False, ["mauvais_scénario"]                  # une seule cause racine, une seule catégorie
    for k in FIELDS:
        if not _eq(g[k], p[k]):
            if k in ("ego_speed_kmh", "lead_speed_kmh") and isinstance(g[k], (int, float)) and isinstance(p[k], (int, float)) \
                    and g[k] and any(abs(p[k] - g[k] * r) < 0.6 for r in (3.6, 1 / 3.6)):
                causes.append("erreur_unité")
            else:
                causes.append(f"mauvaise_valeur:{k}")
    return (not causes), sorted(set(causes))


def evaluate(items, model):
    """model : callable texte -> sortie (dict / str / None). Retourne un rapport."""
    rows, field_ok, field_n, clusters = [], Counter(), Counter(), Counter()
    by_cat = {}
    for it in items:
        out = model(it["text"])
        ok, causes = classify(it, out)
        rows.append({"id": it["id"], "category": it["category"], "ok": ok, "causes": causes})
        by_cat.setdefault(it["category"], []).append(ok)
        for c in causes:
            clusters[c] += 1
        if it.get("gold") is not None:
            d, st = parse(out)
            if st == "ok":
                n, fmt, _ = check(d)
                if not fmt:
                    g, p = flat(check(it["gold"])[0]), flat(n)
                    for k in FIELDS:
                        field_n[k] += 1
                        field_ok[k] += _eq(g[k], p[k])
    n_ok = sum(r["ok"] for r in rows)
    return {"n": len(rows), "correct": n_ok, "accuracy": n_ok / max(len(rows), 1),
            "by_category": {c: (sum(v), len(v)) for c, v in by_cat.items()},
            "field_accuracy": {k: (field_ok[k], field_n[k]) for k in FIELDS if field_n[k]},
            "clusters": dict(clusters.most_common()), "rows": rows}

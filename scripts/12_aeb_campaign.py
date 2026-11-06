r"""
Campagne AEB en boucle fermée : matrice vitesses x météos x (AEB actif / témoin), scénario 'stopped'.
Lance scripts\11_aeb_closed_loop.py pour chaque cas (un processus par essai, CARLA nettoyé entre deux),
agrège les summary.json dans runs\aeb_campaign\<id>\results.csv et affiche un tableau.
Un essai en erreur est noté et la campagne continue. Un cas déjà fait (results.csv présent) n'est pas relancé
si on relance avec --resume <id>.

Usage (CARLA lancé, venv actif) :
  python scripts\12_aeb_campaign.py --speeds 30 50 70 --weathers clear rain fog --seeds 42
  python scripts\12_aeb_campaign.py --resume 20261106-170000
Durée : ~35 s par essai (6 cas par vitesse et par graine pour 3 météos x 2 modes).
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["case", "scenario", "v0", "gap", "weather", "seed", "aeb", "collided", "impact_kmh",
          "min_gap_m", "gap_start_m", "t_warn", "t_brake", "final_speed_kmh", "n_brake_ticks", "max_state", "error"]


def run_case(py, args, v0, weather, seed, aeb, out_dir):
    cmd = [py, str(ROOT / "scripts" / "11_aeb_closed_loop.py"), "--scenario", args.scenario,
           "--lead-speed", str(args.lead_speed),
           "--v0", str(v0), "--gap", str(args.gap), "--weather", weather, "--seed", str(seed),
           "--town", args.town, "--out", str(out_dir)]
    if not aeb:
        cmd.append("--no-aeb")
    before = set(out_dir.glob("*/summary.json"))
    subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    new = sorted(set(out_dir.glob("*/summary.json")) - before)
    if not new:
        return None
    return json.loads(new[-1].read_text())


def to_row(case, args, v0, weather, seed, aeb, s):
    if s is None:
        return {"case": case, "scenario": args.scenario, "v0": v0, "gap": args.gap, "weather": weather,
                "seed": seed, "aeb": aeb, "error": "pas de summary.json"}
    return {"case": case, "scenario": args.scenario, "v0": v0, "gap": args.gap, "weather": weather,
            "seed": seed, "aeb": aeb, "collided": s.get("collided"), "impact_kmh": s.get("impact_speed_kmh"),
            "min_gap_m": s.get("min_gap_m"), "gap_start_m": s.get("gap_start_m"),
            "t_warn": s.get("t_first_warning_s"), "t_brake": s.get("t_first_brake_s"),
            "final_speed_kmh": s.get("ego_final_speed_kmh"), "n_brake_ticks": s.get("n_brake_ticks"),
            "max_state": s.get("max_state"), "error": s.get("error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", type=float, nargs="+", default=[30, 50, 70])
    ap.add_argument("--weathers", nargs="+", default=["clear", "rain", "fog"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--gap", type=float, default=50.0)
    ap.add_argument("--scenario", choices=["stopped", "braking", "adjacent", "follow"], default="stopped")
    ap.add_argument("--lead-speed", type=float, default=30.0, help="km/h (braking, follow)")
    ap.add_argument("--town", default="Town10HD_Opt")
    ap.add_argument("--resume", default=None, help="id d'une campagne à reprendre")
    args = ap.parse_args()

    cid = args.resume or time.strftime("%Y%m%d-%H%M%S")
    camp = ROOT / "runs" / "aeb_campaign" / cid
    camp.mkdir(parents=True, exist_ok=True)
    res_path = camp / "results.csv"
    rows = []
    if res_path.exists():
        with open(res_path, newline="") as f:
            rows = list(csv.DictReader(f))
    done = {r["case"] for r in rows}

    cases = [(v0, w, sd, aeb) for sd in args.seeds for v0 in args.speeds for w in args.weathers
             for aeb in (False, True)]
    print(f"campagne {cid} : {len(cases)} essais, {len(done)} déjà faits")
    for i, (v0, w, sd, aeb) in enumerate(cases, 1):
        case = f"v{int(v0)}_{w}_s{sd}_{'aeb' if aeb else 'noaeb'}"
        if case in done:
            continue
        t0 = time.time()
        s = run_case(sys.executable, args, v0, w, sd, aeb, camp / "runs")
        row = to_row(case, args, v0, w, sd, aeb, s)
        rows.append(row)
        with open(res_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
        print(f"[{i}/{len(cases)}] {case:<28} collision={row.get('collided')} "
              f"impact={row.get('impact_kmh')} gap_min={row.get('min_gap_m')} "
              f"err={row.get('error')} ({time.time() - t0:.0f} s)")

    agg = defaultdict(list)
    for r in rows:
        if r.get("error") in ("", None):
            agg[(float(r["v0"]), str(r["aeb"]) in ("True", "true"))].append(r)
    if args.scenario in ("adjacent", "follow"):
        print(f"\nScénario non dangereux '{args.scenario}' : tout freinage est une fausse activation")
        print(f"{'vitesse':>8}{'mode':>10}{'essais':>8}{'avec freinage':>15}{'alertes (état>=WARNING)':>26}")
        g2 = defaultdict(list)
        for r in rows:
            if r.get("error") in ("", None):
                g2[(float(r["v0"]), str(r["aeb"]) in ("True", "true"))].append(r)
        for (v0, aeb), rs in sorted(g2.items()):
            if not aeb:
                continue
            nb = sum(1 for r in rs if float(r.get("n_brake_ticks") or 0) > 0)
            nw = sum(1 for r in rs if r.get("max_state") not in ("NORMAL", "", None))
            print(f"{v0:>6.0f} km/h{'AEB':>10}{len(rs):>8}{nb:>9}/{len(rs):<5}{nw:>20}/{len(rs)}")
        print(f"\nrésultats : {res_path}")
        return
    print(f"\n{'vitesse':>8}{'mode':>10}{'essais':>8}{'collisions':>12}{'impact moy.':>13}{'écart min méd.':>16}")
    for (v0, aeb), rs in sorted(agg.items()):
        col = [str(r["collided"]) in ("True", "true") for r in rs]
        imp = [float(r["impact_kmh"]) for r, c in zip(rs, col) if c]
        gaps = sorted(float(r["min_gap_m"]) for r in rs)
        print(f"{v0:>6.0f} km/h{'AEB' if aeb else 'témoin':>10}{len(rs):>8}{sum(col):>6}/{len(rs):<5}"
              f"{(sum(imp) / len(imp) if imp else 0):>11.1f}{gaps[len(gaps) // 2]:>16.1f}")
    n_err = sum(1 for r in rows if r.get("error") not in ("", None))
    print(f"\nrésultats : {res_path}" + (f"   ({n_err} essais en erreur)" if n_err else ""))


if __name__ == "__main__":
    main()

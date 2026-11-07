r"""
Évalue un convertisseur langage naturel -> scénario JSON sur reference/agent_benchmark.jsonl (sans CARLA).
Modèles : rules (baseline sans LLM), oracle (renvoie la réponse de référence : vérifie le banc), anthropic (LLM).
Usage : python scripts\13_agent_eval.py --model rules
        python scripts\13_agent_eval.py --model anthropic --save-raw runs\agent_fable.jsonl --out runs\agent_fable.json
Prérequis LLM : pip install anthropic ; $env:ANTHROPIC_API_KEY="..." (clé de la Console API, distincte de l'abonnement claude.ai).
Modèles : claude-fable-5-1 (le plus puissant disponible, défaut), claude-opus-5-5, claude-sonnet-5-5, claude-haiku-5-5.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fusion.agent.prompt import system_prompt  # noqa: E402
from fusion.agent.metrics import evaluate  # noqa: E402
from fusion.agent.rules import parse_text  # noqa: E402

def anthropic_model(name, version="v2"):
    import anthropic                                         # noqa: PLC0415
    client = anthropic.Anthropic()
    system = system_prompt(version)

    def call(text):
        kw = dict(model=name, max_tokens=600, system=system, messages=[{"role": "user", "content": text}])
        try:
            r = client.messages.create(temperature=0, **kw)
        except anthropic.BadRequestError as e:           # certains modèles n'acceptent pas le paramètre temperature
            if "temperature" not in str(e).lower():
                raise
            r = client.messages.create(**kw)
        return next((b.text for b in r.content if b.type == "text"), "")
    return call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rules", "oracle", "anthropic", "openai"], default="rules")
    ap.add_argument("--llm", default="claude-fable-5-1", help="nom du modèle (obligatoire avec --model openai)")
    ap.add_argument("--provider", choices=["groq", "openrouter", "ollama", "gemini"], default="groq", help="--model openai")
    ap.add_argument("--base-url", default=None, help="URL de base compatible OpenAI (remplace --provider)")
    ap.add_argument("--key-env", default="LLM_API_KEY", help="variable d'environnement qui contient la clé")
    ap.add_argument("--sleep", type=float, default=0.0, help="pause (s) entre deux appels, pour les limites de tokens/min")
    ap.add_argument("--prompt", choices=["v1", "v2"], default="v2",
                    help="v1 = prompt initial (33/35 sur Groq gpt-oss-120b) ; v2 = précise la vitesse de l'ego (voir SYSTEM_V2_EXTRA)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="N passages complets : moyenne, min-max, et erreurs stables (à chaque passage) ou intermittentes")
    ap.add_argument("--save-raw", default=None, help="écrit chaque réponse brute du modèle (jsonl) pour l'analyse d'erreurs")
    ap.add_argument("--bench", default=str(ROOT / "reference" / "agent_benchmark.jsonl"))
    ap.add_argument("--out", default=None, help="écrit le rapport JSON ici")
    args = ap.parse_args()
    items = [json.loads(line) for line in Path(args.bench).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.model == "openai" and args.llm == "claude-fable-5-1":
        sys.exit("--llm est obligatoire avec --model openai (ex. openai/gpt-oss-120b pour Groq)")
    if args.model == "rules":
        model = parse_text
    elif args.model == "oracle":
        gold = {it["text"]: it["gold"] for it in items}
        model = lambda t: gold[t]                            # noqa: E731
    elif args.model == "openai":
        from fusion.agent.llm import BASES, key_from_env, make_chat     # noqa: PLC0415
        key = key_from_env(args.key_env)
        if key is None and args.provider != "ollama" and not args.base_url:
            sys.exit(f"clé absente : définir ${args.key_env} (ex. $env:{args.key_env}=\"...\")")
        model = make_chat(args.base_url or BASES[args.provider], args.llm, key,
                          system_prompt(args.prompt), sleep=args.sleep)
    else:
        model = anthropic_model(args.llm, args.prompt)
    raw, reps = [], []
    for run in range(max(args.repeat, 1)):
        def recorded(text, run=run):
            out = model(text)
            raw.append({"run": run + 1, "text": text, "raw": out})
            return out
        reps.append(evaluate(items, recorded))
        if args.repeat > 1:
            print(f"passage {run + 1}/{args.repeat} : {reps[-1]['correct']}/{reps[-1]['n']}", flush=True)
    rep = reps[0]
    if args.save_raw:
        ids = {it["text"]: it["id"] for it in items}
        Path(args.save_raw).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_raw).write_text("\n".join(json.dumps({**r_, "id": ids[r_["text"]]}, ensure_ascii=False, default=str)
                                                for r_ in raw) + "\n", encoding="utf-8")
    if args.repeat > 1:
        accs = [r["correct"] for r in reps]
        fails = {}
        for r in reps:
            for row in r["rows"]:
                if not row["ok"]:
                    fails.setdefault(row["id"], []).append("/".join(row["causes"]))
        print(f"\n{args.repeat} passages : moyenne {sum(accs) / len(accs):.1f}/{rep['n']}, min {min(accs)}, max {max(accs)}")
        stable = {k: v for k, v in fails.items() if len(v) == len(reps)}
        flaky = {k: v for k, v in fails.items() if len(v) < len(reps)}
        print("erreurs stables (à chaque passage) :", ", ".join(f"{k}[{v[0]}]" for k, v in sorted(stable.items())) or "aucune")
        print("erreurs intermittentes :", ", ".join(f"{k} ({len(v)}/{len(reps)}) [{v[0]}]" for k, v in sorted(flaky.items())) or "aucune")
        if args.out:
            Path(args.out).write_text(json.dumps({"repeat": args.repeat, "correct": accs, "stable": stable, "flaky": flaky},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
        return
    print(f"modèle : {args.model}  ({rep['n']} cas)  exact : {rep['correct']}/{rep['n']} = {100 * rep['accuracy']:.0f} %")
    print("\npar catégorie :")
    for c, (k, n) in rep["by_category"].items():
        print(f"  {c:<12}{k:>3}/{n}")
    print("\nprécision par champ (sorties valides, cas réalisables) :")
    for f, (k, n) in rep["field_accuracy"].items():
        print(f"  {f:<16}{k:>3}/{n}")
    print("\nregroupement des erreurs :")
    for c, n in rep["clusters"].items():
        print(f"  {c:<28}{n:>3}")
    print("\ncas en échec :", ", ".join(f"{r['id']}[{'/'.join(r['causes'])}]" for r in rep["rows"] if not r["ok"]) or "aucun")
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

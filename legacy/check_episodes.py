import json, glob, math, collections as C

S = C.defaultdict(lambda: {"n": 0, "jumps": 0, "len": 0.0, "xs": [], "ys": []})
prev = None
for p in sorted(glob.glob("dataset_full_v3/frame_*/annotations.json")):
    a = json.load(open(p))
    w = a["weather"]
    x, y, _ = a["ego"]["location"]
    s = S[w]
    s["n"] += 1; s["xs"].append(x); s["ys"].append(y)
    if prev and prev[0] == w:
        d = math.hypot(x - prev[1], y - prev[2])
        if d > 5: s["jumps"] += 1
        else: s["len"] += d
    prev = (w, x, y)

for w, s in S.items():
    print(f"{w}: {s['n']} frames, {s['jumps']} respawns, trajet {s['len']:.0f} m, "
          f"x[{min(s['xs']):.0f},{max(s['xs']):.0f}] y[{min(s['ys']):.0f},{max(s['ys']):.0f}]")
r"""
Fusion caméra + LiDAR v1 : profondeur des détections et rejet de fausses alarmes.
Pour chaque détection YOLO (vraie ou fausse), on cherche les points LiDAR qui tombent dans la zone
centrale de la boîte (sol retiré) et on garde le premier groupe de points le long de la profondeur.
  A. erreur de profondeur des détections justes (référence : face avant du véhicule), face à une
     estimation caméra seule (hauteur de boîte, véhicule supposé de 1,5 m)
  B. la preuve LiDAR sépare-t-elle vraies détections et fausses alarmes ?
Usage : python scripts\09_fusion_depth.py --data data\v4_town03 --tag town03
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fusion.config import CFG
from fusion import geometry as geo
from fusion.perception import (box_iou, boxes_of, fuse_box, ground_depth, mono_depth, project_lidar,
                               GROUND_MARGIN, INNER, MONO_HEIGHT)

BINS = [(0, 20), (20, 40), (40, 60)]
EST_BINS = BINS + [(60, 1e9)]
COCO_VEHICLES = [2, 5, 7]
THRESHOLDS = [0, 3, 5, 10, 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs" / "train" / "v4_n" / "weights" / "best.pt"))
    ap.add_argument("--data", default=str(ROOT / "data" / "v4"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--coco", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--min-pred-h", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tag", default="fusion")
    args = ap.parse_args()

    data = Path(args.data)
    K = geo.intrinsics(CFG.img_w, CFG.img_h, CFG.fov)
    entries = []
    for lab_p in sorted(data.glob("*/ep*/labels.json")):
        lab = json.loads(lab_p.read_text())
        if lab["split"] != args.split:
            continue
        for fr in lab["frames"]:
            k = fr["idx"]
            entries.append((lab["weather"], lab["episode"], k, lab_p.parent / f"{k:03d}_rgb.png",
                            lab_p.parent / f"{k:03d}_lidar.npy", fr))
    entries = entries[::args.stride]
    print(f"{len(entries)} frames ({args.split}), poids : {args.weights}")

    model = YOLO(args.weights)
    device = 0 if torch.cuda.is_available() else "cpu"
    paths = [str(e[3]) for e in entries]

    def predict_chunks():
        for i in range(0, len(paths), args.batch):
            yield from model.predict(paths[i:i + args.batch], imgsz=args.imgsz, conf=args.conf,
                                     classes=COCO_VEHICLES if args.coco else [0],
                                     device=device, verbose=False)

    rows, n_gt, n_frames = [], 0, 0
    for e, r in zip(entries, predict_chunks()):
        weather, ep, idx, _, lidar_path, fr = e
        objs = fr["objects"]
        gts = [o for o in objs if not o["ignore"]]
        n_gt += len(gts)
        n_frames += 1
        G, IG = boxes_of(gts), boxes_of([o for o in objs if o["ignore"]])
        P = r.boxes.xyxy.cpu().numpy()
        C = r.boxes.conf.cpu().numpy()

        pts = np.load(lidar_path)
        T_wc, T_wl = np.array(fr["T_world_cam"]), np.array(fr["T_world_lidar"])
        uv, depth = project_lidar(pts, T_wl, T_wc, K, CFG.img_w, CFG.img_h, CFG.lidar_pos[2], GROUND_MARGIN)

        used = np.zeros(len(gts), bool)
        iou_g = box_iou(P, G)[0] if len(P) and len(G) else np.zeros((len(P), len(G)))
        ioa_i = box_iou(P, IG)[1] if len(P) and len(IG) else np.zeros((len(P), len(IG)))
        for pi in np.argsort(-C):
            free = np.flatnonzero(~used)
            gt = None
            if len(free) and iou_g[pi, free].max() >= args.iou:
                gt = free[iou_g[pi, free].argmax()]
                used[gt] = True
                status = "tp"
            elif (P[pi, 3] - P[pi, 1]) < args.min_pred_h or \
                    (ioa_i.shape[1] and ioa_i[pi].max() >= 0.5):
                continue                                   # ignorée : ni succès ni fausse alarme
            else:
                status = "fp"
            box = P[pi]
            d_lidar, n_clu = fuse_box(uv, depth, box)
            mono = mono_depth(K, box, MONO_HEIGHT)
            rows.append({
                "weather": weather, "episode": ep, "idx": idx, "status": status, "conf": float(C[pi]),
                "gt_dist": gts[gt]["distance"] if gt is not None else None,
                "gt_face": gts[gt]["depth_face"] if gt is not None else None,
                "lidar_depth": d_lidar, "n_cluster": n_clu, "mono_depth": mono,
                "ground_depth": ground_depth(K, box, CFG.cam_pos[2]),
            })

    tps = [x for x in rows if x["status"] == "tp"]
    fps = [x for x in rows if x["status"] == "fp"]

    def line(err):
        e = np.abs(np.asarray(err))
        return f"{np.median(err):+6.2f} {e.mean():6.2f} {100 * (e < 1).mean():5.0f} %"

    print(f"\n=== A. Profondeur des détections justes ({len(tps)}), référence : face avant ===")
    print(f"{'distance':<10}{'n':>6}{'couverture':>12}   {'LiDAR : méd. MAE <1 m':<28}  {'hauteur fixe 1,5 m':<28}  plan du sol")
    for lo, hi in BINS + [(0, 60)]:
        sel = [x for x in tps if lo <= x["gt_dist"] < hi]
        wl = [x for x in sel if x["lidar_depth"] is not None]
        if not sel:
            continue
        lid = line([x["lidar_depth"] - x["gt_face"] for x in wl]) if wl else "-"
        mono = line([x["mono_depth"] - x["gt_face"] for x in sel])
        gl = [x for x in sel if x["ground_depth"] is not None]
        grd = line([x["ground_depth"] - x["gt_face"] for x in gl]) if gl else "-"
        print(f"{f'{lo}-{hi} m':<10}{len(sel):>6}{100 * len(wl) / len(sel):>10.0f} %   {lid:<28}  {mono:<28}  {grd}")

    print(f"\n=== B. Rejet de détections sans preuve LiDAR (vraies : {len(tps)}, fausses : {len(fps)}) ===")
    print(f"{'seuil':>6}{'vraies gardées':>16}{'fausses gardées':>17}{'précision':>11}{'recall':>9}{'FP/frame':>10}")
    for t in THRESHOLDS:
        tk = sum(x["n_cluster"] >= t for x in tps)
        fk = sum(x["n_cluster"] >= t for x in fps)
        print(f"{'>=' + str(t):>6}{tk:>16}{fk:>17}{100 * tk / max(tk + fk, 1):>10.1f}%"
              f"{100 * tk / max(n_gt, 1):>8.1f}%{fk / max(n_frames, 1):>10.2f}")

    print("\n=== C. Même rejet (au moins 3 points), par distance estimée ===")
    print(f"{'distance':<10}{'vraies':>8}{'avec preuve':>13}{'fausses':>10}{'avec preuve':>13}")
    for lo, hi in EST_BINS:
        def est(x):
            return x["lidar_depth"] if x["lidar_depth"] is not None else x["mono_depth"]
        t = [x for x in tps if lo <= est(x) < hi]
        f = [x for x in fps if lo <= est(x) < hi]
        print(f"{f'{lo}-{hi:.0f} m' if hi < 1e8 else f'>= {lo} m':<10}{len(t):>8}"
              f"{100 * sum(x['n_cluster'] >= 3 for x in t) / max(len(t), 1):>11.0f} %"
              f"{len(f):>10}{100 * sum(x['n_cluster'] >= 3 for x in f) / max(len(f), 1):>11.0f} %")

    out = ROOT / "runs" / f"fusion_{args.tag}_{args.split}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nDétail par détection : {out}")


if __name__ == "__main__":
    main()
r"""
Évalue un détecteur sur le dataset v4 (détection 2D seule, avant fusion).
GT = véhicules non ignorés. Une détection qui tombe sur un véhicule "ignoré" (trop caché) ou qui est
minuscule (véhicule au-delà de la portée étiquetée) n'est ni un succès ni une fausse alarme.
Usage :
  python scripts\05_eval_yolo.py --weights yolo26n.pt --coco               # COCO préentraîné
  python scripts\05_eval_yolo.py --weights <best.pt>                       # modèle fine-tuné (classe 0)
  python scripts\05_eval_yolo.py --weights yolo26n.pt --coco --dump-fp 12  # + images de fausses alarmes
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
BINS = [(0, 20), (20, 40), (40, 60)]
COCO_VEHICLES = [2, 5, 7]       # car, bus, truck


def box_iou(a, b):
    """IoU (N,M) et IoA = intersection / aire de a (N,M)."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)
    ioa = inter / np.maximum(area_a[:, None], 1e-9)
    return iou, ioa


def boxes_of(objs):
    return np.array([[o["box2d"][0], o["box2d"][1], o["box2d"][2] + 1, o["box2d"][3] + 1]
                     for o in objs], float).reshape(-1, 4)


def draw(img, boxes, color, label=None):
    for i, b in enumerate(boxes):
        p0, p1 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
        cv2.rectangle(img, p0, p1, color, 1)
        if label is not None:
            cv2.putText(img, label[i], (p0[0], max(p0[1] - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--data", default=str(ROOT / "data" / "v4"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--coco", action="store_true", help="classes COCO car/bus/truck (sinon classe 0)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--min-pred-h", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=1, help="1 frame sur N (test rapide)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--batch", type=int, default=8, help="images par lot d'inférence")
    ap.add_argument("--device", default=None, help="0, cpu... (défaut : GPU si disponible)")
    ap.add_argument("--dump-fp", type=int, default=0,
                    help="sauvegarde N images de fausses alarmes par météo")
    args = ap.parse_args()

    data = Path(args.data)
    entries = []
    for lab_p in sorted(data.glob("*/ep*/labels.json")):
        lab = json.loads(lab_p.read_text())
        if lab["split"] != args.split:
            continue
        for fr in lab["frames"]:
            entries.append((lab["weather"], lab["episode"], fr["idx"],
                            lab_p.parent / f"{fr['idx']:03d}_rgb.png", fr["objects"]))
    entries = entries[::args.stride]
    print(f"{len(entries)} frames ({args.split}), poids : {args.weights}, lots de {args.batch}")

    model = YOLO(args.weights)
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    paths = [str(e[3]) for e in entries]

    def predict_chunks():
        # Découpage explicite : ultralytics ignore "batch" quand on lui passe une liste de chemins
        for i in range(0, len(paths), args.batch):
            yield from model.predict(paths[i:i + args.batch], imgsz=args.imgsz, conf=args.conf,
                                     classes=COCO_VEHICLES if args.coco else [0],
                                     device=device, verbose=False)

    gt_stat = defaultdict(lambda: [0, 0])        # (météo, bin) -> [GT, détectés]
    det_stat = defaultdict(lambda: [0, 0, 0])    # météo -> [TP, FP, ignorées]
    n_frames = defaultdict(int)
    rows, fp_frames = [], []
    for e, r in zip(entries, predict_chunks()):
        weather, ep, idx, img_path, objs = e
        P = r.boxes.xyxy.cpu().numpy()
        C = r.boxes.conf.cpu().numpy()
        gts = [o for o in objs if not o["ignore"]]
        G, IG = boxes_of(gts), boxes_of([o for o in objs if o["ignore"]])
        used = np.zeros(len(gts), bool)
        gt_iou, gt_conf = np.zeros(len(gts)), np.zeros(len(gts))
        tp = fp = ign = 0
        fp_list = []
        if len(P):
            iou_g = box_iou(P, G)[0] if len(G) else np.zeros((len(P), 0))
            ioa_i = box_iou(P, IG)[1] if len(IG) else np.zeros((len(P), 0))
            for pi in np.argsort(-C):
                free = np.flatnonzero(~used)
                if len(free) and iou_g[pi, free].max() >= args.iou:
                    j = free[iou_g[pi, free].argmax()]
                    used[j], gt_iou[j], gt_conf[j] = True, iou_g[pi, j], C[pi]
                    tp += 1
                elif (P[pi, 3] - P[pi, 1]) < args.min_pred_h or \
                        (ioa_i.shape[1] and ioa_i[pi].max() >= 0.5):
                    ign += 1
                else:
                    fp += 1
                    fp_list.append((P[pi].tolist(), float(C[pi])))
        for k, v in enumerate((tp, fp, ign)):
            det_stat[weather][k] += v
        n_frames[weather] += 1
        if fp_list:
            fp_frames.append((weather, ep, idx, img_path, G, IG, fp_list))
        for j, o in enumerate(gts):
            for b in BINS:
                if b[0] <= o["distance"] < b[1]:
                    gt_stat[(weather, b)][0] += 1
                    gt_stat[(weather, b)][1] += int(used[j])
            rows.append([weather, ep, idx, o["id"], o["distance"], o["n_lidar"],
                         o["depth_face"], int(used[j]), round(float(gt_iou[j]), 3),
                         round(float(gt_conf[j]), 3)])

    weathers = sorted(n_frames)

    def cell(n, d):
        return f"{n:>4}/{d:<4} {100 * n / d:5.1f} %" if d else f"{'-':>16}"

    print(f"\n{'météo':<8}" + "".join(f"{f'{lo}-{hi} m':>17}" for lo, hi in BINS)
          + f"{'tout':>17}{'précision':>11}{'FP/frame':>10}")
    for w in weathers + ["toutes"]:
        ws = weathers if w == "toutes" else [w]
        cells, tn, td = [], 0, 0
        for b in BINS:
            n = sum(gt_stat[(x, b)][1] for x in ws)
            d = sum(gt_stat[(x, b)][0] for x in ws)
            cells.append(cell(n, d))
            tn, td = tn + n, td + d
        tp = sum(det_stat[x][0] for x in ws)
        fp = sum(det_stat[x][1] for x in ws)
        fr = sum(n_frames[x] for x in ws)
        prec = f"{100 * tp / (tp + fp):.1f} %" if tp + fp else "-"
        print(f"{w:<8}" + "".join(f"{c:>17}" for c in cells)
              + f"{cell(tn, td):>17}{prec:>11}{fp / fr:>10.2f}")

    tag = args.tag or Path(args.weights).stem
    out = ROOT / "runs" / f"eval_{tag}_{args.split}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["weather", "episode", "idx", "obj_id", "distance", "n_lidar",
                     "depth_face", "detected", "iou", "conf"])
        wr.writerows(rows)
    print(f"\nDétail par véhicule : {out}")

    if args.dump_fp:
        rng = random.Random(0)
        out_dir = ROOT / "runs" / f"fp_{tag}_{args.split}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for w in weathers:
            cand = [f for f in fp_frames if f[0] == w]
            for weather, ep, idx, p, G, IG, fps in rng.sample(cand, min(args.dump_fp, len(cand))):
                img = cv2.imread(str(p))
                draw(img, G, (0, 255, 0))                       # vert : vérité terrain
                draw(img, IG, (0, 165, 255))                    # orange : ignoré (trop caché)
                draw(img, [b for b, _ in fps], (0, 0, 255),     # rouge : fausse alarme + confiance
                     [f"{c:.2f}" for _, c in fps])
                cv2.imwrite(str(out_dir / f"{weather}_ep{ep:03d}_{idx:03d}.jpg"), img)
        print(f"Images de fausses alarmes : {out_dir}")


if __name__ == "__main__":
    main()
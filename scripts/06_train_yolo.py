r"""
Fine-tuning de YOLO sur le dataset v4 (une classe : vehicle). Fermer CARLA avant (VRAM).
Usage :
  python scripts\06_train_yolo.py --epochs 1 --fraction 0.05 --name smoke   # test de fumée
  python scripts\06_train_yolo.py --epochs 40 --name v4_n
Les poids sortent dans runs\train\<name>\weights\best.pt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--data", default=str(ROOT / "data" / "yolo_v4" / "dataset.yaml"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--fraction", type=float, default=1.0, help="part du train utilisée")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="v4_n")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                fraction=args.fraction, workers=args.workers, device=0, patience=10, seed=0,
                project=str(ROOT / "runs" / "train"), name=args.name, exist_ok=True, plots=True)


if __name__ == "__main__":     # obligatoire sous Windows pour les workers du dataloader
    main()
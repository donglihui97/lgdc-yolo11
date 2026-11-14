"""Train the uncompressed YOLO11n (the parent) at the camera-matched input size.

NEU-DET:  --ds neu --imgsz 256                         (300 epochs from COCO-pretrained yolo11n.pt)
GC10-DET: --ds gc10 --imgsz 640                        (300 epochs, square batches), then
          --ds gc10 --imgsz 640 --rect --model <square best.pt> --epochs 50 --lr0 0.001 --warmup 0 --tag ft
          (rectangular batches give the 320x640 input). Seeds are fixed integers, never time-based.
"""
import argparse

import _common  # noqa: F401
from ultralytics import YOLO

from lgdc import paths

ap = argparse.ArgumentParser()
ap.add_argument('--ds', required=True, choices=['neu', 'gc10'])
ap.add_argument('--imgsz', type=int, required=True)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--model', default='yolo11n.pt')
ap.add_argument('--tag', default='yolo11n')
ap.add_argument('--rect', action='store_true')
ap.add_argument('--epochs', type=int, default=300)
ap.add_argument('--lr0', type=float, default=0.01)
ap.add_argument('--warmup', type=float, default=3.0)
ap.add_argument('--workers', type=int, default=4)
a = ap.parse_args()
YOLO(a.model).train(
    data=paths.data_yaml(a.ds), imgsz=a.imgsz, epochs=a.epochs, batch=32, nbs=64, seed=a.seed, deterministic=True,
    optimizer='SGD', lr0=a.lr0, momentum=0.937, weight_decay=0.0005, warmup_epochs=a.warmup, amp=True, patience=0,
    workers=a.workers, device='0', save_period=-1, project=str(paths.RUNS),
    name=f'{a.ds}_{a.tag}_i{a.imgsz}{"r" if a.rect else ""}_s{a.seed}', exist_ok=False, plots=False, rect=a.rect)

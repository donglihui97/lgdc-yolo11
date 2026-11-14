"""Compress a trained parent and fine-tune it.

Depth compression (LGDC):   --units U4,U7,U8,U10      (paper names or c3k2.8,... module names)
Control (no compression):   --units none
Channel pruning baseline:   --speedup 2.0 [--round-to 8]
    DepGraph groups, L2 group importance with LAMP normalization, global iterative pruning
    until the FLOPs fall by the given factor; C2PSA and the final prediction convolutions are kept.
Add --distill <parent best.pt> for feature distillation from the parent (Ultralytics, lambda = 6).

Fine-tuning: SGD, lr0 0.002, no warmup, batch 32. The paper uses 300 epochs on NEU-DET and
100 epochs with --rect on GC10-DET; the 10-epoch runs of measure_units.py use the same script.
"""
import argparse
import copy

import _common  # noqa: F401
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C2PSA, Detect

from lgdc import paths
from lgdc.units import parse_units, remove_units, split_c2f

COMPRESSED = {}


class CompressedTrainer(DetectionTrainer):
    """Fine-tune the compressed module itself instead of rebuilding the model from its YAML."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        return COMPRESSED['model']


def channel_prune(model, imgsz, speedup, steps=200, round_to=None):
    import torch_pruning as tp
    model = split_c2f(copy.deepcopy(model).float().cpu())
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    det = model.model[-1]
    for m in model.modules():  # the export-style forward returns one tensor, which DepGraph can trace
        if isinstance(m, Detect):
            m.export, m.format = True, 'onnx'
    ignored = [det.dfl] + [b[-1] for b in det.cv2] + [b[-1] for b in det.cv3]
    ignored += [m for m in model.modules() if isinstance(m, C2PSA)]
    x = torch.zeros(1, 3, imgsz, imgsz)
    base_macs, base_params = tp.utils.count_ops_and_params(model, x)
    imp = tp.importance.GroupMagnitudeImportance(p=2, normalizer='lamp')
    pruner = tp.pruner.MetaPruner(model, x, importance=imp, iterative_steps=steps, pruning_ratio=0.9,
                                  ignored_layers=ignored, global_pruning=True, round_to=round_to)
    macs, params = base_macs, base_params
    for i in range(steps):
        pruner.step()
        macs, params = tp.utils.count_ops_and_params(model, x)
        if base_macs / macs >= speedup:
            break
    for m in model.modules():
        if isinstance(m, Detect):
            m.export = False
    print(f'pruned: MACs {base_macs / 1e9:.3f}G -> {macs / 1e9:.3f}G (x{base_macs / macs:.2f}), '
          f'params {base_params / 1e6:.3f}M -> {params / 1e6:.3f}M after {i + 1} steps', flush=True)
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True, choices=['neu', 'gc10'])
    ap.add_argument('--weights', required=True, help='trained parent best.pt')
    ap.add_argument('--imgsz', type=int, required=True)
    ap.add_argument('--units', default=None, help="units to remove, or 'none' for the control")
    ap.add_argument('--speedup', type=float, default=None, help='channel pruning to this FLOPs reduction factor')
    ap.add_argument('--round-to', type=int, default=None, help='round pruned channel counts to this multiple')
    ap.add_argument('--distill', default=None, help='teacher checkpoint (the parent)')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--rect', action='store_true', help='rectangular batches (GC10-DET, 320x640)')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--project', default=str(paths.RUNS))
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    if (a.units is None) == (a.speedup is None):
        ap.error('give exactly one of --units and --speedup')
    base = YOLO(a.weights)
    if a.units is not None:
        units = parse_units(a.units)
        COMPRESSED['model'] = remove_units(copy.deepcopy(base.model).float(), units)
        print('depth compression, removed:', units or 'nothing (control)', flush=True)
    else:
        COMPRESSED['model'] = channel_prune(base.model, a.imgsz, a.speedup, round_to=a.round_to)
    base.train(trainer=CompressedTrainer, data=paths.data_yaml(a.ds), imgsz=a.imgsz, epochs=a.epochs, batch=32,
               nbs=64, seed=a.seed, deterministic=True, optimizer='SGD', lr0=0.002, momentum=0.937,
               weight_decay=0.0005, warmup_epochs=0, amp=True, patience=0, workers=a.workers, device='0',
               save_period=-1, project=a.project, name=f'{a.ds}_{a.tag}_i{a.imgsz}_s{a.seed}', exist_ok=False,
               plots=False, rect=a.rect, **({'distill_model': a.distill} if a.distill else {}))

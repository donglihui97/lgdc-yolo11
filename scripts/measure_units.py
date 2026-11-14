"""LGDC step 1: measure the latency benefit and the accuracy cost of every removable unit.

--engines   builds, for the parent and for the parent with each single unit removed, gray-input
            FP16 TensorRT engines at batch 1 and 16, and writes the timing configs
            configs/units_<tag>_b1.json and _b16.json. Time them in one locked-clock session
            with bench_gpu.py (tools/locked_timing.ps1).
--finetune  fine-tunes the parent with each single unit removed, and the complete parent, for
            10 epochs (seed 0) with compress.py, into runs/sens/. The last-epoch validation
            mAP50-95 of these runs gives a_u and a_0. The test split is not used.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import _common  # noqa: F401
from lgdc import paths
from lgdc import trt_utils as tu
from lgdc.units import removable_units

ap = argparse.ArgumentParser()
ap.add_argument('--ds', required=True, choices=['neu', 'gc10'])
ap.add_argument('--weights', required=True, help='trained parent best.pt')
ap.add_argument('--h', type=int, required=True, help='engine input height (NEU 256, GC10 320)')
ap.add_argument('--w', type=int, required=True, help='engine input width (NEU 256, GC10 640)')
ap.add_argument('--imgsz', type=int, required=True, help='training image size (NEU 256, GC10 640 with --rect)')
ap.add_argument('--rect', action='store_true')
ap.add_argument('--tag', required=True)
ap.add_argument('--engines', action='store_true')
ap.add_argument('--finetune', action='store_true')
ap.add_argument('--epochs', type=int, default=10)
a = ap.parse_args()

units = ['none'] + removable_units(tu.load_detector(a.weights))
if a.engines:
    eng_dir = paths.ENGINES / 'units'
    eng_dir.mkdir(parents=True, exist_ok=True)
    for u in units:
        for b in (1, 16):
            eng = eng_dir / f'{a.tag}_{u}_b{b}.engine'
            if not eng.exists():
                tu.make_engine(a.weights, eng, b, a.h, a.w, units=[] if u == 'none' else [u])
            print('engine', eng.name, flush=True)
    (paths.REPO / 'configs').mkdir(exist_ok=True)
    for b in (1, 16):
        cfg = {'name': f'units_{a.tag}_b{b}', 'rounds': 20, 'block': 128 if b == 1 else 32,
               'engines': {u: f'engines/units/{a.tag}_{u}_b{b}.engine' for u in units}}
        path = paths.REPO / 'configs' / f'units_{a.tag}_b{b}.json'
        path.write_text(json.dumps(cfg, indent=1))
        print('timing config', path, flush=True)
if a.finetune:
    script = str(Path(__file__).resolve().parent / 'compress.py')
    for u in units:
        name = f'{a.ds}_sens_{u.replace(".", "_")}_i{a.imgsz}_s0'
        if (paths.RUNS / 'sens' / name / 'weights' / 'last.pt').exists():
            continue
        cmd = [sys.executable, script, '--ds', a.ds, '--weights', a.weights, '--imgsz', str(a.imgsz), '--units', u,
               '--seed', '0', '--epochs', str(a.epochs), '--tag', f'sens_{u.replace(".", "_")}',
               '--project', str(paths.RUNS / 'sens')] + (['--rect'] if a.rect else [])
        print(' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)

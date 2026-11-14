"""Full-pipeline (JPEG to detections) throughput and latency of several engines, serial and overlapped.

Config (JSON): {"name": ..., "dataset": "neu", "geom": [H, W, decode_scale], "rounds": 20,
                "reference": "parent", "engines": {"parent": "engines/x.engine", ...}}
Every round runs each (engine, mode) condition once on the same 256 validation images, in an order
drawn from random.Random(0). Throughput gains are paired per round against the reference engine.
CPU: decoding and resizing (8 threads). GPU: engine and batch-level NMS.
"""
import argparse
import json
import random
import statistics
import threading
import time
from pathlib import Path

import _common  # noqa: F401
import torch

from lgdc import paths
from lgdc import pipeline as P
from lgdc import trt_utils as tu
from lgdc.stats import block_ci, clock_states, telemetry

IMAGES = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    cfg = json.loads(Path(ap.parse_args().config).read_text())
    ds, (H, W, s) = cfg['dataset'], cfg['geom']
    out = paths.RESULTS / cfg['name']
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(cfg, indent=1))
    pipes = {}
    for name, path in cfg['engines'].items():
        e = tu.Engine(paths.resolve(path))
        for mode, ov in (('serial', False), ('overlapped', True)):
            pipes[f'{name}|{mode}'] = P.Pipeline(e, P.GrayPrep(H, W, s), P.batched_nms, P.NC[ds], workers=8, overlap=ov)
    rows = P.split_rows(ds, 'val')
    names = list(pipes)
    rng = random.Random(0)
    stop, tele = threading.Event(), []
    th = threading.Thread(target=telemetry, args=(stop, tele), daemon=True)
    th.start()
    raw = []
    with torch.inference_mode():
        for n in names:
            pipes[n].run(P.make_batches(rows, pipes[n].B, 8))
        for r in range(cfg['rounds']):
            order = names[:]
            rng.shuffle(order)
            for n in order:
                p = pipes[n]
                batches = P.make_batches(rows, p.B, IMAGES // p.B, offset=r * IMAGES)
                p.run(P.make_batches(rows, p.B, 2))
                torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                _, timing = p.run(batches, record_stages=n.endswith('serial'))
                wall = (time.perf_counter_ns() - t0) / 1e9
                rec = {'round': r, 'condition': n, 'img_per_s': IMAGES / wall,
                       'latency_ms': [(t['end'] - t['start']) / 1e6 for t in timing]}
                if n.endswith('serial'):
                    rec['inference_ms'] = [t['inference_ms'] for t in timing]
                raw.append(rec)
            print(f'round {r + 1}/{cfg["rounds"]} done', flush=True)
    stop.set(); th.join()
    for p in pipes.values():
        p.close()
    (out / 'raw.json').write_text(json.dumps(raw))
    base = cfg['reference']
    summary = {}
    for n in names:
        xs = sorted([x for x in raw if x['condition'] == n], key=lambda x: x['round'])
        ref = sorted([x for x in raw if x['condition'] == f'{base}|{n.split("|")[1]}'], key=lambda x: x['round'])
        thr = [x['img_per_s'] for x in xs]
        lat = sorted(v for x in xs for v in x['latency_ms'])
        gain = [100 * (a['img_per_s'] / b['img_per_s'] - 1) for a, b in zip(xs, ref)]
        summary[n] = {'throughput_median': statistics.median(thr), 'throughput_ci': block_ci(thr),
                      'latency_median_ms': statistics.median(lat), 'latency_p95_ms': lat[int(.95 * len(lat)) - 1],
                      'throughput_gain_vs_reference_pct': statistics.median(gain),
                      'gain_ci_pct': block_ci(gain)}
        if 'inference_ms' in xs[0]:
            summary[n]['inference_median_ms'] = statistics.median(v for x in xs for v in x['inference_ms'])
    summary['_telemetry'] = clock_states(tele)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    for n in names:
        x = summary[n]
        print(f"{n:34s} {x['throughput_median']:8.1f} img/s  gain {x['throughput_gain_vs_reference_pct']:+.2f}% "
              f"[{x['gain_ci_pct'][0]:+.2f}, {x['gain_ci_pct'][1]:+.2f}]  lat {x['latency_median_ms']:.2f}/{x['latency_p95_ms']:.2f}"
              + (f"  infer {x['inference_median_ms']:.2f}" if 'inference_median_ms' in x else ''))
    print(summary['_telemetry'])


if __name__ == '__main__':
    main()

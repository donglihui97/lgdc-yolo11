"""GPU network latency (engine input to engine output) of several TensorRT engines in one session.

Config (JSON): {"name": ..., "rounds": 20, "block": 128, "engines": {"label": "engines/x.engine", ...}}
Paths are relative to LGDC_ROOT. Every round runs each engine for one block of CUDA Graph replays,
in an order drawn from random.Random(0). Reported per engine: median and p95 over all replays and a
moving-block bootstrap interval of the per-block medians. Run it with locked clocks
(tools/locked_timing.ps1); the GPU clock is logged once per second.
"""
import argparse
import json
import random
import statistics
import threading
from pathlib import Path

import _common  # noqa: F401
import torch

from lgdc import paths
from lgdc import trt_utils as tu
from lgdc.stats import block_ci, clock_states, telemetry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    cfg = json.loads(Path(ap.parse_args().config).read_text())
    out = paths.RESULTS / cfg['name']
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(cfg, indent=1))
    engines = {k: tu.Engine(paths.resolve(v)) for k, v in cfg['engines'].items()}
    for e in engines.values():
        e.time_block(200)
    rng = random.Random(0)
    stop, tele = threading.Event(), []
    th = threading.Thread(target=telemetry, args=(stop, tele), daemon=True)
    th.start()
    names = list(engines)
    raw = {k: [] for k in names}
    with torch.inference_mode():
        for _ in range(cfg['rounds']):
            order = names[:]
            rng.shuffle(order)
            for k in order:
                engines[k].time_block(5)
                raw[k].append(engines[k].time_block(cfg['block']))
    stop.set(); th.join()
    summary = {}
    base = None
    for k in names:
        flat = sorted(v for b in raw[k] for v in b)
        med = statistics.median(flat)
        base = med if base is None else base
        summary[k] = {'median_ms': med, 'p95_ms': flat[int(.95 * len(flat)) - 1],
                      'block_median_ci': block_ci([statistics.median(b) for b in raw[k]])}
        print(f"{k:28s} {med:7.3f} ms  p95 {summary[k]['p95_ms']:7.3f}  vs first {100 * (med - base) / base:+6.1f}%",
              flush=True)
    summary['_telemetry'] = clock_states(tele)
    print('GPU clock states:', summary['_telemetry'])
    (out / 'raw.json').write_text(json.dumps(raw))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

"""CPU network latency with ONNX Runtime for the ONNX files written by build_eval.py.

Config (JSON): {"name": ..., "threads": 8, "rounds": 20, "block": 25, "models": {"label": "engines/x.onnx", ...}}
All models are timed in one session; the model order of each round is drawn from random.Random(0)
and the input is a fixed uint8 image drawn with numpy seed 0.
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import onnxruntime as ort

from lgdc import paths

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
cfg = json.loads(Path(ap.parse_args().config).read_text())
opts = ort.SessionOptions()
opts.intra_op_num_threads = cfg['threads']
opts.inter_op_num_threads = 1
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess, feeds = {}, {}
rng_in = np.random.default_rng(0)
for k, p in cfg['models'].items():
    s = ort.InferenceSession(str(paths.resolve(p)), opts, providers=['CPUExecutionProvider'])
    i = s.get_inputs()[0]
    feeds[k] = {i.name: rng_in.integers(0, 255, size=i.shape, dtype=np.uint8)}
    sess[k] = s
    for _ in range(10):
        s.run(None, feeds[k])
rng = random.Random(0)
names = list(sess)
times = {k: [] for k in names}
for _ in range(cfg['rounds']):
    order = names[:]
    rng.shuffle(order)
    for k in order:
        for _ in range(2):
            sess[k].run(None, feeds[k])
        for _ in range(cfg['block']):
            t0 = time.perf_counter()
            sess[k].run(None, feeds[k])
            times[k].append(1000 * (time.perf_counter() - t0))
out = {k: {'median_ms': statistics.median(v), 'p95_ms': sorted(v)[int(.95 * len(v)) - 1], 'n': len(v)}
       for k, v in times.items()}
base = out[names[0]]['median_ms']
for k in names:
    print(f"{k:22s} {out[k]['median_ms']:8.3f} ms  p95 {out[k]['p95_ms']:8.3f}  "
          f"vs first {100 * (out[k]['median_ms'] - base) / base:+6.1f}%", flush=True)
paths.RESULTS.mkdir(parents=True, exist_ok=True)
(paths.RESULTS / f"{cfg['name']}.json").write_text(json.dumps({'config': cfg, 'results': out}, indent=1))

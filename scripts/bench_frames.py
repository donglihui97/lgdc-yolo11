"""Frame-grabber pipeline: decoded gray frames in host memory to detections, one frame at a time.

Inline cameras deliver uncompressed frames through a frame grabber, so this benchmark starts from
gray frames that are already decoded into host memory and measures, per frame, the CPU resize and
letterbox (prepare), the host-to-GPU copy (transfer), the engine (inference), and the GPU NMS with
the copy of the detections back to the host (post). Serial mode processes one frame after the other;
overlapped mode prepares frame k+1 on the CPU while the GPU processes frame k.

Config (JSON): {"name": ..., "dataset": "gc10", "geom": [H, W], "rounds": 20, "frames": 256,
                "reference": "parent", "post": ["gpu", "host"], "engines": {"parent": "engines/x_b1.engine", ...}}
"post" selects the NMS: "gpu" (batched_nms on the GPU) and/or "host" (host_nms on the CPU).
Every round runs each (engine, mode) condition once on the same frames, in an order drawn from
random.Random(0). Throughput gains are paired per round against the reference engine.
"""
import argparse
import json
import random
import statistics
import threading
import time
from pathlib import Path

import _common  # noqa: F401
import cv2
import numpy as np
import torch

from lgdc import paths
from lgdc import pipeline as P
from lgdc import trt_utils as tu
from lgdc.stats import block_ci, clock_states, telemetry


class FramePrep:
    """Letterbox a decoded gray frame into the one-channel uint8 engine input."""
    gray = True

    def __init__(self, H, W):
        self.H, self.W = H, W

    def image(self, frame, out):
        h, w = frame.shape
        r, nw, nh, top, left = P.letterbox_params(h, w, self.H, self.W)
        im = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else frame
        o = out[0]
        if top or left or nh != self.H or nw != self.W:
            o.fill(P.PAD)
        o[top:top + nh, left:left + nw] = im
        return P.Meta(r, top, left, 1.0, 1.0, w, h)

    def to_engine(self, host, dev_input, stream):
        with torch.cuda.stream(stream):
            dev_input.copy_(host, non_blocking=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    cfg = json.loads(Path(ap.parse_args().config).read_text())
    ds, (H, W) = cfg['dataset'], cfg['geom']
    out = paths.RESULTS / cfg['name']
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(cfg, indent=1))
    rows = P.split_rows(ds, 'val')
    n = cfg['frames']
    frames = [cv2.imdecode(np.fromfile(rows[i % len(rows)]['path'], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
              for i in range(n)]
    batches = [([f], [(f.shape[1], f.shape[0])]) for f in frames]
    posts = {'gpu': P.batched_nms, 'host': P.host_nms}
    pipes = {}
    for name, path in cfg['engines'].items():
        e = tu.Engine(paths.resolve(path))
        for post in cfg.get('post', ['gpu']):
            for mode, ov in (('serial', False), ('overlapped', True)):
                pipes[f'{name}|{post}|{mode}'] = P.Pipeline(e, FramePrep(H, W), posts[post], P.NC[ds], workers=0,
                                                            overlap=ov)
    names = list(pipes)
    rng = random.Random(0)
    stop, tele = threading.Event(), []
    th = threading.Thread(target=telemetry, args=(stop, tele), daemon=True)
    th.start()
    raw = []
    with torch.inference_mode():
        for k in names:
            pipes[k].run(batches[:32])
        for r in range(cfg['rounds']):
            order = names[:]
            rng.shuffle(order)
            for k in order:
                p = pipes[k]
                p.run(batches[:4])
                torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                _, timing = p.run(batches, record_stages=k.endswith('serial'))
                wall = (time.perf_counter_ns() - t0) / 1e9
                rec = {'round': r, 'condition': k, 'fps': n / wall,
                       'latency_ms': [(t['end'] - t['start']) / 1e6 for t in timing]}
                if k.endswith('serial'):
                    for s in ('prepare_ms', 'transfer_ms', 'inference_ms', 'post_ms'):
                        rec[s] = [t[s] for t in timing]
                raw.append(rec)
            print(f'round {r + 1}/{cfg["rounds"]} done', flush=True)
    stop.set(); th.join()
    (out / 'raw.json').write_text(json.dumps(raw))
    base = cfg['reference']
    summary = {}
    for k in names:
        xs = sorted([x for x in raw if x['condition'] == k], key=lambda x: x['round'])
        ref = sorted([x for x in raw if x['condition'] == f'{base}|{k.split("|", 1)[1]}'], key=lambda x: x['round'])
        fps = [x['fps'] for x in xs]
        lat = sorted(v for x in xs for v in x['latency_ms'])
        gain = [100 * (a['fps'] / b['fps'] - 1) for a, b in zip(xs, ref)]
        summary[k] = {'fps_median': statistics.median(fps), 'fps_ci': block_ci(fps),
                      'latency_median_ms': statistics.median(lat), 'latency_p95_ms': lat[int(.95 * len(lat)) - 1],
                      'latency_p99_ms': lat[int(.99 * len(lat)) - 1],
                      'fps_gain_vs_reference_pct': statistics.median(gain), 'gain_ci_pct': block_ci(gain)}
        if 'prepare_ms' in xs[0]:
            for s in ('prepare_ms', 'transfer_ms', 'inference_ms', 'post_ms'):
                summary[k][s] = statistics.median(v for x in xs for v in x[s])
    summary['_telemetry'] = clock_states(tele)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    for k in names:
        x = summary[k]
        stages = ''.join(f'  {s[:-3]} {x[s]:.3f}' for s in ('prepare_ms', 'transfer_ms', 'inference_ms', 'post_ms') if s in x)
        print(f"{k:34s} {x['fps_median']:7.1f} fps  gain {x['fps_gain_vs_reference_pct']:+.2f}% "
              f"[{x['gain_ci_pct'][0]:+.2f}, {x['gain_ci_pct'][1]:+.2f}]  lat {x['latency_median_ms']:.3f}/"
              f"{x['latency_p95_ms']:.3f}/{x['latency_p99_ms']:.3f}{stages}")
    print('GPU clock states:', summary['_telemetry'])


if __name__ == '__main__':
    main()

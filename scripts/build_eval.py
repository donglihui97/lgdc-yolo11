"""Build the deployment engine of trained weights and evaluate it on the validation and test splits.

The engine takes a one-channel uint8 image (gray folding) and runs in FP16. Evaluation decodes the
JPEGs, letterboxes them to H x W, runs the engine and a batch-level NMS (conf 0.001, IoU 0.7,
300 detections), and computes COCO mAP50-95 / mAP50 in original-image coordinates.
NEU-DET: --h 256 --w 256.  GC10-DET: --h 320 --w 640 --scale 2 (JPEG decoded at 1/2 scale).
The ONNX file written next to the engine is the model used by bench_cpu.py.
"""
import argparse
import json

import _common  # noqa: F401
from lgdc import paths
from lgdc import pipeline as P
from lgdc import trt_utils as tu

ap = argparse.ArgumentParser()
ap.add_argument('--ds', required=True, choices=['neu', 'gc10'])
ap.add_argument('--weights', required=True)
ap.add_argument('--h', type=int, required=True)
ap.add_argument('--w', type=int, required=True)
ap.add_argument('--scale', type=int, default=1, choices=[1, 2, 4, 8], help='JPEG DCT decoding scale 1/s')
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--tag', required=True)
ap.add_argument('--std', action='store_true', help='3-channel FP32-input engine instead of gray folding')
ap.add_argument('--no-eval', action='store_true')
a = ap.parse_args()
paths.ENGINES.mkdir(parents=True, exist_ok=True)
eng = paths.ENGINES / f'{a.tag}_{a.h}x{a.w}_{"f32rgb" if a.std else "u8g"}_b{a.batch}.engine'
if not eng.exists():
    tu.make_engine(a.weights, eng, a.batch, a.h, a.w, gray=not a.std)
print('engine', eng, flush=True)
if not a.no_eval:
    prep = P.StandardPrep(a.h, a.w) if a.std else P.GrayPrep(a.h, a.w, a.scale)
    pipe = P.Pipeline(tu.Engine(eng), prep, P.batched_nms, P.NC[a.ds], workers=8, overlap=True)
    res = {s: P.evaluate(pipe, a.ds, s) for s in ('val', 'test')}
    pipe.close()
    summary = {s: {'ap50_95': round(100 * r['ap50_95'], 3), 'ap50': round(100 * r['ap50'], 3)} for s, r in res.items()}
    print(a.tag, summary, flush=True)
    (eng.parent / f'{eng.stem}_ap.json').write_text(json.dumps({'args': vars(a), 'result': res}, indent=1))

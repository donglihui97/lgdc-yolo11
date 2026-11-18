"""JPEG-to-detection pipeline around a static-batch TensorRT engine.

The pipeline maps a list of JPEG paths to detections in original-image coordinates,
(K, 7) rows of x1, y1, x2, y2, score, class, image index. Decoding and resizing run on
the CPU; the engine and a batch-level NMS run on the GPU.
"""
from __future__ import annotations

import csv
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torchvision

from .paths import DATA, NC  # noqa: F401  (NC is re-exported for the scripts)

PAD = 114


# ----------------------------------------------------------------------------- data
def split_rows(ds, split):
    with open(DATA[ds] / 'manifest.csv', encoding='utf8') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == split]
    rows.sort(key=lambda r: r['id'])
    for i, r in enumerate(rows):
        r['image_id'] = i
        r['path'] = str(DATA[ds] / r['image'])
        r['width'], r['height'] = int(r['width']), int(r['height'])
    return rows


def coco_ground_truth(ds, split):
    rows = split_rows(ds, split)
    names = [l.split(':', 1)[1].strip() for l in (DATA[ds] / 'data.yaml').read_text(encoding='utf8').splitlines()
             if l.strip()[:1].isdigit() and ':' in l]
    images, anns = [], []
    for r in rows:
        w, h = r['width'], r['height']
        images.append({'id': r['image_id'], 'file_name': r['image'], 'width': w, 'height': h})
        for line in (DATA[ds] / r['label']).read_text(encoding='utf8').split('\n'):
            if not line.strip():
                continue
            c, x, y, bw, bh = line.split()
            bw, bh = float(bw) * w, float(bh) * h
            x0, y0 = float(x) * w - bw / 2, float(y) * h - bh / 2
            anns.append({'id': len(anns) + 1, 'image_id': r['image_id'], 'category_id': int(c),
                         'bbox': [x0, y0, bw, bh], 'area': bw * bh, 'iscrowd': 0})
    cats = [{'id': i, 'name': n} for i, n in enumerate(names)]
    return {'images': images, 'annotations': anns, 'categories': cats}


def coco_map(gt_dict, dets_per_image, max_det=300):
    """dets_per_image: {image_id: (K, 6) array x1 y1 x2 y2 score cls}."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import contextlib, io
    res = []
    for iid, d in dets_per_image.items():
        for x1, y1, x2, y2, s, c in d.tolist():
            if x2 > x1 and y2 > y1:
                res.append({'image_id': iid, 'category_id': int(c), 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': s})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = gt_dict; gt.createIndex()
        dt = gt.loadRes(res) if res else COCO()
        ev = COCOeval(gt, dt, 'bbox'); ev.params.maxDets = [1, 10, max_det]
        ev.evaluate(); ev.accumulate()
    p = ev.eval['precision']
    mean = lambda v: float(v[v > -1].mean()) if (v > -1).any() else float('nan')
    per_class = {gt.cats[c]['name']: mean(p[:, :, i, 0, -1]) for i, c in enumerate(ev.params.catIds)}
    return {'ap50_95': mean(p[:, :, :, 0, -1]), 'ap50': mean(p[0, :, :, 0, -1]), 'per_class': per_class,
            'predictions': len(res)}


# ----------------------------------------------------------------------------- geometry
def letterbox_params(h, w, H, W):
    """Ultralytics LetterBox(auto=False, center=True) geometry."""
    r = min(H / h, W / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (W - nw) / 2, (H - nh) / 2
    return r, nw, nh, int(round(dh - 0.1)), int(round(dw - 0.1))


DECODE_FLAGS = {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2,
                4: cv2.IMREAD_REDUCED_GRAYSCALE_4, 8: cv2.IMREAD_REDUCED_GRAYSCALE_8}


def matched_decode_scale(h, w, H, W):
    """Largest DCT scale 1/s whose decoded image still needs no upsampling to fit H x W."""
    r = min(H / h, W / w)
    s = 1
    while s < 8 and r * 2 * s <= 1.0:
        s *= 2
    return s


@dataclass
class Meta:
    """Map network coordinates back to the original image."""
    ratio: float
    top: int
    left: int
    sx: float
    sy: float
    width: int
    height: int


# ----------------------------------------------------------------------------- preparation
class StandardPrep:
    """Colour decode, letterbox, RGB CHW uint8; FP32 /255 on the GPU (3-channel engines)."""
    gray = False

    def __init__(self, H, W):
        self.H, self.W = H, W

    def image(self, path, out):
        data = np.fromfile(path, dtype=np.uint8)
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        r, nw, nh, top, left = letterbox_params(h, w, self.H, self.W)
        if (nw, nh) != (w, h):
            im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.H, self.W, 3), PAD, np.uint8)
        canvas[top:top + nh, left:left + nw] = im
        out[:] = canvas[:, :, ::-1].transpose(2, 0, 1)
        return Meta(r, top, left, 1.0, 1.0, w, h)

    def to_engine(self, host, dev_input, stream):
        with torch.cuda.stream(stream):
            dev_input.copy_(host.to('cuda', non_blocking=True).float().div_(255.0))


class GrayPrep:
    """Luma-only decode at the matched DCT scale, letterbox into a 1-channel uint8 slot."""
    gray = True

    def __init__(self, H, W, scale=1):
        self.H, self.W, self.scale = H, W, scale

    def image(self, path, out):
        im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), DECODE_FLAGS[self.scale])
        h, w = im.shape[:2]
        r, nw, nh, top, left = letterbox_params(h, w, self.H, self.W)
        if (nw, nh) != (w, h):
            im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        o = out[0]
        if top or left or nh != self.H or nw != self.W:
            o.fill(PAD)
        o[top:top + nh, left:left + nw] = im
        return Meta(r, top, left, None, None, w, h)  # sx, sy are filled by the caller

    def to_engine(self, host, dev_input, stream):
        with torch.cuda.stream(stream):
            dev_input.copy_(host, non_blocking=True)


# ----------------------------------------------------------------------------- postprocessing
def batched_nms(raw, conf, iou, max_det, nc):
    """One class-aware NMS over the whole batch with a single host synchronization.

    Boxes of different (image, class) groups are offset so that they never overlap;
    the offsets are applied in float64 so the IoU of boxes in one group is unchanged.
    """
    B = raw.shape[0]
    best, cls = raw[:, 4:, :].max(1)
    b, a = (best > conf).nonzero(as_tuple=True)
    if b.numel() == 0:
        return torch.zeros((0, 7), device=raw.device)
    score, c = best[b, a], cls[b, a]
    xywh = raw[b, :4, a]
    xyxy = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
    group = (b * nc + c).double()
    shifted = xyxy.double() + group[:, None] * 4096.0
    keep = torchvision.ops.nms(shifted, score.double(), iou)
    kb = b[keep]
    order = torch.argsort(kb, stable=True)
    keep, kb = keep[order], kb[order]
    counts = torch.bincount(kb, minlength=B)
    start = torch.cumsum(counts, 0) - counts
    rank = torch.arange(len(kb), device=raw.device) - start[kb]
    keep = keep[rank < max_det]
    return torch.cat([xyxy[keep], score[keep, None], c[keep, None].float(), b[keep, None].float()], 1)


def host_nms(raw, conf, iou, max_det, nc):
    """Copy the raw output to the host once, then filter by confidence and run class-aware NMS on the CPU.

    For one frame with a few dozen candidates this avoids the many small GPU kernels and the
    synchronizations of batched_nms. Returns the same (K, 7) layout on the CPU.
    """
    r = raw.cpu().numpy()
    out = []
    for i in range(r.shape[0]):
        scores = r[i, 4:, :]
        best = scores.max(0)
        keep = np.nonzero(best > conf)[0]
        if not len(keep):
            continue
        cls = scores[:, keep].argmax(0)
        xywh = r[i, :4, keep]  # (K, 4): the advanced indices i and keep put K first
        xyxy = torch.from_numpy(np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1))
        s, c = torch.from_numpy(best[keep]), torch.from_numpy(cls)
        k = torchvision.ops.batched_nms(xyxy, s, c, iou)[:max_det]
        out.append(torch.cat([xyxy[k], s[k, None], c[k, None].float(), torch.full((len(k), 1), float(i))], 1))
    return torch.cat(out) if out else torch.zeros((0, 7))


def to_original(det, metas):
    """det: (K, 7) CPU array in network coordinates -> original coordinates (in place)."""
    if not len(det):
        return det
    idx = det[:, 6].astype(int)
    r = np.array([m.ratio for m in metas])[idx]
    top = np.array([m.top for m in metas])[idx]
    left = np.array([m.left for m in metas])[idx]
    sx = np.array([m.sx for m in metas])[idx]
    sy = np.array([m.sy for m in metas])[idx]
    W = np.array([m.width for m in metas])[idx]
    H = np.array([m.height for m in metas])[idx]
    det[:, 0] = np.clip((det[:, 0] - left) / r * sx, 0, W)
    det[:, 2] = np.clip((det[:, 2] - left) / r * sx, 0, W)
    det[:, 1] = np.clip((det[:, 1] - top) / r * sy, 0, H)
    det[:, 3] = np.clip((det[:, 3] - top) / r * sy, 0, H)
    return det


# ----------------------------------------------------------------------------- pipeline
class Pipeline:
    """A JPEG-to-detection pipeline around one static-batch TensorRT engine.

    workers=0 decodes in the calling thread one image at a time;
    workers>0 decodes the images of a batch in parallel threads.
    overlap=True prepares batch k+1 on the CPU while the GPU processes batch k.
    """

    def __init__(self, engine, prep, post, nc, workers=0, overlap=False, conf=0.25, iou=0.7, max_det=300,
                 slots=3):
        self.engine, self.prep, self.post, self.nc = engine, prep, post, nc
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.B = engine.input.shape[0]
        self.workers, self.overlap = workers, overlap
        self.pool = ThreadPoolExecutor(workers) if workers else None
        c = 1 if prep.gray else 3
        self.host = [torch.empty((self.B, c, prep.H, prep.W), dtype=torch.uint8).pin_memory()
                     for _ in range(slots if overlap else 1)]
        self.stream = engine.stream

    def close(self):
        if self.pool:
            self.pool.shutdown()

    def _prepare(self, paths, slot, sizes):
        buf = self.host[slot].numpy()
        def one(i):
            m = self.prep.image(paths[i], buf[i])
            if m.sx is None:
                w0, h0 = sizes[i]
                m.sx, m.sy, m.width, m.height = w0 / m.width, h0 / m.height, w0, h0
            return m
        n = len(paths)
        metas = list(self.pool.map(one, range(n))) if self.pool else [one(i) for i in range(n)]
        for i in range(n, self.B):  # pad an incomplete batch; its detections are dropped
            buf[i].fill(PAD)
        return metas

    def _gpu(self, slot, metas, n, stage=None):
        self.prep.to_engine(self.host[slot], self.engine.input, self.stream)
        if stage is not None:
            self.stream.synchronize(); stage.append(time.perf_counter_ns())
        raw = self.engine.run()
        if stage is not None:
            self.stream.synchronize(); stage.append(time.perf_counter_ns())
        with torch.cuda.stream(self.stream):
            det = self.post(raw, self.conf, self.iou, self.max_det, self.nc)
            det = det[det[:, 6] < n]
            host = det.cpu()
        if stage is not None:
            stage.append(time.perf_counter_ns())
        return to_original(host.numpy().copy(), metas)

    def run(self, batches, record_stages=False):
        """batches: list of (paths, sizes). Returns per-batch detections and timing records."""
        results, timing = [], []
        if not self.overlap:
            for paths, sizes in batches:
                t0 = time.perf_counter_ns()
                metas = self._prepare(paths, 0, sizes)
                st = [time.perf_counter_ns()] if record_stages else None
                det = self._gpu(0, metas, len(paths), st)
                t1 = time.perf_counter_ns()
                results.append(det)
                rec = {'start': t0, 'end': t1}
                if record_stages:
                    rec.update(prepare_ms=(st[0] - t0) / 1e6, transfer_ms=(st[1] - st[0]) / 1e6,
                               inference_ms=(st[2] - st[1]) / 1e6, post_ms=(st[3] - st[2]) / 1e6)
                timing.append(rec)
            return results, timing
        q = queue.Queue(maxsize=len(self.host) - 1)
        free = queue.Queue()
        for s in range(len(self.host)):
            free.put(s)
        error = []

        def producer():
            try:
                for paths, sizes in batches:
                    slot = free.get()
                    t0 = time.perf_counter_ns()
                    metas = self._prepare(paths, slot, sizes)
                    q.put((slot, metas, len(paths), t0))
            except BaseException as e:  # surfaced in the consumer
                error.append(e)
            q.put(None)

        th = threading.Thread(target=producer, daemon=True)
        th.start()
        while True:
            item = q.get()
            if item is None:
                break
            slot, metas, n, t0 = item
            det = self._gpu(slot, metas, n)
            free.put(slot)
            results.append(det)
            timing.append({'start': t0, 'end': time.perf_counter_ns()})
        th.join()
        if error:
            raise error[0]
        return results, timing


def make_batches(rows, B, n_batches=None, offset=0):
    """Consecutive batches over the split, cycling when more batches than images are requested."""
    if n_batches is None:
        out = []
        for i in range(0, len(rows), B):
            chunk = rows[i:i + B]
            out.append(([r['path'] for r in chunk], [(r['width'], r['height']) for r in chunk]))
        return out
    out = []
    for k in range(n_batches):
        chunk = [rows[(offset + k * B + j) % len(rows)] for j in range(B)]
        out.append(([r['path'] for r in chunk], [(r['width'], r['height']) for r in chunk]))
    return out


def evaluate(pipe, ds, split, conf=0.001):
    rows = split_rows(ds, split)
    pipe.conf = conf
    batches = make_batches(rows, pipe.B)
    results, _ = pipe.run(batches)
    dets = {}
    for k, det in enumerate(results):
        base = k * pipe.B
        for i in range(len(batches[k][0])):
            dets[base + i] = det[det[:, 6] == i][:, :6]
    return coco_map(coco_ground_truth(ds, split), dets)

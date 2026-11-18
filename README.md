# Latency-Guided Depth Compression (LGDC) for YOLO11n

Code for the paper *Latency-Guided Depth Compression for Single-Image GPU Inference in Steel Surface Defect Detection*.

Inline steel inspection often sends the frames of one camera to the GPU one image at a time. At the camera input size (256×256 on NEU-DET, 320×640 on GC10-DET), the batch-1 latency of YOLO11n is set by the fixed cost of each layer, so removing layers saves more time than making them narrower. LGDC removes whole units of the trained network. For each of twelve removable units it measures, on the target GPU, the latency saved by removing that unit and the accuracy lost after a short fine-tuning. It then removes the units with the best ratio until the summed saving reaches a target, and recovers accuracy by distillation from the uncompressed network.

## Results

Three training seeds; GPU latency is the TensorRT FP16 network latency (engine input to engine output) on an RTX 3080 Ti Laptop GPU with the SM clock locked at 1200 MHz. The channel-pruning baseline halves the FLOPs, and its "rounded" variant rounds every pruned channel count to a multiple of 8.

| Dataset | Network | Test mAP50–95 | GPU latency, batch 1 | GPU latency, batch 16 |
|---|---|---|---|---|
| NEU-DET | Parent YOLO11n | 41.30 ± 0.75 | 0.821 ms | 1.910 ms |
| NEU-DET | Channel pruning, rounded | 41.09 ± 0.73 | 0.787 ms (−4.1%) | 1.575 ms (−17.5%) |
| NEU-DET | LGDC | 40.49 ± 0.61 | 0.689 ms (−16.1%) | 1.709 ms (−10.5%) |
| GC10-DET | Parent YOLO11n | 31.51 ± 1.49 | 0.959 ms | 4.158 ms |
| GC10-DET | Channel pruning, rounded | 30.20 ± 1.18 | 0.896 ms (−6.6%) | 3.457 ms (−16.9%) |
| GC10-DET | LGDC | 31.49 ± 0.61 | 0.736 ms (−23.3%) | 3.682 ms (−11.4%) |

At batch 16 the layers are limited by arithmetic, and channel pruning becomes the faster choice. The paper also reports CPU latency (ONNX Runtime), the frame rate of a frame-grabber pipeline with host-side NMS, the variation between TensorRT builds, and an ablation of the selection rule.

## Removable units

Removing any of these units leaves every tensor shape unchanged, so no other layer has to be modified (`lgdc/units.py`).

| Paper | Module name | What is removed |
|---|---|---|
| U1–U4 | `c3k2.2`, `c3k2.4`, `c3k2.6`, `c3k2.8` | inner blocks of the backbone C3k2 modules (replaced by identities) |
| U5 | `sppf.9` | SPPF block (identity) |
| U6 | `c2psa.10` | C2PSA attention block (identity) |
| U7–U10 | `c3k2.13`, `c3k2.16`, `c3k2.19`, `c3k2.22` | inner blocks of the neck C3k2 modules |
| U11 | `head.reg` | second 3×3 stage of the box-regression branch, at all three scales |
| U12 | `head.cls` | second 3×3 stage of the classification branch, at all three scales |

With a target of 25% at batch 1, LGDC selects U4, U7, U8, U10 on NEU-DET and U3, U5, U6, U10 on GC10-DET. The measurements behind these selections are in `measurements/`, and you can reproduce both selections without a GPU:

```bash
python scripts/select_units.py --table measurements/units_neu.json
python scripts/select_units.py --table measurements/units_gc10.json
```

## Repository layout

```
lgdc/            units.py (unit removal, C3k2 split for pruning), trt_utils.py (ONNX/TensorRT, gray folding),
                 pipeline.py (decoding, batched NMS, COCO evaluation), stats.py, paths.py
scripts/         train_parent.py, measure_units.py, select_units.py, compress.py,
                 build_eval.py, bench_gpu.py, bench_cpu.py, bench_pipeline.py, bench_frames.py
tools/           prepare_data.py (dataset conversion and split), clock_ctl.ps1, locked_timing.ps1
configs/         timing-session configs for the comparisons of the paper
splits/          image ids of the train/val/test splits used in the paper
measurements/    per-unit latency benefit and accuracy cost (seed 0) used for the selection
```

## Installation

```bash
pip install -r requirements.txt
```

Install `torch` and `torchvision` from the PyTorch index that matches your CUDA version. If TensorRT is installed outside the Python environment, set `TRT_PATH` to the directory that contains the `tensorrt` package. Generated files go to `runs/`, `engines/` and `results/` under the repository root; set `LGDC_ROOT` to use another working directory and `LGDC_DATA` to point to the prepared datasets elsewhere.

## Data

Put the original archives under `data/`: `NEU-DET.zip` and the GC10-DET `archive.zip`. Then run

```bash
python tools/prepare_data.py --project .
```

The script reads the archives without extracting them, converts the VOC annotations to YOLO labels, groups exact and near-duplicate images (pHash distance ≤ 4) so that a group never spans two splits, and draws a 70/15/15 split stratified by class with seed 42. It writes `data/processed/{neu_det,gc10_det}/` with `images/`, `labels/`, `data.yaml`, `manifest.csv` and `splits/{train,val,test}.txt`, and an audit of every annotation change to `reports/`. The split lists should equal the ones in `splits/`; their SHA-256 values are

| File | SHA-256 |
|---|---|
| neu_det train | `aaec4bc11eb7afe354e4b97d9102a359e4f5e94f2839ed3c8d7bd55cc82aabf0` |
| neu_det val | `dac5664ff244a183b35720b62d631577a9c6643ad989781a837a3d949e870d0c` |
| neu_det test | `00dd927b1c21fb7b43997d6688eece6268102ff94dafa9a755733db8a757c971` |
| gc10_det train | `e74c397074afa107d73f7e2b8c1d7945b80bdbe443a0003c1c151aab52e29d1f` |
| gc10_det val | `e2d1559f5e2f198f1a216b61002c34fbb400cf4568c30fda35fb0272280b00b4` |
| gc10_det test | `a60c1f4e01c6fda0b67b97951a3cc260033b58c94656db4c79df44e0d9617df1` |

NEU-DET has 1800 images (1260/270/270); GC10-DET keeps 2291 annotated images (1605/345/341).

## Reproducing the paper

The commands below use NEU-DET and seed 0; the paper repeats every training with seeds 0, 1 and 2. Seeds are fixed integers and never derived from the clock. GC10-DET differences are given after each step.

**1. Train the parent** (300 epochs, SGD, lr0 0.01, batch 32, from COCO-pretrained `yolo11n.pt`).

```bash
python scripts/train_parent.py --ds neu --imgsz 256 --seed 0
```

GC10-DET: train at `--imgsz 640` with square batches, then fine-tune with rectangular batches, which gives the 320×640 input (rectangular batches from the start would disable mosaic augmentation):

```bash
python scripts/train_parent.py --ds gc10 --imgsz 640 --seed 0
python scripts/train_parent.py --ds gc10 --imgsz 640 --rect --seed 0 --model runs/gc10_yolo11n_i640_s0/weights/best.pt --epochs 50 --lr0 0.001 --warmup 0 --tag ft
```

Below, `P` is the parent checkpoint (`runs/neu_yolo11n_i256_s0/weights/best.pt`, or `runs/gc10_ft_i640r_s0/weights/best.pt` for GC10-DET).

**2. Measure every unit** (parent seed 0 only). This builds one engine per unit at batch 1 and 16, and runs one 10-epoch fine-tuning per unit and one for the complete network.

```bash
python scripts/measure_units.py --ds neu --weights P --h 256 --w 256 --imgsz 256 --tag neu --engines --finetune
tools\locked_timing.ps1 -Script bench_gpu.py -Configs configs\units_neu_b1.json,configs\units_neu_b16.json
```

GC10-DET: `--h 320 --w 640 --imgsz 640 --rect --tag gc10`.

**3. Select the units** (benefit at batch 1, target 25%, cost floor 0.1 points).

```bash
python scripts/select_units.py --timing results/units_neu_b1/summary.json --ds neu --imgsz 256
```

**4. Compress and recover** with distillation from the parent (NEU-DET 300 epochs; GC10-DET 100 epochs with `--rect`).

```bash
# LGDC
python scripts/compress.py --ds neu --weights P --imgsz 256 --units U4,U7,U8,U10 --distill P --epochs 300 --seed 0 --tag lgdc
# control: same fine-tuning, nothing removed
python scripts/compress.py --ds neu --weights P --imgsz 256 --units none --distill P --epochs 300 --seed 0 --tag control
# channel pruning to half the FLOPs, irregular and rounded channel counts
python scripts/compress.py --ds neu --weights P --imgsz 256 --speedup 2.0 --distill P --epochs 300 --seed 0 --tag chprune
python scripts/compress.py --ds neu --weights P --imgsz 256 --speedup 2.0 --round-to 8 --distill P --epochs 300 --seed 0 --tag chprune_r8
```

GC10-DET: `--imgsz 640 --rect --epochs 100 --units U3,U5,U6,U10`.

**5. Build the engines and evaluate** on the validation and test splits. The engine reads one-channel uint8 images; the RGB replication and the division by 255 are folded into the first convolution.

```bash
python scripts/build_eval.py --ds neu --weights runs/neu_lgdc_i256_s0/weights/best.pt --h 256 --w 256 --batch 16 --tag neu_lgdc_s0
python scripts/build_eval.py --ds neu --weights runs/neu_lgdc_i256_s0/weights/best.pt --h 256 --w 256 --batch 1 --tag neu_lgdc_s0 --no-eval
```

GC10-DET: `--h 320 --w 640 --scale 2` (JPEGs are decoded at half resolution before resizing). Use the tags `{ds}_parent_s0`, `{ds}_lgdc_s0`, `{ds}_chprune_s0` and `{ds}_chprune_r8_s0` so that the engine names match `configs/`.

**6. Time the networks** with locked clocks: GPU network latency, CPU latency (ONNX Runtime, 8 threads), and the full pipeline.

```bash
tools\locked_timing.ps1 -Script bench_gpu.py -Configs configs\gpu_neu_b1.json,configs\gpu_neu_b16.json
tools\locked_timing.ps1 -Script bench_cpu.py -Configs configs\cpu_neu_b1.json,configs\cpu_neu_b16.json
tools\locked_timing.ps1 -Script bench_pipeline.py -Configs configs\pipeline_neu_b1.json,configs\pipeline_neu_b16.json
```

**7. Frame budget.** `bench_frames.py` starts from gray frames already decoded into host memory, as a frame grabber delivers them, and times prepare, transfer, network and NMS per frame at batch 1, serial and overlapped. `"post": ["gpu", "host"]` compares the batched NMS on the GPU with `host_nms` (one copy of the raw output to the host, then NMS on the CPU), which returns the same detections. Example config: `configs/frames_gc10_b1.json`.

```bash
tools\locked_timing.ps1 -Script bench_frames.py -Configs configs\frames_neu_b1.json,configs\frames_gc10_b1.json
```

## Timing protocol

A laptop lowers its clocks as it heats up, so every timing session runs with locked clocks. `tools/clock_ctl.ps1` fixes the GPU SM clock at 1200 MHz (`nvidia-smi -lgc`) and disables CPU turbo boost, which keeps the CPU at its 2.3 GHz base clock; it needs an elevated PowerShell, or two scheduled tasks that run it with highest privileges. All engines compared in one table are timed in the same session: 20 rounds, the engine order of each round drawn from `random.Random(0)`, and a block of CUDA Graph replays per engine and round (128 at batch 1, 32 at batch 16). The scripts report the median and the 95th percentile and a moving-block bootstrap interval of the median (block 5, 2000 resamples, seed 0). `bench_gpu.py` samples the GPU clock once per second and prints every clock state seen during the session, so a session in which the clock left 1200 MHz can be discarded.

## Citation

```bibtex
@misc{lgdc2026,
  title   = {Latency-Guided Depth Compression for Single-Image GPU Inference in Steel Surface Defect Detection},
  author  = {...},
  note    = {Manuscript under review},
  year    = {2026}
}
```

## License

The code builds on [Ultralytics](https://github.com/ultralytics/ultralytics), which is released under AGPL-3.0, and on [Torch-Pruning](https://github.com/VainF/Torch-Pruning). NEU-DET and GC10-DET are distributed by their authors under their own terms and are not included here.

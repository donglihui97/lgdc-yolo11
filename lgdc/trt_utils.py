"""ONNX export, TensorRT engine building and CUDA Graph execution."""
import copy
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_DLL = []


def import_trt():
    """Import tensorrt. If it is not installed in the environment, set TRT_PATH to a
    directory that contains the tensorrt package (and tensorrt_libs on Windows)."""
    extra = os.environ.get('TRT_PATH')
    if extra and extra not in sys.path:
        sys.path.insert(0, extra)
    if os.name == 'nt' and not _DLL:
        dirs = [Path(torch.__file__).parent / 'lib'] + ([Path(extra) / 'tensorrt_libs'] if extra else [])
        for d in dirs:
            if d.is_dir():
                _DLL.append(os.add_dll_directory(str(d)))
    import tensorrt as trt
    return trt


def load_detector(weights):
    """Fused FP32 Ultralytics detection model whose head returns decoded (B, 4+nc, N)."""
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import Detect
    model = YOLO(str(weights)).model.float().eval().fuse()
    for m in model.modules():
        if isinstance(m, Detect):
            m.export, m.format, m.dynamic = True, 'onnx', False
    return model


class Standard(nn.Module):
    """Engine input: (B, 3, H, W) float32 RGB in [0, 1]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class GrayFolded(nn.Module):
    """Engine input: (B, 1, H, W) uint8 gray levels.

    For an image whose three channels are equal, conv(W, g/255 * 1_3) equals
    conv(sum_c W_c / 255, g), so channel replication and scaling are folded into
    the stem weights. The model is copied, so the caller's model is unchanged.
    """

    def __init__(self, model):
        super().__init__()
        model = copy.deepcopy(model)
        stem = model.model[0].conv
        assert stem.in_channels == 3 and stem.groups == 1
        folded = nn.Conv2d(1, stem.out_channels, stem.kernel_size, stem.stride, stem.padding,
                           stem.dilation, bias=stem.bias is not None)
        with torch.no_grad():
            folded.weight.copy_(stem.weight.sum(dim=1, keepdim=True) / 255.0)
            if stem.bias is not None:
                folded.bias.copy_(stem.bias)
        model.model[0].conv = folded
        self.model = model

    def forward(self, g):
        return self.model(g.float())


def export_onnx(wrapper, path, batch, height, width, gray):
    shape = (batch, 1 if gray else 3, height, width)
    x = torch.zeros(shape, dtype=torch.uint8) if gray else torch.zeros(shape)
    with torch.no_grad():
        wrapper(x)
        torch.onnx.export(wrapper, x, str(path), opset_version=17, input_names=['images'],
                          output_names=['output0'], do_constant_folding=True, dynamo=False)
    return path


def build_engine(onnx_path, engine_path, fp16=True, workspace_gb=4):
    trt = import_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError('; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError('engine build failed')
    Path(engine_path).write_bytes(bytes(blob))
    return engine_path


def make_engine(weights, engine_path, batch, height, width, units=(), gray=True):
    """Trained weights -> (optional unit removal) -> ONNX -> FP16 TensorRT engine."""
    from .units import remove_units
    model = load_detector(weights)
    if units:
        remove_units(model, list(units))
    wrap = (GrayFolded(model) if gray else Standard(model)).eval()
    onnx_path = Path(engine_path).with_suffix('.onnx')
    export_onnx(wrap, onnx_path, batch, height, width, gray)
    return build_engine(onnx_path, engine_path)


def engine_bytes(path):
    """Serialized engine, skipping the JSON metadata prefix written by Ultralytics exports."""
    data = Path(path).read_bytes()
    n = int.from_bytes(data[:4], 'little')
    if 0 < n < 1 << 20 and data[4:5] == b'{':
        return data[4 + n:]
    return data


class Engine:
    """Static-shape engine bound to torch CUDA tensors and replayed through a CUDA Graph.

    Inputs are filled with random values, which is what the latency benchmarks use;
    the detection pipeline overwrites the input tensor 'images' before each run.
    """

    def __init__(self, engine_path, stream=None):
        trt = import_trt()
        dt = {trt.float32: torch.float32, trt.float16: torch.float16, trt.uint8: torch.uint8,
              trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool}
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes(engine_path))
        self.context = self.engine.create_execution_context()
        self.stream = stream or torch.cuda.Stream()
        self.tensors = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            t = torch.zeros(shape, dtype=dt[self.engine.get_tensor_dtype(name)], device='cuda')
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if t.dtype == torch.uint8:
                    t.random_(0, 255)
                elif t.is_floating_point():
                    t.uniform_()
            self.tensors[name] = t
            self.context.set_tensor_address(name, t.data_ptr())
        self.input = self.tensors.get('images')
        self.output = self.tensors.get('output0')
        self.graph = None

    def capture(self):
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

    def run(self):
        """Enqueue one inference on self.stream and return the output tensor."""
        if self.graph is None:
            self.capture()
        with torch.cuda.stream(self.stream):
            self.graph.replay()
        return self.output

    def time_block(self, n):
        """GPU time (ms) of n consecutive graph replays, one value per replay."""
        if self.graph is None:
            self.capture()
        out = []
        for _ in range(n):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(self.stream)
            with torch.cuda.stream(self.stream):
                self.graph.replay()
            e.record(self.stream)
            e.synchronize()
            out.append(s.elapsed_time(e))
        return out

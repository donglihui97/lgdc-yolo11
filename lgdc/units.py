"""Removable units of YOLO11n and the C3k2 split used by channel pruning."""
import copy

import torch
import torch.nn as nn

from .unit_names import PAPER_NAME, UNIT_IDS, parse_units  # noqa: F401


class C2fSplit(nn.Module):
    """C2f/C3k2 whose first 1x1 convolution is split into its two output halves.

    The output is identical to the original module. The split lets Torch-Pruning
    trace the channel dependencies of the chunk operation in C3k2.
    """

    def __init__(self, src):
        super().__init__()
        c = src.c
        self.c, self.m, self.cv2 = c, src.m, src.cv2
        self.cv0, self.cv1 = copy.deepcopy(src.cv1), copy.deepcopy(src.cv1)
        for half, sl in ((self.cv0, slice(0, c)), (self.cv1, slice(c, 2 * c))):
            conv, bn = half.conv, half.bn
            conv.weight = nn.Parameter(conv.weight.data[sl].clone())
            conv.out_channels = c
            for name in ('weight', 'bias'):
                setattr(bn, name, nn.Parameter(getattr(bn, name).data[sl].clone()))
            bn.running_mean, bn.running_var = bn.running_mean[sl].clone(), bn.running_var[sl].clone()
            bn.num_features = c
        for attr in ('f', 'i', 'type', 'np'):
            if hasattr(src, attr):
                setattr(self, attr, getattr(src, attr))

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def split_c2f(model):
    from ultralytics.nn.modules.block import C2f
    for i, m in enumerate(model.model):
        if isinstance(m, C2f):
            model.model[i] = C2fSplit(m)
    return model


def removable_units(model):
    """Units whose removal keeps every tensor shape: the inner block of each C3k2, SPPF,
    C2PSA, and the second stage of the regression and classification branches of the head."""
    from ultralytics.nn.modules import SPPF, C2PSA, Detect
    from ultralytics.nn.modules.block import C2f
    units = []
    for i, m in enumerate(model.model):
        if isinstance(m, (C2f, C2fSplit)):
            units.append(f'c3k2.{i}')
        elif isinstance(m, SPPF) and m.cv1.conv.in_channels == m.cv2.conv.out_channels:
            units.append(f'sppf.{i}')
        elif isinstance(m, C2PSA):
            units.append(f'c2psa.{i}')
        elif isinstance(m, Detect):
            units += ['head.reg', 'head.cls']
    return units


class Skip(nn.Module):
    """Identity that keeps the Ultralytics routing attributes of the module it replaces."""

    def __init__(self, src):
        super().__init__()
        for attr in ('f', 'i', 'type', 'np'):
            if hasattr(src, attr):
                setattr(self, attr, getattr(src, attr))

    def forward(self, x):
        return x


def remove_units(model, units):
    """Remove the given units in place and return the model.

    c3k2.i    the inner blocks of C3k2 module i become identities (the module still
              concatenates its input split with an unprocessed copy);
    sppf.i, c2psa.i  the module becomes an identity;
    head.reg, head.cls  the second 3x3 stage of the regression / classification branch
              is deleted at all three scales.
    """
    det = model.model[-1]
    for u in units:
        kind, _, idx = u.partition('.')
        if kind == 'c3k2':
            blk = model.model[int(idx)]
            blk.m = nn.ModuleList(nn.Identity() for _ in blk.m)
        elif kind in ('sppf', 'c2psa'):
            model.model[int(idx)] = Skip(model.model[int(idx)])
        elif u == 'head.reg':
            det.cv2 = nn.ModuleList(nn.Sequential(b[0], b[-1]) for b in det.cv2)
        elif u == 'head.cls':
            det.cv3 = nn.ModuleList(nn.Sequential(b[0], b[-1]) for b in det.cv3)
        else:
            raise ValueError(f'unknown unit {u}')
    return model

"""This file contains experimental modules"""

import numpy as np
import torch
from torch import nn

from .common import Conv


class CrossConv(nn.Module):
    """
       Cross Convolution Downsample
       args: ch_in, ch_out, kernel, stride, groups, expansion, shortcut
    """

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

        super().__init__()

    def forward(self, x):
        """Forward"""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """
        Cross Convolution CSP
        args: ch_in, ch_out, number, shortcut, groups, expansion
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

        super().__init__()

    def forward(self, x):
        """Forward"""

        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)

        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class GhostConv(nn.Module):
    """
        Ghost Convolution https://github.com/huawei-noah/ghostnet
        args: ch_in, ch_out, kernel, stride, groups
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, c_, act)

        super().__init__()

    def forward(self, x):
        """Forward"""

        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class MixConv2d(nn.Module):
    """Mixed Depthwise Conv https://arxiv.org/abs/1907.09595"""

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super().__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum()
                  for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            # solve for equal weight indices, ax = b
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        """Forward"""
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    """Ensemble of models"""

    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False):
        """Forward"""
        y = []
        for module in self:
            y.append(module(x, augment)[0])

        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a"""

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        data = torch.load(w, map_location=map_location)
        data_model = data['model']
        model.append(data_model.float().fuse().eval())  # load FP32 model

    if len(model) == 1:
        return model[-1]

    for k in ['names', 'stride']:
        setattr(model, k, getattr(model[-1], k))
    return model

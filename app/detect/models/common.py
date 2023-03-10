"""This file contains modules common to various models"""

import collections
import math

import torch
from mish_cuda import MishCuda as Mish
from torch import nn


def autopad(k, p=None):
    """Pad to 'same'
       arguments: kernel, padding"""

    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def dw_conv(c1, c2, k=1, s=1, act=True):
    """Depthwise convolution"""
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    """Standard convolution
       arguments: ch_in, ch_out, kernel, stride, padding, groups"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p),
                              groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = Mish() if act else nn.Identity()

    def forward(self, x):
        """Run network"""
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        """Run fuse"""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck
       arguments: ch_in, ch_out, shortcut, groups, expansion"""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Run network"""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
       ch_in, ch_out, number, shortcut, groups, expansion"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = Mish()
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        """Run network"""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)

        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class BottleneckCSP2(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
       arguments: ch_in, ch_out, number, shortcut, groups, expansion:"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1):
        super().__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = Mish()
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        """Run network"""
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)

        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class VoVCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
       arguments: ch_in, ch_out"""

    def __init__(self, c1, c2):
        super().__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1//2, c_//2, 3, 1)
        self.cv2 = Conv(c_//2, c_//2, 3, 1)
        self.cv3 = Conv(c_, c2, 1, 1)

    def forward(self, x):
        """Run network"""
        _, x1 = x.chunk(2, dim=1)
        x1 = self.cv1(x1)
        x2 = self.cv2(x1)

        return self.cv3(torch.cat((x1, x2), dim=1))


class SPP(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Run network"""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPCSP(nn.Module):
    """CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks"""
    def __init__(self, c1, c2, _=1, __=False, ___=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.bn = nn.BatchNorm2d(2 * c_) 
        self.act = Mish()
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Focus(nn.Module):
    """Focus wh information into c-space
       arguments: ch_in, ch_out, kernel, stride, padding, groups"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        """Run network"""
        return self.conv(torch.cat([x[..., ::2, ::2],
                                    x[..., 1::2, ::2],
                                    x[..., ::2, 1::2],
                                    x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension"""

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Run network"""
        return torch.cat(x, self.d)


class CombConvLayer(nn.Sequential):
    """ConvLayer + DWConvLayer"""

    def __init__(self, in_channels, out_channels, kernel=1, stride=1):
        super().__init__()
        self.add_module("layer1", ConvLayer(in_channels, out_channels, kernel))
        self.add_module("layer2", DWConvLayer(out_channels, stride=stride))


class DWConvLayer(nn.Sequential):
    """dwconv + norm"""

    def __init__(self, in_channels, stride=1, bias=False):
        super().__init__()
        groups = in_channels
        self.add_module("dwconv", nn.Conv2d(groups,
                                            groups,
                                            kernel_size=3,
                                            stride=stride,
                                            padding=1,
                                            groups=groups,
                                            bias=bias))
        self.add_module("norm", nn.BatchNorm2d(groups))


class ConvLayer(nn.Sequential):
    """conv + norm + relu"""

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        self.add_module("conv", nn.Conv2d(in_channels,
                                          out_ch,
                                          kernel_size=kernel,
                                          stride=stride,
                                          padding=kernel // 2,
                                          groups=groups,
                                          bias=bias))
        self.add_module("norm", nn.BatchNorm2d(out_ch))
        self.add_module("relu", nn.ReLU6(True))


class HarDBlock(nn.Module):
    """HarDBlock"""

    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0  # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(
                i+1, in_channels, growth_rate, grmul)
            self.links.append(link)
            if dwconv:
                layers_.append(CombConvLayer(inch, outch))
            else:
                layers_.append(Conv(inch, outch, k=3))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        """Run network"""
        layers_ = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or (i == t-1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


class BRLayer(nn.Sequential):
    """BRLayer (norm + relu)"""

    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))


class HarDBlock2(nn.Module):
    """HarDBlock2"""

    def get_link(self, layer, base_ch, growth_rate, grmul):
        """Returns out channels, in channels, link"""

        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.insert(0, k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        """Returs out channels"""
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers):
        super().__init__()
        self.links = []
        conv_layers_ = []
        bnrelu_layers_ = []
        self.layer_bias = []
        self.out_channels = 0
        self.out_partition = collections.defaultdict(list)

        for i in range(n_layers):
            outch, _, link = self.get_link(
                i+1, in_channels, growth_rate, grmul)
            self.links.append(link)
            for j in link:
                self.out_partition[j].append(outch)

        cur_ch = in_channels
        for i in range(n_layers):
            accum_out_ch = sum(self.out_partition[i])
            real_out_ch = self.out_partition[i][0]

            conv_layers_.append(nn.Conv2d(cur_ch,
                                          accum_out_ch,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=True))
            bnrelu_layers_.append(BRLayer(real_out_ch))
            cur_ch = real_out_ch
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += real_out_ch

        self.conv_layers = nn.ModuleList(conv_layers_)
        self.bnrelu_layers = nn.ModuleList(bnrelu_layers_)

    def transform(self, blk, trt=False):
        """Transform weight matrix from a pretrained HarDBlock v1"""

        in_ch = blk.layers[0][0].weight.shape[1]
        for i in range(len(self.conv_layers)):
            link = self.links[i].copy()
            link_ch = [blk.layers[k-1][0].weight.shape[0] if k > 0 else
                       blk.layers[0][0].weight.shape[1] for k in link]
            part = self.out_partition[i]
            w_src = blk.layers[i][0].weight
            b_src = blk.layers[i][0].bias

            self.conv_layers[i].weight[0:part[0],
                                       :, :, :] = w_src[:, 0:in_ch, :, :]
            self.layer_bias.append(b_src)

            if b_src is not None:
                if trt:
                    self.conv_layers[i].bias[1:part[0]] = b_src[1:]
                    self.conv_layers[i].bias[0] = b_src[0]
                    self.conv_layers[i].bias[part[0]:] = 0
                    self.layer_bias[i] = None
                else:
                    self.conv_layers[i].bias = None
            else:
                self.conv_layers[i].bias = None

            in_ch = part[0]
            link_ch.reverse()
            link.reverse()
            if len(link) > 1:
                for j in range(1, len(link)):
                    ly = link[j]
                    part_id = self.out_partition[ly].index(part[0])
                    chos = sum(self.out_partition[ly][0:part_id])
                    choe = chos + part[0]
                    chis = sum(link_ch[0:j])
                    chie = chis + link_ch[j]
                    self.conv_layers[ly].weight[chos:choe,
                                                :, :, :] = w_src[:, chis:chie, :, :]

            self.bnrelu_layers[i] = None
            if isinstance(blk.layers[i][1], nn.BatchNorm2d):
                self.bnrelu_layers[i] = nn.Sequential(
                    blk.layers[i][1], blk.layers[i][2])
            else:
                self.bnrelu_layers[i] = blk.layers[i][1]

    def forward(self, x):
        """Run network"""

        layers_ = []
        outs_ = []
        xin = x
        for i in enumerate(self.conv_layers):
            link = self.links[i]
            part = self.out_partition[i]
            xout = self.conv_layers[i](xin)
            layers_.append(xout)
            xin = xout[:, 0:part[0], :, :] if len(part) > 1 else xout
            if len(link) > 1:
                for j in range(len(link) - 1):
                    ly = link[j]
                    part_id = self.out_partition[ly].index(part[0])
                    chs = sum(self.out_partition[ly][0:part_id])
                    che = chs + part[0]

                    xin += layers_[ly][:, chs:che, :, :]
            xin = self.bnrelu_layers[i](xin)
            if i % 2 == 0 or i == len(self.conv_layers)-1:
                outs_.append(xin)
        out = torch.cat(outs_, 1)
        return out

"""YOLO module"""

import math
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from ..utils.general import check_anchor_order, make_divisible
from ..utils.torch_utils import (fuse_conv_and_bn, initialize_weights,
                                 model_info, scale_img, time_synchronized)
from .common import *
from .experimental import C3, CrossConv, MixConv2d


class Detect(nn.Module):
    """Detect nn-module"""

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer

        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(
            self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1)
                               for x in ch)  # output conv
        self.export = False  # onnx export
        super().__init__()

    def forward(self, x):
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(
                0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
                               self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * \
                    self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    """args: model, input channels, number of classes"""

    def __init__(self, cfg='yolov4-p5.yaml', ch=3, nc=None):
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding %s nc=%g with nc=%g' %
                  (cfg, self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(
            deepcopy(self.yaml), input_channels=[ch])  # model, savelist, ch_out

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()

        super().__init__()

    def forward(self, x, augment=False, profile=False):
        """Forward"""

        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            # single-scale inference, train
            return self.forward_once(x, profile)

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[
                        0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    # initialize biases into Detect(), cf is class frequency
    def _initialize_biases(self, cf=None):
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b[:, 4].data += math.log(8 / (640 / s) ** 2)
            b[:, 5:].data += math.log(0.6 / (m.nc - 0.99)
                                      ) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) %
                  (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self):  # print model information
        model_info(self)


def parse_model(model_dict, input_channels):  # model_dict, input_channels(3)
    """Parse model"""

    anchors = model_dict['anchors']
    classes_num = model_dict['nc']
    depth_multiple = model_dict['depth_multiple']
    width_multiple = model_dict['width_multiple']

    anchors_num = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    outputs_num = anchors_num * (classes_num + 5)

    layers = []  # layers
    save = []  # savelist
    out_channel = input_channels[-1]  # ch out

    backbone = model_dict['backbone']
    head = model_dict['head']
    confs = backbone + head

    # from, number, module, args
    for i, (_from, number, module, args) in enumerate(confs):
        module = eval(module) if isinstance(module, str) else module  # eval strings

        for j, arg in enumerate(args):
            args[j] = eval(arg) if isinstance(arg, str) else arg  # eval strings

        number = max(round(number * depth_multiple), 1) if number > 1 else number  # depth gain

        case_0 = [nn.Conv2d,
                  Conv,
                  Bottleneck,
                  SPP,
                  dw_conv,
                  MixConv2d,
                  Focus,
                  CrossConv,
                  BottleneckCSP,
                  BottleneckCSP2,
                  SPPCSP,
                  VoVCSP,
                  C3]

        case_1 = [HarDBlock, HarDBlock2]

        if module in case_0:
            in_channel = input_channels[_from]
            out_channel = args[0]
            out_channel = make_divisible(out_channel * width_multiple, 8) if out_channel != outputs_num else out_channel
            args = [in_channel, out_channel, *args[1:]]

            sub_case_0 = [BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP, C3]
            if module in sub_case_0:
                args.insert(2, number)
                number = 1
        elif module in case_1:
            in_channel = input_channels[_from]
            args = [in_channel, *args[:]]
        elif module is nn.BatchNorm2d:
            args = [input_channels[_from]]
        elif module is Concat:
            out_channel = sum([input_channels[-1 if x == -1 else x + 1] for x in _from])
        elif module is Detect:
            args.append([input_channels[x + 1] for x in _from])
            # number of anchors
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(_from)
        else:
            out_channel = input_channels[_from]

        dups = [module(*args) for _ in range(number)]
        seq = nn.Sequential(*dups) if number > 1 else module(*args)  # module

        module_type = str(module)[8:-2].replace("__main__.", "")  # module type
        np = sum([x.numel() for x in seq.parameters()])  # number params

        # attach index, 'from' index, type, number params
        seq.i = i
        seq.f = _from
        seq.type = module_type
        seq.np = np

        print_args = (i, _from, number, np, module_type, args)
        print("%3s%18s%3s%10.0f  %-40s%-30s" % print_args)  # print

        save.extend(x % i for x in ([_from] if isinstance(_from, int) else _from) if x != -1)  # append to savelist
        layers.append(seq)

        if module in case_1:
            out_channel = seq.get_out_ch()
        input_channels.append(out_channel)

    return nn.Sequential(*layers), sorted(save)

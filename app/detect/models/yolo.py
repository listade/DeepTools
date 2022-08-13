"""YOLO module"""

import math
from copy import deepcopy
from pathlib import Path

# import cv2
import torch
import yaml
from torch import nn

from ..utils.general import check_anchor_order, make_divisible
from ..utils.torch_utils import (fuse_conv_and_bn, initialize_weights,
                                 model_info, scale_img, time_synchronized)
from .common import (SPP, SPPCSP, Bottleneck, BottleneckCSP, BottleneckCSP2,
                     Concat, Conv, Focus, HarDBlock, HarDBlock2, VoVCSP,
                     dw_conv)
from .experimental import C3, CrossConv, MixConv2d


class Detect(nn.Module):
    """YOLO Detect nn.Dodule"""

    def __init__(self, nc=80, anchors=(), channels=()):  # detection layer
        super().__init__()

        self.stride = ()  # strides computed during build
        self.nc = nc
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors

        self.grid = [torch.zeros(1)] * self.nl  # init grid

        t1 = torch.tensor(anchors).float().view(self.nl, -1, 2) # shape(nl,na,2)
        t2 = t1.clone().view(self.nl, 1, -1, 1, 1, 2) # shape(nl,1,na,1,1,2)

        out_ch = self.no * self.na
        convs = (nn.Conv2d(in_ch, out_ch, 1) for in_ch in channels)

        self.m = nn.ModuleList(convs)  # output conv

        self.register_buffer('anchors', t1)
        self.register_buffer('anchor_grid', t2)

    def forward(self, x):
        """Inference"""

        output = []
        for i in range(self.nl):
            conv = self.m[i]
            x[i] = conv(x[i])  # conv

            bs, _, ny, nx = x[i].shape

            # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx) \
                        .permute(0, 1, 3, 4, 2) \
                        .contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()

                stride = self.stride[i]

                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * stride  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                output.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(output, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])

        return torch.stack((xv, yv), 2) \
                    .view((1, 1, ny, nx, 2)) \
                    .float()


class Model(nn.Module):
    """args: model, input channels, number of classes"""

    def __init__(self, cfg='yolov4-p5.yaml', ch=3, nc=None):
        super().__init__()

        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="utf-8") as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)

        # Define model
        if nc and nc != self.yaml['nc']:
            print(f"Overriding {cfg} nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc

        model_copy = deepcopy(self.yaml)
        self.model, self.save = parse_model(model_copy, input_channels=[ch])

        # Build strides, anchors
        model = self.model[-1]  # Detect()

        if isinstance(model, Detect):
            s = 256  # 2x min stride
            t = torch.zeros(1, ch, s, s)

            forwards = [s / x.shape[-2] for x in self.forward(t)]
            model.stride = torch.tensor(forwards)  # forward
            model.anchors /= model.stride.view(-1, 1, 1)

            check_anchor_order(model)

            self.stride = model.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()


    def forward(self, x, augment=False, profile=False):
        """Forward"""

        if augment:
            img_size = x.shape[-2:]  # height, width
            scales = [1, 0.83, 0.67]  # scales
            flips = [None, 3, None]  # flips (2-ud, 3-lr)
            outputs = []  # outputs

            for sc_val, fl_val in zip(scales, flips):
                fl_img = x.flip(fl_val) if fl_val else x
                sc_img = scale_img(fl_img, sc_val)

                # forward augmented
                y = self.forward_once(sc_img)[0]

                # np_sc_img = 255 * sc_img[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1]
                # cv2.imwrite(f"img_{sc_val}.jpg", np_sc_img)

                y[..., :4] /= sc_val  # unscale

                if fl_val == 2:
                    y[..., 1] = img_size[0] - y[..., 1]  # unflip up-down

                elif fl_val == 3:
                    y[..., 0] = img_size[1] - y[..., 0]  # unflip left-right

                outputs.append(y)

            return torch.cat(outputs, 1), None  # inference, train
        return self.forward_once(x, profile)


    def forward_once(self, x, profile=False):
        """Foobar"""

        y = []
        delta_t = []  # outputs

        for model in self.model:
            # if not from previous layer
            if model.f != -1:
                if isinstance(model.f, int):
                    x = y[model.f]
                else:
                    # from earlier layers
                    x = [x if j == -1 else y[j] for j in model.f]

            if profile:
                t_now = time_synchronized()

                for _ in range(10):
                    _ = model(x)

                delta_t.append((time_synchronized() - t_now) * 100)
                print(f"{model.np} {delta_t[-1]} {model.type}")

            x = model(x)  # run

            # save output
            y.append(x if model.i in self.save else None)

        if profile:
            print(f"{sum(delta_t)}ms total")

        return x


    def _initialize_biases(self, cf=None):
        """
            Initialize biases into Detect()
            arg: cf - class frequency
         """
        module = self.model[-1]  # Detect() module

        for m_i, stride in zip(module.m, module.stride):
            # conv.bias(255) to (3,85)
            bias_v = m_i.bias.view(module.na, -1)

            # obj (8 objects per 640 image)
            bias_v[:, 4].data += math.log(8 / (640 / stride) ** 2)

            if cf is not None:
                bias_v[:, 5:].data += torch.log(cf / cf.sum())
            else:
                bias_v[:, 5:].data += math.log(0.6 / (module.nc - 0.99)) # cls

            m_i.bias = torch.nn.Parameter(bias_v.view(-1), requires_grad=True)

    def fuse(self):
        """Fuse model Conv2d() + BatchNorm2d() layers"""
        print('Fusing layers... ')

        for m in self.model.modules():
            if isinstance(m, Conv):
                # m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()

        return self

    def info(self):
        """Print model information"""
        model_info(self, verbose=False)


def parse_model(model_dict, input_channels):  # model_dict, input_channels(3)
    """Parse model"""

    anchors = model_dict['anchors']
    nc = model_dict['nc']
    depth_multiple = model_dict['depth_multiple']
    width_multiple = model_dict['width_multiple']

    anchors_num = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    outputs_num = anchors_num * (nc + 5)

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
            try:
                args[j] = eval(arg) if isinstance(arg, str) else arg  # eval strings
            except NameError:
                pass

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

            if out_channel != outputs_num:
                out_channel = make_divisible(out_channel * width_multiple, 8)

            args = [in_channel, out_channel, *args[1:]]

            sub_case = [BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP, C3]
            if module in sub_case:
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
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(_from) # number of anchors
        else:
            out_channel = input_channels[_from]

        dups = [module(*args) for _ in range(number)]
        seq = nn.Sequential(*dups) if number > 1 else module(*args)  # module

        module_type = str(module)[8:-2].replace("__main__.", "")  # module type
        params_num = sum([x.numel() for x in seq.parameters()])  # number params

        # attach index, 'from' index, type, number params
        seq.i = i
        seq.f = _from
        seq.type = module_type
        seq.np = params_num

        print(f"{i} {_from} {number} {params_num} {module_type} {args}")

        lst = [x % i for x in ([_from] if isinstance(_from, int) else _from) if x != -1]

        # append to savelist
        save.extend(lst)
        layers.append(seq)

        if module in case_1:
            out_channel = seq.get_out_ch()
        input_channels.append(out_channel)

    return nn.Sequential(*layers), sorted(save)

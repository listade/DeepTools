"""PyTorch utils module"""

import math
import os
import time
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn


def init_seeds(seed=0):
    """Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html"""

    torch.manual_seed(seed)
    if seed == 0: # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else: # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def select_device(device, batch_size=1):
    """Device cpu or cuda"""

    if device == "cpu":
        print(f"Using {device}")
        return torch.device("cpu")

    words = device.split(":")
    if words[0] != "cuda":
        raise Exception(f"invalid device: {device}")

    os.environ["CUDA_VISIBLE_DEVICES"] = device
    if not torch.cuda.is_available():
        raise Exception("cuda is unavailable")

    device_count = torch.cuda.device_count()
    props = [torch.cuda.get_device_properties(i) for i in range(device_count)]
    infos = [f"{v.name} {v.total_memory / (1024 ** 2)}MiB" for v in props]

    if len(words) == 1:
        if device_count > 1 and device_count % batch_size != 0:
            raise Exception(f"invalid batch size {batch_size}")

        print("Using", ",".join(infos))
        return torch.device(device)

    num = words[-1]
    if int(num) >= device_count:
        raise Exception(f"invalid device num: {num}")

    print(f"Using {props[num]}")
    return torch.device(device)



def time_synchronized():
    """Waits for all kernels in all streams on a CUDA device to complete and return time"""

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def is_parallel(model):
    """Is model nn.parallel.DataParallel or nn.parallel.DistributedDataParallel"""
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def intersect_dicts(dict_x, dict_y, exclude=()):
    """Dictionary intersection"""

    return {k: v for k, v in dict_x.items() \
           if k in dict_y \
              and not any(x in k for x in exclude) \
              and v.shape == dict_y[k].shape}


def initialize_weights(model):
    """Init modules paramgs"""

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03
            continue
        if type(module) in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            module.inplace = True


def fuse_conv_and_bn(conv, bn):
    """https://tehnokv.com/posts/fusing-batchnorm-and-conv/"""

    with torch.no_grad():
        # init
        fusedconv = nn.Conv2d(conv.in_channels,
                              conv.out_channels,
                              kernel_size=conv.kernel_size,
                              stride=conv.stride,
                              padding=conv.padding,
                              bias=True).to(conv.weight.device)
        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        b_conv = conv.bias or torch.zeros(conv.weight.size(0), device=conv.weight.device)
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv


def model_info(model, verbose=False):
    """Plots a line-by-line description of a PyTorch model"""

    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients

    layers = len(list(model.parameters()))
    print(f"Model Summary: {layers} layers, {n_p} parameters, {n_g} gradients")

    if verbose:
        header = ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma")
        print("%5s %40s %9s %12s %20s %10s %10s" % header)

        params = model.named_parameters()
        for i, (name, p) in enumerate(params):
            name = name.replace("module_list.", "")
            values = (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std())
            print("%5g %40s %9s %12g %20s %10.3g %10.3g" % values)


def scale_img(img, ratio=1.0, same_shape=False):  # img(16,3,256,416), r=ratio
    """Scales img(bs,3,y,x) by ratio"""

    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        gs = 128#64#32  # (pixels) grid size
        h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]

    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
        Keep a moving average of everything in the model state_dict (parameters and buffers).
        This is intended to allow functionality like
        https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        A smoothed version of the weights is necessary for some training schemes to perform well.
        This class is sensitive where it is initialized in the sequence of model init,
        GPU assignment and distributed training wrappers. """

    def __init__(self, model, decay=0.9999, updates=0):
        # create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval() # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """Update EMA parameters"""

        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            # model state_dict
            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """Update EMA attributes"""

        for k, v in model.__dict__.items():
            if (len(include) > 0 and k not in include) or k.startswith('_') or k in exclude:
                continue
            setattr(self.ema, k, v)

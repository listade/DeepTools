"""This module contains general util functions"""

import glob
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from scipy.cluster.vq import kmeans
from torch import nn
from tqdm import tqdm

from .torch_utils import is_parallel

# Set printoptions
torch.set_printoptions(linewidth=320,
                       precision=5,
                       profile='long')

# format short g, %precision=5
np.set_printoptions(linewidth=320,
                    formatter={'float_kind': '{:11.5g}'.format})

matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


@contextmanager
def torch_distributed_zero_first(local_rank):
    """Wait distributed training for each local_master"""

    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def get_latest_run(path='./runs'):
    """Return path to most recent 'last.pt' in /runs (i.e. to --resume from)"""

    last_list = glob.glob(f'{path}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime)


def check_img_size(img_size, stride=32):
    """Verify img_size is a multiple of stride s"""

    new_size = make_divisible(img_size, int(stride))  # ceil gs-multiple

    if new_size != img_size:
        print(f"WARNING: --img-size {img_size} \
                must be multiple of max stride {stride}, \
                updating to {new_size}")

    return new_size


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    """Check anchor fit to data, recompute if necessary"""

    print("Analyzing anchors... ", end="")
    # Detect()
    m = model.module.model[-1] if hasattr(model, "module") else model.model[-1]
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(
        shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate(
        [l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1. / thr).float().mean()  # best possible recall
        return bpr, aat

    bpr, aat = metric(m.anchor_grid.clone().cpu().view(-1, 2))
    print("anchors/target = %.2f, Best Possible Recall (BPR) = %.4f" %
          (aat, bpr), end="")

    if bpr < 0.98:  # threshold to recompute
        print(
            f". Attempting to generate improved anchors, please wait...{bpr}")
        na = m.anchor_grid.numel() // 2  # number of anchors
        new_anchors = kmean_anchors(
            dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(new_anchors.reshape(-1, 2))[0]
        if new_bpr > bpr:  # replace anchors
            new_anchors = torch.tensor(
                new_anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchor_grid[:] = new_anchors.clone().view_as(
                m.anchor_grid)  # for inference
            m.anchors[:] = new_anchors.clone().view_as(
                m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            check_anchor_order(m)
            print(
                "New anchors saved to model. Update model *.yaml to use these anchors in the future.")
        else:
            print(
                "Original anchors better than new anchors. Proceeding with original anchors.")


def check_anchor_order(m):
    """Check anchor order against stride order for YOLO Detect() module m,
       and correct if necessary"""

    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def find_file(file):
    """Searches for file if not found locally"""

    if os.path.isfile(file):
        return file
    files = glob.glob('./**/' + file, recursive=True)  # find file
    if len(files) == 0:
        raise FileNotFoundError()
    return files[0]  # return first file if multiple found


def make_divisible(x, divisor):
    """Returns x evenly divisble by divisor"""

    return math.ceil(x / divisor) * divisor


def labels_to_class_weights(labels, nc=80):
    """Get class weights (inverse frequency) from training labels"""

    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class
    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize

    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    """Produces image weights based on class mAPs"""

    n = len(labels)
    class_counts = np.array(
        [np.bincount(labels[i][:, 0].astype(np.int), minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)

    return image_weights


def xyxy2xywh(x):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h]"""
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height

    return y


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape"""
    if ratio_pad is None:  # calculate from img0_shape
        # gain  = old / new
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])
        # wh padding
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    clip_coords(coords, img0_shape)

    return coords


def clip_coords(boxes, img_shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)"""

    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def ap_per_class(tp, conf, pred_cls, target_cls):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    # Find unique classes
    unique_classes = np.unique(target_cls)
    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    s = [unique_classes.shape[0], tp.shape[1]]
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects
        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)
            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            # r at pr_score, negative x, xp because xp decreases
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])
            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            # p at pr_score
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])
            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    f1 = 2 * p * r / (p + r + 1e-16)
    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))
    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        # points where x axis (recall) changes
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True, gi_ou=False, di_ou=False, ci_ou=False):
    """Returns the IoU of box1 to box2. box1 is 4, box2 is nx4"""

    box2 = box2.T
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if gi_ou or di_ou or ci_ou:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        # convex height
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if gi_ou:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if di_ou or ci_ou:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + \
                ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if di_ou:
                return iou - rho2 / c2  # DIoU
            # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
            if ci_ou:
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v + 1e-16)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(),
       i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)"""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        """TF implementation 
            https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py"""
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def smooth_bce(eps=0.1):
    """Returns positive, negative label smoothing BCE targets
    https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441"""

    return 1.0 - 0.5 * eps, 0.5 * eps


def compute_loss(p, targets, model):
    """Args: predictions, targets, model"""
    device = targets.device

    lcls = torch.zeros(1, device=device)
    lbox = torch.zeros(1, device=device)
    lobj = torch.zeros(1, device=device)

    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters

    # Define criteria
    bce_cls = nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    bce_obj = nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor([h['obj_pw']])).to(device)

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_bce(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        bce_cls, bce_obj = FocalLoss(bce_cls, g), FocalLoss(bce_obj, g)

    # Losses
    nt = 0  # number of targets
    ouputs_num = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if ouputs_num == 3 else [
        4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    balance = [4.0, 1.0, 0.5, 0.4, 0.1] if ouputs_num == 5 else balance
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            # giou(prediction, target)
            giou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, ci_ou=True)
            lbox += (1.0 - giou).mean()  # giou loss
            # Objectness, giou ratio
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * \
                giou.detach().clamp(0).type(tobj.dtype)
            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += bce_cls(ps[:, 5:], t)  # BCE

        lobj += bce_obj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / ouputs_num  # output count scaling
    lbox *= h['giou'] * s
    lobj *= h['obj'] * s * (1.4 if ouputs_num >= 4 else 1.)
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


def build_targets(p, targets, model):
    """Build targets for compute_loss(), input targets(image,class,x,y,w,h)"""

    # Detect() module
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(
        na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    # append anchor indices
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):
        anchors = det.anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        t = targets * gain
        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(
                r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3].int()), gi.clamp_(
            0, gain[2].int())))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # (pixels) minimum and maximum box width and height
    _, max_wh = 2, 4096
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float(
                ) / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except RuntimeError as e:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(e, x, i, x.shape, i.shape)

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def strip_optimizer(weights="weights/best.pt", dest=""):
    """Strip optimizer from "f" to finalize training, optionally save as "s"""

    x = torch.load(weights, map_location=torch.device("cpu"))
    x["optimizer"] = None
    x["training_results"] = None
    x["epoch"] = -1
    x["model"].half()  # to FP16

    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, dest or weights)
    mb = os.path.getsize(dest or weights) / 1E6  # filesize

    print_args = (weights, (" saved as %s," % dest) if dest else "", mb)
    print("Optimizer stripped from %s,%s %.1fMB" % print_args)


def kmean_anchors(path='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: path to dataset *.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.utils import *; _ = kmean_anchors()
    """
    thr = 1. / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def _fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float(
        ).mean() * n  # best possible recall, anch > thr
        print('thr=%.2f: %.4f best possible recall, %.2f anchors past thr' %
              (thr, bpr, aat))
        print('n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: ' %
              (n, img_size, x.mean(), best.mean(), x[x > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(
                k) - 1 else '\n')  # use in *.cfg
        return k

    # if isinstance(path, str):  # *.yaml file
    #     with open(path, encoding="utf-8") as f:
    #         data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    #     from .datasets import LoadImagesAndLabels
    #     dataset = LoadImagesAndLabels(data_dict["train"],
    #                                   augment=True,
    #                                   rect=True)
    # else:
    dataset = path  # dataset

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate(
        [l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print('WARNING: Extremely small objects found. '
              '%g of %g labels are < 3 pixels in width or height.' % (i, len(wh0)))
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

    # Kmeans calculation
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, _ = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unflitered
    k = print_results(k)

    # Evolve
    npr = np.random
    # fitness, generations, mutation prob, sigma
    f, sh, mp, s = _fitness(k), k.shape, 0.9, 0.1
    # progress bar
    pbar = tqdm(range(gen), desc='Evolving anchors with Genetic Algorithm')
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() *
                 npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = _fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f
            if verbose:
                print_results(k)

    return print_results(k)


def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml'):
    """Print mutation results to evolve.txt (for use with train.py --evolve)"""

    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    c = '%10.4g' * len(results) % results

    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    with open("evolve.txt", "a", encoding="utf-8") as f:  # append result
        f.write(c + b + '\n')

    x = np.unique(np.loadtxt('evolve.txt', ndmin=2),
                  axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort

    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    # Save yaml
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])

    with open(yaml_file, "w", encoding="utf-8") as f:
        results = tuple(x[0, :7])
        # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
        c = '%10.4g' * len(results) % results
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(
            x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)


def fitness(x):
    """Returns fitness (for use with results.txt or evolve.txt)"""

    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def output_to_target(output, width, height):
    """Convert model output to target format [batch_id, class_id, x, y, w, h, conf]"""
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    targets = []
    for i, o in enumerate(output):
        if o is not None:
            for pred in o.cpu():
                box = pred[:4]
                w = (box[2] - box[0]) / width
                h = (box[3] - box[1]) / height
                x = box[0] / width + w / 2
                y = box[1] / height + h / 2
                conf = pred[4]
                cls = int(pred[5])

                targets.append([i, cls, x, y, w, h, conf])

    return np.array(targets)


def increment_dir(path, comment=""):
    """Increments a directory runs/exp1 --> runs/exp2_comment"""

    n = 0  # number
    path = str(Path(path))  # os-agnostic
    d = sorted(glob.glob(path + "*"))  # directories
    if len(d):
        n = max([int(x[len(path):x.find("_") if "_" in x else None])
                for x in d]) + 1  # increment

    return path + str(n) + ("_" + comment if comment else "")


# Plotting functions ---------------------------------------------------------------------------------------------------
def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(
        x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """Plots one bounding box on image img"""

    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    """Draw mosaic"""

    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    if os.path.isfile(fname):  # do not overwrite
        return None

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h): return tuple(
        int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            gt = image_targets.shape[1] == 6  # ground truth if no conf column
            # check for confidence presence (gt vs pred)
            conf = None if gt else image_targets[:, 6]

            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = color_lut[cls % len(color_lut)]
                cls = names[cls] if names else cls
                if gt or conf[j] > 0.3:  # 0.3 conf thresh
                    label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label=label,
                                 color=color, line_thickness=tl)

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w,
                      block_y + h), (255, 255, 255), thickness=3)

    if fname is not None:
        mosaic = cv2.resize(
            mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic


def plot_labels(labels, save_dir=''):
    """Plot dataset labels"""

    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes

    _, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_xlabel('classes')
    ax[1].scatter(b[0], b[1], c=hist2d(b[0], b[1], 90), cmap='jet')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[2].scatter(b[2], b[3], c=hist2d(b[2], b[3], 90), cmap='jet')
    ax[2].set_xlabel('width')
    ax[2].set_ylabel('height')
    plt.savefig(Path(save_dir) / 'labels.png', dpi=200)
    plt.close()


def plot_evolution(yaml_file='runs/evolve/hyp_evolved.yaml'):
    """Plot hyperparameter evolution results in evolve.txt"""

    with open(yaml_file, encoding="utf-8") as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    # weights = (f - f.min()) ** 2  # for weighted results
    plt.figure(figsize=(10, 10), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, _) in enumerate(hyp.items()):
        y = x[:, i + 7]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(5, 5, i + 1)
        plt.scatter(y, f, c=hist2d(y, f, 20), cmap='viridis',
                    alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title('%s = %.3g' % (k, mu), fontdict={
                  'size': 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)
    print('\nPlot saved as evolve.png')


def plot_results(start=0, stop=0, bucket='', _id=(), labels=(), save_dir=''):
    """Plot training 'results*.txt' as seen in https://github.com/ultralytics/yolov3"""

    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    ax = ax.ravel()
    s = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val GIoU', 'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']
    if bucket:
        os.system('rm -rf storage.googleapis.com')
        files = ['https://storage.googleapis.com/%s/results%g.txt' %
                 (bucket, x) for x in _id]
    else:
        files = glob.glob(str(Path(save_dir) / 'results*.txt')) + \
            glob.glob('../../Downloads/results*.txt')  # [!]
    for fi, f in enumerate(files):
        results = np.loadtxt(
            f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        for i in range(10):
            y = results[i, x]
            if i in [0, 1, 2, 5, 6, 7]:
                y[y == 0] = np.nan  # dont show zero loss values
                # y /= y[0]  # normalize
            label = labels[fi] if len(labels) > 0 else Path(f).stem
            ax[i].plot(x, y, marker='.', label=label,
                       linewidth=2, markersize=8)
            ax[i].set_title(s[i])

    fig.tight_layout()
    ax[1].legend()
    fig.savefig(Path(save_dir) / 'results.png', dpi=200)

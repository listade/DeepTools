"""YOLO test module"""

import argparse
import glob
import os
import shutil
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from tqdm import tqdm

from .models.experimental import attempt_load
from .utils.datasets import create_dataloader
from .utils.general import (ap_per_class, box_iou, check_img_size, clip_coords,
                            compute_loss, non_max_suppression,
                            output_to_target, plot_images, scale_coords,
                            xywh2xyxy, xyxy2xywh)
from .utils.torch_utils import select_device, time_synchronized


def test(data: str,
         weights: str,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir='',
         merge=False,
         save_txt=False):

    """Test weights"""

    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(device, batch_size=batch_size)
        if save_txt:
            out = Path("inference/output")
            if os.path.exists(out):
                shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder

        mask = str(Path(save_dir) / "test_batch*.jpg") # remove previous
        for v in glob.glob(mask):
            os.remove(v)

        model = attempt_load(weights, map_location=device)  # load FP32 model
        max_stride = model.stride.max()
        imgsz = check_img_size(imgsz, stride=max_stride)  # check img_size

    half = device.type != "cpu"  # half precision only supported on CUDA
    if half:
        model.half()

    model.eval() # configure

    with open(data, encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    class_num = 1 if single_cls else int(data["nc"])  # number of classes

    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    iouv = iouv.to(device)
    niou = iouv.numel()  # num of elements in tensor

    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != "cpu" else None # run once

        path = data['val'] # path to valid images

        dataloader, _ = create_dataloader(path,
                                          imgsz,
                                          batch_size,
                                          model.stride.max(),
                                          cache=False,
                                          pad=0.5,
                                          rect=True) # dataloader
    seen = 0
    names = model.names if hasattr(model, "names") else model.module.names

    s = ("%20s" + "%12s" * 6) % ("Class", "Images", "Targets", "P", "R", "mAP@.5", "mAP@.5:.95")
    p, r, mp, mr, map50, _map, t0, t1 = .0, .0, .0, .0, .0, .0, .0, .0
    stats, ap, ap_class = [], [], []

    loss = torch.zeros(3, device=device)

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0

        targets = targets.to(device)
        _, _, height, width = img.shape # batch size, channels, height, width

        whwh = torch.Tensor([width, height, width, height]).to(device)

        with torch.no_grad(): # disable gradients
            t = time_synchronized()
            inf_out, train_out = model(img, augment=augment) # inference and training outputs
            t0 += time_synchronized() - t

            if training:  # if model has loss hyperparameters
                loss += compute_loss([x.float()for x in train_out], targets, model)[1][:3] # GIoU, obj, cls

            t = time_synchronized()
            output = non_max_suppression(inf_out,
                                         conf_thres=conf_thres,
                                         iou_thres=iou_thres,
                                         merge=merge) # run NMS
            t1 += time_synchronized() - t

        for si, pred in enumerate(output): # statistics per image
            labels = targets[targets[:, 0] == si, 1:]
            labels_len = len(labels)
            tcls = labels[:, 0].tolist() if labels_len else []  # target class
            seen += 1

            if pred is None:
                if labels_len:
                    stat = (torch.zeros(0, niou, dtype=torch.bool),
                            torch.Tensor(),
                            torch.Tensor(),
                            tcls)
                    stats.append(stat)
                continue

            if save_txt: # append to text file
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]] # normalization gain whwh
                txt_path = str(out / Path(paths[si]).stem)

                pred[:, :4] = scale_coords(img[si].shape[1:],
                                           pred[:, :4],
                                           shapes[si][0],
                                           shapes[si][1])  # to original

                for *xyxy, _, cls in pred:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                    with open(txt_path + ".txt", "a", encoding="utf-8") as f:
                        line = ("%g " * 5 + "\n") % (cls, *xywh)  # label format
                        f.write(line)

            clip_coords(pred, (height, width)) # clip boxes to image bounds

            correct = torch.zeros(pred.shape[0],
                                  niou,
                                  dtype=torch.bool,
                                  device=device) # assign all predictions as incorrect

            if labels_len:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh # target boxes

                for cls in torch.unique(tcls_tensor): # per target class
                    ti = (cls == tcls_tensor) \
                        .nonzero(as_tuple=False) \
                        .view(-1)  # prediction indices

                    pi = (cls == pred[:, 5]) \
                        .nonzero(as_tuple=False) \
                        .view(-1)  # target indices

                    if pi.shape[0]: # search for detections
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # prediction to target ious; best ious, indices
                        detected_set = set() # append detections

                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target

                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)

                                correct[pi[j]] = ious[j] > iouv # iou_thres is 1xn
                                if len(detected) == labels_len: # all targets already located in image
                                    break

            stat = (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)
            stats.append(stat) # append statistics (correct, conf, pcls, tcls)

        if batch_i < 1:
            f = Path(save_dir) / (f"test_batch_{batch_i}_gt.jpg")  # plot images
            plot_images(img,
                        targets,
                        paths,
                        str(f),
                        names)  # ground truth

            f = Path(save_dir) / (f"test_batch_{batch_i}_pred.jpg")
            plot_images(img,
                        output_to_target(output, width, height),
                        paths,
                        str(f),
                        names)  # predictions

    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # compute statistics

    if len(stats) and stats[0].any():
        p, r, ap, _, ap_class = ap_per_class(*stats)
        p = p[:, 0]
        r = r[:, 0]
        ap50 = ap[:, 0]
        ap = ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]

        mp = p.mean()
        mr = r.mean()
        map50 = ap50.mean()
        _map = ap.mean()

        targets_num = np.bincount(stats[3].astype(np.int64),
                                  minlength=class_num) # number of targets per class
    else:
        targets_num = torch.zeros(1)

    print(f"all {seen} {targets_num.sum()} {mp} {mr} {map50} {_map}") # print results

    if verbose and class_num > 1 and len(stats):
        for i, cls in enumerate(ap_class):
            print(f"{names[cls]} {seen} {targets_num[cls]} {p[i]} {r[i]} {ap50[i]} {ap[i]}") # print results per class

    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # print speeds

    if not training:
        print("Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g" % t)

    model.float()  # for training
    maps = np.zeros(class_num) + _map

    for i, cls in enumerate(ap_class):
        maps[cls] = ap[i]

    return (mp, mr, map50, _map, *(loss.cpu() / len(dataloader)).tolist()), maps, t # return results


def study(weights: str,
          data: str,
          batch_size: int,
          conf_thres: float,
          iou_thres: float):
    """Running test over parameters"""

    filename = f"study_{Path(data).stem}_{Path(weights).stem}.txt" # filename to save to

    x = list(range(352, 832, 64))  # x axis
    y = []  # y axis

    for v in x:  # img-size
        print(f"Running {filename} point {v}...")
        r, _, t = test(data, weights, batch_size, v, conf_thres, iou_thres)
        y.append(r + t)  # results and times

    np.savetxt(filename, y, fmt="%10.4g")  # save


def main():
    """Entry point"""

    parser = argparse.ArgumentParser()

    parser.add_argument("--weights",
                        type=str,
                        required=True)

    parser.add_argument("--data",
                        type=str,
                        required=True)

    parser.add_argument("--batch-size",
                        type=int,
                        default=4)

    parser.add_argument("--img-size",
                        type=int,
                        default=640)

    parser.add_argument("--conf-thres",
                        type=float,
                        default=0.001)

    parser.add_argument("--iou-thres",
                        type=float,
                        default=0.65)

    parser.add_argument("--study",
                        type=bool,
                        action="store_true")

    parser.add_argument("--device",
                        type=str,
                        default="cuda")

    parser.add_argument("--single-cls",
                        action="store_true")

    parser.add_argument("--augment",
                        action="store_true")

    parser.add_argument("--merge",
                        action="store_true",)

    parser.add_argument("--verbose",
                        action="store_true")

    parser.add_argument("--save-txt",
                        action="store_true")

    opt = parser.parse_args()

    if opt.study:  # run over a range of settings and save/plot
        study(opt.weights,
              opt.data,
              opt.batch_size,
              opt.conf_thres,
              opt.iou_thres)
        sys.exit()

    test(opt.data,
         opt.weights,
         opt.batch_size,
         opt.img_size,
         opt.conf_thres,
         opt.iou_thres,
         opt.single_cls,
         opt.augment,
         opt.verbose)


if __name__ == '__main__':
    main()

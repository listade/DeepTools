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


def test(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir='',
         merge=False,
         save_txt=False):
    """Test weights"""

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(opt.device, batch_size=batch_size)
        merge = opt.merge
        save_txt = opt.save_txt  # use Merge NMS, save *.txt labels

        if save_txt:
            out = Path("inference/output")
            if os.path.exists(out):
                shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder

        # Remove previous
        mask = str(Path(save_dir) / "test_batch*.jpg")
        for v in glob.glob(mask):
            os.remove(v)

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        max_stride = model.stride.max()
        imgsz = check_img_size(imgsz, stride=max_stride)  # check img_size

    # Half
    half = device.type != "cpu"  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()

    with open(data, encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    class_num = 1 if single_cls else int(data["nc"])  # number of classes

    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    iouv = iouv.to(device)
    niou = iouv.numel()  # num of elements in tensor

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

        # run once
        _ = model(img.half() if half else img) if device.type != "cpu" else None

        # path to val/test images
        path = data['test'] if opt.task == 'test' else data['val']

        dataloader, _ = create_dataloader(path,
                                          imgsz,
                                          batch_size,
                                          model.stride.max(),
                                          cache=False,
                                          pad=0.5,
                                          rect=True)

    seen = 0
    names = model.names if hasattr(model, "names") else model.module.names

    s = ("%20s" + "%12s" * 6) % ("Class", 
                                 "Images",
                                 "Targets", 
                                 "P", 
                                 "R", 
                                 "mAP@.5", 
                                 "mAP@.5:.95")

    p, r, mp, mr, map50, _map, t0, t1 = .0, .0, .0, .0, .0, .0, .0, .0
    stats, ap, ap_class = [], [], []

    loss = torch.zeros(3, device=device)

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32

        # 0 - 255 to 0.0 - 1.0
        img /= 255.0
        targets = targets.to(device)

        # batch size, channels, height, width
        _, _, height, width = img.shape
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()

            # inference and training outputs
            inf_out, train_out = model(img, augment=augment)
            t0 += time_synchronized() - t

            # Compute loss
            if training:  # if model has loss hyperparameters
                # GIoU, obj, cls
                loss += compute_loss([x.float()
                                     for x in train_out], targets, model)[1][:3]

            # Run NMS
            t = time_synchronized()
            output = non_max_suppression(inf_out,
                                         conf_thres=conf_thres,
                                         iou_thres=iou_thres,
                                         merge=merge)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
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

            # Append to text file
            if save_txt:
                # normalization gain whwh
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]
                txt_path = str(out / Path(paths[si]).stem)

                pred[:, :4] = scale_coords(img[si].shape[1:],
                                           pred[:, :4],
                                           shapes[si][0],
                                           shapes[si][1])  # to original

                for *xyxy, _, cls in pred:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn) \
                        .view(-1) \
                        .tolist()  # normalized xywh

                    with open(txt_path + ".txt", "a", encoding="utf-8") as f:
                        line = ("%g " * 5 + "\n") % (cls, *xywh)  # label format
                        f.write(line)

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0],
                                  niou,
                                  dtype=torch.bool,
                                  device=device)

            if labels_len:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor) \
                        .nonzero(as_tuple=False) \
                        .view(-1)  # prediction indices

                    pi = (cls == pred[:, 5]) \
                        .nonzero(as_tuple=False) \
                        .view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(
                            1)  # best ious, indices

                        # Append detections
                        detected_set = set()

                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target

                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)

                                # iou_thres is 1xn
                                correct[pi[j]] = ious[j] > iouv

                                # all targets already located in image
                                if len(detected) == labels_len:
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stat = (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)
            stats.append(stat)

        # Plot images
        if batch_i < 1:
            f = Path(save_dir) / (f"test_batch_{batch_i}_gt.jpg")  # filename
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

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

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

        # number of targets per class
        targets_num = np.bincount(
            stats[3].astype(np.int64), minlength=class_num)
    else:
        targets_num = torch.zeros(1)

    # Print results
    print(f"all {seen} {targets_num.sum()} {mp} {mr} {map50} {_map}")

    # Print results per class
    if verbose and class_num > 1 and len(stats):
        for i, cls in enumerate(ap_class):
            print(
                f"{names[cls]} {seen} {targets_num[cls]} {p[i]} {r[i]} {ap50[i]} {ap[i]}")

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + \
        (imgsz, imgsz, batch_size)  # tuple

    if not training:
        print("Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g" % t)

    # Return results
    model.float()  # for training
    maps = np.zeros(class_num) + _map

    for i, cls in enumerate(ap_class):
        maps[cls] = ap[i]

    return (mp, mr, map50, _map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def study(weights, data, batch_size, conf_thres, iou_thres):
    """Study"""
    for w in weights:  # [!]
        # filename to save to
        filename = f"study_{Path(opt.data).stem}_{Path(w).stem}.txt"

        x = list(range(352, 832, 64))  # x axis
        y = []  # y axis

        for v in x:  # img-size
            print(f"Running {filename} point {v}...")

            r, _, t = test(data,
                           w,
                           batch_size,
                           v,
                           conf_thres,
                           iou_thres)
            y.append(r + t)  # results and times
        np.savetxt(filename, y, fmt="%10.4g")  # save


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights",
                        type=str,
                        nargs="+",
                        required=True,
                        help="model.pt path(s)")

    parser.add_argument("--data",
                        type=str,
                        required=True,
                        help="data.yml path")

    parser.add_argument("--batch-size",
                        type=int,
                        default=4,
                        help="size of each image batch")

    parser.add_argument("--img-size",
                        type=int,
                        default=640,
                        help="inference size (px)")

    parser.add_argument("--conf-thres",
                        type=float,
                        default=0.001,
                        help="object confidence threshold")

    parser.add_argument("--iou-thres",
                        type=float,
                        default=0.65,
                        help="IOU threshold for NMS")

    parser.add_argument("--task",
                        type=str,
                        default="val",
                        help="'val', 'test', 'study'")

    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="cuda device, i.e. 0 or 0,1,2,3 or cpu")

    parser.add_argument("--single-cls",
                        action="store_true",
                        help="treat as single-class dataset")

    parser.add_argument("--augment",
                        action="store_true",
                        help="augmented inference")

    parser.add_argument("--merge",
                        action="store_true",
                        help="use Merge NMS")

    parser.add_argument("--verbose",
                        action="store_true",
                        help="report mAP by class")

    parser.add_argument("--save-txt",
                        action="store_true",
                        help="save results to *.txt")

    opt = parser.parse_args()

    if opt.task in ("val", "test"):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.single_cls,
             opt.augment,
             opt.verbose)
        sys.exit()

    if opt.task == "study":  # run over a range of settings and save/plot
        study(opt.weights,
              opt.data,
              opt.batch_size,
              opt.conf_thres,
              opt.iou_thres)

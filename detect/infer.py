import argparse
import configparser
import os
import warnings

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from .utils.datasets import LoadImages_
from .utils.general import non_max_suppression, plot_one_box, scale_coords

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default.ini")
    opt = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(opt.config)

    net = config["Net"]
    device = net["Device"]

    pre = config["Pre"]
    width = int(pre["Width"])
    overlap = int(pre["Overlap"])

    infer = config["Infer"]
    input_dir = infer["InputDir"]
    output_dir = infer["OutputDir"]
    weights = infer["WeightsPath"]
    conf_thres = float(infer["ConfThres"])
    iou_thres = float(infer["IouThres"])

    device = torch.device(device)
    dataset = LoadImages_(input_dir, img_size=width)

    with torch.no_grad():
        model = torch.load(weights, map_location=device)['model'].float().fuse().eval()  # load FP32 model

        for path, img_np in dataset:
            im_height, im_width = img_np.shape[:-1]
            preds = torch.zeros((0, 6), dtype=torch.float32).to(device)

            for hstep in range(0, im_height, width-overlap):
                for wstep in range(0, im_width, width-overlap):

                    im_tile = img_np[hstep:hstep+width, wstep:wstep+width, :] # tile
                    tile_height, tile_width = im_tile.shape[:-1]

                    if tile_width < width and tile_height == width:
                        im_tile = np.pad(im_tile, [(0, 0), (0, width-tile_width), (0, 0)], mode='constant', constant_values=114)
                    elif tile_height < width and tile_width == width:
                        im_tile = np.pad(im_tile, [(0, width-tile_height), (0, 0), (0, 0)], mode='constant', constant_values=114)
                    elif tile_width < width and tile_height < width:
                        im_tile = np.pad(im_tile, [(0, width-tile_height), (0, width-tile_width), (0, 0)], mode='constant', constant_values=114)
                    elif tile_width > width or tile_height > width:
                        raise ValueError('dimension mismatch: cropped image size')

                    init_im_size = im_tile.shape[:-1]  # initial crop size 1024x1024
                    im_tile = cv2.resize(im_tile, (640, 640))

                    # Convert
                    im_tile = im_tile[:, :, ::-1]  # BGR to RGB
                    im_tile = np.ascontiguousarray(im_tile)

                    im_tile = torch.from_numpy(im_tile).to(device)
                    im_tile = im_tile.float()  # uint8 to fp16/32
                    im_tile /= 255.0  # 0 - 255 to 0.0 - 1.0
                    im_tile = im_tile.permute(2, 0, 1)

                    if im_tile.ndimension() == 3:
                        im_tile = im_tile.unsqueeze(0)

                    # Inference
                    pred = model(im_tile, augment=False)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)

                    if pred[0] is not None:
                        # rescaling coordinates from 640 to 1024
                        pred[0][:, :4] = scale_coords(im_tile.shape[2:],  # torch tensor
                                                    pred[0][:, :4],
                                                    init_im_size).round()

                        # setting bbox coords to the corresponding crop windows
                        pred[0][:, 0] = pred[0][:, 0] + wstep
                        pred[0][:, 1] = pred[0][:, 1] + hstep
                        pred[0][:, 2] = pred[0][:, 2] + wstep
                        pred[0][:, 3] = pred[0][:, 3] + hstep

                        # gathering all coordinates in one tensor
                        preds = torch.cat((preds, pred[0]), dim=0)

            dets = preds[nms(preds[:, :4], iou_threshold=0.1, scores=preds[:, 4])]
            dets = dets.cpu().numpy()
            dets[:, [4, 5]] = dets[:, [5, 4]]

            file = os.path.join(output_dir, os.path.basename(path))
            np.savetxt(file + '.txt', dets, fmt=('%d', '%d', '%d', '%d', '%d', '%1.3f')) # x1 y1 x2 y2 cls score

            for v in dets:
                plot_one_box(v[:4], img_np, line_thickness=1)
            cv2.imwrite(file, img_np)

"""Object detection main module"""

import argparse
import glob
import os
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.ops import nms

from .utils.general import non_max_suppression, plot_one_box, scale_coords

warnings.filterwarnings("ignore", category=UserWarning)


class ImagesDataset(Dataset):
    """Images numpy arrays dataset"""

    def __init__(self, path="."):
        mask = os.path.join(path, "*.jpg")
        self.files = glob.glob(mask)

    def __getitem__(self, i):
        path = self.files[i]
        print(f"[{i+1}/{len(self)}] {path}")

        img_np = cv2.imread(path)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        return path, img_np

    def __len__(self):
        return len(self.files)


def main(opt):
    """Inference running"""

    device = torch.device(opt.device)
    dataset = ImagesDataset(opt.input)

    with torch.no_grad():
        model = torch.load(opt.weights, map_location=device)
        model = model["model"].float().fuse().eval()  # load FP32 model

        for path, img_np in dataset:
            im_height, im_width = img_np.shape[:-1]
            preds = torch.zeros((0, 6), dtype=torch.float32).to(device)

            for hstep in range(0, im_height, opt.img_size-opt.overlap):
                for wstep in range(0, im_width, opt.img_size-opt.overlap):
                    im_tile = img_np[hstep:hstep+opt.img_size, wstep:wstep+opt.img_size, :] # tile
                    tile_height, tile_width = im_tile.shape[:-1]

                    if tile_width < opt.img_size and tile_height == opt.img_size:
                        im_tile = np.pad(im_tile,
                                         [
                                            (0, 0),
                                            (0, opt.img_size-tile_width),
                                            (0, 0)
                                         ],
                                         mode="constant",
                                         constant_values=114)
                    elif tile_height < opt.img_size and tile_width == opt.img_size:
                        im_tile = np.pad(im_tile,
                                         [
                                            (0, opt.img_size-tile_height),
                                            (0, 0),
                                            (0, 0)
                                         ],
                                         mode="constant",
                                         constant_values=114)
                    elif tile_width < opt.img_size and tile_height < opt.img_size:
                        im_tile = np.pad(im_tile,
                                         [
                                            (0, opt.img_size-tile_height),
                                            (0, opt.img_size-tile_width),
                                            (0, 0)
                                         ],
                                         mode="constant",
                                         constant_values=114)
                    elif tile_width > opt.img_size or tile_height > opt.img_size:
                        raise ValueError("dimension mismatch: cropped image size")

                    init_im_size = im_tile.shape[:-1]  # initial crop size 1024x1024
                    im_tile = cv2.resize(im_tile, (640, 640))
                    im_tile = np.ascontiguousarray(im_tile)
                    im_tile = torch.from_numpy(im_tile).to(device)
                    im_tile = im_tile.float()  # uint8 to fp16/32
                    im_tile /= 255.0  # 0 - 255 to 0.0 - 1.0
                    im_tile = im_tile.permute(2, 0, 1)

                    if im_tile.ndimension() == 3:
                        im_tile = im_tile.unsqueeze(0)

                    pred = model(im_tile, augment=False)[0]
                    pred = non_max_suppression(pred,
                                               conf_thres=opt.conf_thres,
                                               iou_thres=opt.iou_thres)
                    if pred[0] is not None:
                        # rescaling coordinates from 640 to 1024
                        pred[0][:, :4] = scale_coords(im_tile.shape[2:], # torch tensor
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

            file = os.path.join(opt.output, os.path.basename(path))
            fmt = ("%d", "%d", "%d", "%d", "%d", "%1.3f")

            np.savetxt(file.replace(".jpg", ".txt"), dets, fmt=fmt) # x1 y1 x2 y2 cls score

            for det in dets:
                plot_one_box(det[:4], img_np, line_thickness=1)

            cv2.imwrite(file, img_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default="input", metavar="<path-to-images>")
    parser.add_argument("--output", type=str, default="output", metavar="<path-to-txt>")
    parser.add_argument("--weights", type=str, default="yolov4-p5.pt", metavar="<path-to-*.pt>")
    parser.add_argument("--device", type=str, default="cuda", metavar="<cuda|cpu>")

    parser.add_argument("--conf-thres", type=float, default=0.5, metavar="<0-1.0>")
    parser.add_argument("--iou-thres", type=float, default=0.1, metavar="<0-1.0>")
    parser.add_argument("--img-size", type=int, default=640, metavar="<px>")
    parser.add_argument("--overlap", type=int, default=100, metavar="<px>")

    main(parser.parse_args())

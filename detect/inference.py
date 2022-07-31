"""Object detection module"""

import argparse
import glob
import os
import warnings

import cv2
import numpy as np
import tiler
import torch
from torch.utils.data import Dataset
from torchvision.ops import nms

from .utils.torch_utils import select_device
from .utils.general import non_max_suppression, plot_one_box

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

    device = select_device(opt.device)
    dataset = ImagesDataset(opt.input)

    with torch.no_grad():
        weights = torch.load(opt.weights, map_location=device)
        model = weights["model"]
        infer = model.float().fuse().eval()  # load FP32 model

        for path, img_np in dataset:
            total = torch.zeros((0, 6), dtype=torch.float32).to(device)
            img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
            tiler_obj = tiler.Tiler(data_shape=img_np.shape,
                                    tile_shape=(3, opt.img_size, opt.img_size),
                                    channel_dimension=0,
                                    overlap=opt.overlap)
            tiles = tiler_obj(img_np)
            for i, im_tile in tiles:
                im_tile = np.ascontiguousarray(im_tile)
                im_tile = torch.from_numpy(im_tile).float().to(device)
                im_tile /= 255.0 # 0-255 to 0.0-1.0
                if im_tile.ndimension() == 3:
                    im_tile = im_tile.unsqueeze(0)

                res = infer(im_tile, augment=opt.augment)
                pred = res[0]
                nms_pred = non_max_suppression(pred,
                                               conf_thres=opt.conf_thres,
                                               iou_thres=opt.iou_thres)
                dets = nms_pred[-1]
                if dets is None:
                    continue
                (x,y), _ = tiler_obj.get_tile_bbox(i)

                dets[:,0] = dets[:,0] + y
                dets[:,1] = dets[:,1] + x
                dets[:,2] = dets[:,2] + y
                dets[:,3] = dets[:,3] + x

                total = torch.cat((total, dets), dim=0) # gathering all coordinates in one tensor

            total = total[nms(total[:,:4], iou_threshold=opt.iou_thres, scores=total[:,4])]
            np_total = total.cpu().numpy()
            img_np = img_np.transpose(1,2,0)  # CHW -> HWC

            img = os.path.join(opt.output, os.path.basename(path))

            if opt.save_img:
                for det in np_total:
                    bbox = det[:4].round()
                    score = det[4]
                    plot_one_box(bbox, img_np, label="{:.3f}".format(score), line_thickness=2)
                cv2.imwrite(img, img_np)

            ext = img.split(".")[-1]
            txt = img.replace(ext, "txt")

            np_total[:,[4,5]] = np_total[:,[5,4]] # score cls -> cls score
            np.savetxt(txt, np_total, fmt=("%d", "%d", "%d", "%d", "%d", "%1.3f"))


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

    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--save-img", action="store_true")

    main(parser.parse_args())

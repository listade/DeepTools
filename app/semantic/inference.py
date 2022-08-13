"""This module runs semantic segmentation inference"""

import argparse
import glob
import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from tiler import Merger, Tiler
from torch.utils.data import Dataset

from .model import SegModel


class ImagesDataset(Dataset):
    """Dataset of numpy images"""

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
    """Module entry point"""

    device = torch.device(opt.device)
    state_dict = torch.load(opt.weights)
    model = SegModel("UnetPlusPlus",
                     "resnext50_32x4d",
                     in_channels=3,
                     out_classes=1)

    model.load_state_dict(state_dict["state_dict"])
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        prep = smp.encoders.get_preprocessing_fn(encoder_name="resnext50_32x4d",
                                                 pretrained="imagenet")
        dataset = ImagesDataset(opt.input)

        for image, img_np in dataset:
            img_res = cv2.resize(img_np,
                                 None,
                                 fx=opt.shrink,
                                 fy=opt.shrink,
                                 interpolation=cv2.INTER_AREA)

            img_res = prep(img_res)
            img_res = img_res.transpose(2, 0, 1)  # HWC -> CHW

            tiler = Tiler(data_shape=img_res.shape,
                          tile_shape=(3, opt.img_size, opt.img_size),
                          channel_dimension=0,
                          overlap=opt.overlap)

            mtiler = Tiler(data_shape=img_res.shape[1:],
                           tile_shape=(opt.img_size, opt.img_size),
                           channel_dimension=None,
                           overlap=opt.overlap)

            mmerger = Merger(mtiler)

            for i, batch in tiler(img_res, batch_size=opt.batch_size):
                batch = torch.from_numpy(batch).float().to(device)
                logits = model.forward(batch)
                pr_masks = logits.sigmoid().squeeze().cpu().numpy()

                mmerger.add_batch(i, opt.batch_size, pr_masks)

            height, width = img_np.shape[:-1]

            mask = mmerger.merge(unpad=True)
            mask = cv2.resize(mask, (width, height))

            mask = (mask > opt.conf_thres).astype(np.uint8)
            img_np = cv2.bitwise_and(img_np, img_np, mask=mask)

            img_in = os.path.basename(image).replace(".jpg", ".png")
            img_out = os.path.join(opt.output, img_in)

            cv2.imwrite(img_out, img_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input",
                        type=str,
                        required=True,
                        metavar="<path>")

    parser.add_argument("--output",
                        type=str,
                        default=".",
                        metavar="<path>")

    parser.add_argument("--weights",
                        type=str,
                        required=True,
                        metavar="<path>")

    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        metavar="<cuda|cpu>")

    parser.add_argument("--batch-size",
                        type=int,
                        default=2,
                        metavar="<int>")

    parser.add_argument("--img-size",
                        type=int,
                        default=640,
                        metavar="<px>")

    parser.add_argument("--overlap",
                        type=int,
                        default=100,
                        metavar="<px>")

    parser.add_argument("--shrink",
                        type=float,
                        default=0.85,
                        metavar="(.0-1.0)")

    parser.add_argument("--conf-thres",
                        type=float,
                        default=0.7,
                        metavar="(.0-1.0)")

    main(parser.parse_args())

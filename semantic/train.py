"""Semantic inference training module"""

import argparse
import glob
import os
import warnings

import cv2
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from tiler import Tiler
from torch.utils.data import DataLoader, IterableDataset

from .model import (SegModel, get_preprocessing, get_training_augmentation,
                    get_validation_augmentation)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Dataset(IterableDataset):
    """Iterable image dataset with tilling"""

    def __init__(self, path=".",
                 width=640,
                 overlap=100,
                 shrink=0.85,
                 fill=0.05,
                 encoder="resnext50_32x4d",
                 encoder_weights="imagenet",
                 augment=None):

        mask = os.path.join(path, "*.jpg")
        prep_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

        self.files = glob.glob(mask)
        self.width = width
        self.overlap = overlap
        self.shrink = shrink
        self.fill = fill
        self.augment = augment
        self.prep = get_preprocessing(prep_fn)

    def __iter__(self):
        for _, path in enumerate(self.files):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, None, fx=self.shrink,
                             fy=self.shrink, interpolation=cv2.INTER_AREA)

            mask = path.replace(".jpg", ".png")

            msk = cv2.imread(mask)
            msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
            msk = cv2.resize(msk, None, fx=self.shrink,
                             fy=self.shrink, interpolation=cv2.INTER_AREA)

            msk = (msk > 0).astype(np.uint8)*255
            alpha = np.dstack((img, msk))

            tiler = Tiler(data_shape=alpha.shape,
                          tile_shape=(self.width, self.width, 4),
                          channel_dimension=2,
                          overlap=self.overlap)

            for _, arr in tiler(alpha):
                tile_area = arr[:, :, -1].sum()

                if (tile_area / self.width**2) >= self.fill:
                    img_tile = arr[:, :, :3]
                    msk_tile = arr[:, :, -1]
                    msk_tile[msk_tile > 0] = 1
                    msk_tile = np.expand_dims(msk_tile, axis=-1)

                    sample = dict(image=img_tile, mask=msk_tile)

                    if self.augment is not None:
                        sample = self.augment(**sample)
                        sample = self.prep(**sample)

                    yield sample

    def __getitem__(self, _):
        raise NotImplementedError()


def main(opt):
    """Module entry point"""

    train_dataset = Dataset(path=opt.train,
                            width=opt.img_size,
                            overlap=opt.overlap,
                            shrink=opt.shrink,
                            fill=opt.fill,
                            encoder=opt.encoder,
                            encoder_weights=opt.encoder_weights,
                            augment=get_training_augmentation())

    valid_dataset = Dataset(path=opt.valid,
                            width=opt.img_size,
                            overlap=opt.overlap,
                            shrink=opt.shrink,
                            fill=opt.fill,
                            encoder=opt.encoder,
                            encoder_weights=opt.encoder_weights,
                            augment=get_validation_augmentation())

    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batch_size,
                              num_workers=opt.workers)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=opt.batch_size,
                              num_workers=opt.workers)

    checkpoint = ModelCheckpoint(save_top_k=1,
                                 every_n_epochs=1,
                                 monitor="valid_dataset_iou",
                                 mode="max",
                                 dirpath=".",
                                 filename=opt.name)

    gpus = 1 if opt.device == "cuda" else None

    trainer = pl.Trainer(gpus=gpus,
                         max_epochs=opt.epochs,
                         callbacks=[checkpoint])

    device = torch.device(opt.device)

    model = SegModel(opt.arch,
                     opt.encoder,
                     in_channels=3,
                     out_classes=opt.nc)
    model.to(device)

    trainer.fit(model.to(device),
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str, default="train",
                        metavar="<path-to-images>")
    parser.add_argument("--valid", type=str, default="valid",
                        metavar="<path-to-images>")
    parser.add_argument("--nc", type=int, default=1, metavar="<classes>")
    parser.add_argument("--name", type=str, default="weights", metavar="<str>")
    parser.add_argument("--batch-size", type=int, default=4, metavar="<int>")
    parser.add_argument("--device", type=str,
                        default="cuda", metavar="<cuda|cpu>")
    parser.add_argument("--epochs", type=int, default=300, metavar="<int>")
    parser.add_argument("--arch", type=str,
                        default="UnetPlusPlus", metavar="<str>")
    parser.add_argument("--encoder", type=str,
                        default="resnext50_32x4d", metavar="<str>")
    parser.add_argument("--encoder-weights", type=str,
                        default="imagenet", metavar="<str>")
    parser.add_argument("--img-size", type=int, default=640, metavar="<px>")
    parser.add_argument("--overlap", type=int, default=100, metavar="<px>")
    parser.add_argument("--shrink", type=float,
                        default=0.85, metavar="<0-1.0>")
    parser.add_argument("--fill", type=float, default=0.05, metavar="<0-1.0>")
    parser.add_argument("--workers", type=int, default=4, metavar="<int>")

    main(parser.parse_args())

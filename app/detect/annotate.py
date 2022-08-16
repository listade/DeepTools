"""The annotator module for bboxes"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
from tiler import Tiler


class Annotator:
    """Bounding box annotator"""
    base_w, base_h = 1024, 768

    def __init__(self, img, winname="cv2"):
        if isinstance(img, str):
            self._img_np = cv2.imread(img)
            if self._img_np is None:
                raise ValueError()

        elif isinstance(img, np.ndarray):
            self._img_np = img
        else:
            raise TypeError()

        h, w, _ = self._img_np.shape
        scale = (self.base_h / h)

        self._win = winname
        self._draw = False
        self._x0, self._y0 = 0, 0
        self._tmp = None
        self._bboxes = []

        def on_mouse_event(event, x, y, *_):
            """Mouse events handler"""

            cv2.setWindowTitle(self._win, f"{self._win} | X {x} Y {y}")
            if event == cv2.EVENT_LBUTTONDOWN:
                self._draw = True
                self._x0, self._y0 = x, y
                self._tmp = self._img_np.copy()

            elif event == cv2.EVENT_MOUSEMOVE:
                if self._draw is True:
                    w = abs(x - self._x0)
                    h = abs(y - self._y0)

                    title = f"{self._win} | X {self._x0} Y {self._y0}  W {w} H {h}"
                    cv2.setWindowTitle(self._win, title)

                    self._img_np = self._tmp.copy()
                    cv2.rectangle(self._img_np,
                                  (self._x0, self._y0),
                                  (x, y),
                                  color=(50, 200, 50),
                                  thickness=12)

            elif event == cv2.EVENT_LBUTTONUP:
                self._draw = False
                self._bboxes.append((self._x0, self._y0, x, y))

        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._win, int(w * scale), int(h * scale))
        cv2.setMouseCallback(self._win, on_mouse_event)


    def get_bboxes(self):
        """Returns bboxes list"""
        while cv2.getWindowProperty(self._win, cv2.WND_PROP_VISIBLE) > 0:
            cv2.imshow(self._win, self._img_np)
            cv2.waitKey(20)

        return self._bboxes[:]


    def save_bboxes(self, filename):
        """Save bboxes to file"""
        lines = ["%d %.1f %.1f %.1f %.1f\n" % tuple((0, *v)) for v in self._bboxes]

        with open(filename, "a", encoding="utf-8") as f:
            f.writelines(lines)


def crop_image(img, size, overlap=100):
    """Crop image and save tiles"""
    img_np = cv2.imread(img)
    tiler = Tiler(data_shape=img_np.shape,
                tile_shape=(size, size, 3),
                channel_dimension=2,
                overlap=overlap)
    path = Path(img)
    for i, tile in tiler(img_np):
        name = str(path).replace(path.stem, f"{path.stem}_{i}")
        cv2.imwrite(name, tile)
        yield name


def main(args):
    """Entry point"""
    imgs = []

    for v in args.images:
        imgs += glob.glob(v, recursive=True)

    if opt.crop:
        cropd = []
        for v in imgs[:]:
            cropd += list(crop_image(v, args.crop))
            imgs.remove(v)
        imgs += cropd


    for i, v in enumerate(imgs):
        path = Path(v)
        img = str(path)

        antor = Annotator(img, winname=f"{path.name} ({i+1}/{len(imgs)})")
        antor.get_bboxes()
        antor.save_bboxes(img.replace(path.suffix, ".txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("images",
                        type=str,
                        nargs="+")

    parser.add_argument("--crop",
                        type=int)

    opt = parser.parse_args()
    main(opt)

"""The annotator module for bboxes"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

BASE_WIDTH, BASE_HEIGHT = 1024, 768

draw = False
img_np = None
tmp = None
x0, y0 = 0, 0


def on_set_mouse(event, x, y, *_):
    """Mouse events handler"""
    global draw, img_np, tmp, x0, y0

    cv2.setWindowTitle(WINDOW, f"{WINDOW} | X {x} Y {y}")

    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        x0, y0 = x, y
        tmp = img_np.copy()

    elif event == cv2.EVENT_MOUSEMOVE:
        if draw == True:
            w = abs(x - x0)
            h = abs(y - y0)

            title = f"{WINDOW} | X {x0} Y {y0}  W {w} H {h}"
            cv2.setWindowTitle(WINDOW, title)

            img_np = tmp.copy()
            cv2.rectangle(img_np,
                          (x0, y0),
                          (x, y),
                          color=(50, 200, 50),
                          thickness=12)

    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
        txt = img.replace(path.suffix, ".txt")
        bbox = np.array((0, float(x0), float(y0), float(x), float(y)))  # [!]
        line = "%d %.1f %.1f %.1f %.1f" % tuple(bbox)

        with open(txt, "a", encoding="utf-8") as f:
            print(line, file=f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("images",
                        type=str,
                        nargs="+")

    parser.add_argument("--img-size",
                        default=640,
                        type=int)

    opt = parser.parse_args()
    imgs = []

    for v in opt.images:
        imgs += glob.glob(v, recursive=True)

    for i, v in enumerate(imgs):
        path = Path(v)
        img = str(path)
        img_np = cv2.imread(img)

        if img_np is None:
            continue

        H, W, _ = img_np.shape
        scale = (BASE_HEIGHT / H)
        WINDOW = f"{path.name} ({i+1}/{len(imgs)})"

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, int(W * scale), int(H * scale))
        cv2.setMouseCallback(WINDOW, on_set_mouse)

        while cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) > 0:
            cv2.imshow(WINDOW, img_np)
            cv2.waitKey(20)

"""The annotator module for bboxes"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

BASE_WIDTH, BASE_HEIGHT = 1024, 768

mouse_down = False
img_np = None
tmp = None
win_name = ""
x0, y0 = 0, 0


def on_set_mouse(event, x, y, *_):
    """Mouse events handler"""
    global mouse_down, img_np, tmp, x0, y0

    cv2.setWindowTitle(win_name, f"{win_name} x:{x} y:{y}")

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down = True
        x0, y0 = x, y
        tmp = img_np.copy()

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_down == True:
            w = abs(x - x0)
            h = abs(y - y0)

            title = f"{win_name} x: {x0} y: {y0} w: {w} h: {h}"
            cv2.setWindowTitle(win_name, title)

            img_np = tmp.copy()
            cv2.rectangle(img_np,
                          (x0, y0),
                          (x, y),
                          color=(50, 200, 50),
                          thickness=12)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_down = False
        txt = img.replace(path.suffix, ".txt")
        write_bbox(np.array((x0, y0, x, y)), txt)


def write_bbox(bbox, file):
    """Save bbox to file"""

    with open(file, "a", encoding="utf-8") as f:
        f.write(" ".join(str(v) for v in bbox))
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("images",
                        type=str,
                        nargs="+")

    opt = parser.parse_args()
    imgs = []

    for v in opt.images:
        imgs += glob.glob(v, recursive=True)

    for v in imgs:
        path = Path(v)
        img = str(path)
        img_np = cv2.imread(img)

        if img_np is None:
            continue

        h, w, _ = img_np.shape
        win_name = f"{path.name} {w}x{h}"

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        scale = (BASE_HEIGHT / h)

        cv2.resizeWindow(win_name, int(w * scale), int(h * scale))
        cv2.setMouseCallback(win_name, on_set_mouse)

        while cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) > 0:
            cv2.imshow(win_name, img_np)
            cv2.waitKey(20)

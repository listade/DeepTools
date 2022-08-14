"""The annotator module for bboxes"""


import argparse
from pathlib import Path

import cv2
import numpy as np

WIN_WIDTH = 1024

mouse_down = False
img = None
tmp = None
window_name = ""
x0, y0 = 0, 0


def on_set_mouse(event, x, y, *_):
    """Mouse events handler"""
    global mouse_down, img, tmp, x0, y0

    cv2.setWindowTitle(window_name, f"{window_name} x:{x} y:{y}")

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down = True
        x0, y0 = x, y
        tmp = img.copy()

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_down == True:
            w = abs(x - x0)
            h = abs(y - y0)
            cv2.setWindowTitle(window_name, f"{window_name} x: {x0} y: {y0} w: {w} h: {h}")

            img = tmp.copy()
            cv2.rectangle(img, (x0, y0), (x, y), color=(50,200,50), thickness=12)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_down = False
        txt_file = str(path).replace(path.suffix, ".txt")
        write_bbox(np.array((x0, y0, x, y)), txt_file)


def write_bbox(bbox, file):
    """Save bbox to file"""
    with open(file, "a", encoding="utf-8") as f:
        f.write(" ".join(str(v) for v in bbox))
        f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("image", type=str)
    opt = parser.parse_args()

    path = Path(opt.image)
    img_file = str(path)
    img = cv2.imread(img_file)

    h, w, _ = img.shape
    window_name = f"{path.name} {w}x{h}"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WIN_WIDTH, int(h * (WIN_WIDTH / w)))
    cv2.setMouseCallback(window_name, on_set_mouse)

    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
        cv2.imshow(window_name, img)
        cv2.waitKey(20)

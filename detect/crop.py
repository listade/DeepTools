"""Image cropping module"""

import argparse
import glob
import os

import cv2
import numpy as np

from .utils.general import xywh2xyxy, xyxy2xywh


def crop(img_path, txt_path, img_size=640, draw_labels=False):
    """Split images to tiles"""

    img_np = cv2.imread(img_path)
    img_height, img_width, _ = img_np.shape
    true_positive, false_positive, false_negative = 0, 1, 2
    input_xyxy = np.loadtxt(txt_path, ndmin=2)

    xywh = xyxy2xywh(input_xyxy[:, 1:])  # remove cls column
    xywh[:, [0, 2]] /= img_width  # get abs x-coords
    xywh[:, [1, 3]] /= img_height  # get abs y-coords
    xywh = np.hstack([input_xyxy[:, [0]], xywh])  # append cls column

    # assign all FN to TP
    clsmask = xywh[:, 0] == false_negative
    xywh[clsmask, 0] = true_positive
    tpfp = xywh[:, [0]].copy()
    xywh = xywh[:, 1::].copy()

    # converting to pixel coordinates
    xywh[:, [0, 2]] = xywh[:, [0, 2]] * img_width
    xywh[:, [1, 3]] = xywh[:, [1, 3]] * img_height
    xywh = xywh.round().astype(np.int32)
    xyxy = xywh2xyxy(xywh)

    ovrlap_height, ovrlap_width = xywh[:, [2, 3]].max(axis=0)

    # corners of the bboxes
    xmin, ymin = xyxy[:, 0], xyxy[:, 1]
    xmax, ymax = xyxy[:, 2], xyxy[:, 3]

    # bbox centers
    xc, yc = xywh[:, 0], xywh[:, 1]
    index = 0

    for hstep in range(0, img_height, img_size-ovrlap_height):
        for wstep in range(0, img_width, img_size-ovrlap_width):
            # corners of the window
            x1, y1 = wstep, hstep
            x2, y2 = wstep + img_size, hstep + img_size

            bboxkeep = np.array(
                [xc >= x1, yc >= y1, xc <= x2, yc <= y2]).all(axis=0)
            bboxrem = np.array(
                [xmax > x1, ymax > y1, xmin < x2, ymin < y2]).all(axis=0)
            bboxrem = ~bboxkeep
            keep_xyxy = xyxy[bboxkeep]
            rem_xyxy = np.hstack((tpfp[bboxrem], xyxy[bboxrem]))

            # the window contains at least one bbox (full or crossed)
            if keep_xyxy.size > 0:
                img_crop = img_np.copy()[y1:y2, x1:x2, :]
                if img_crop.size == 0:
                    return None
                h, w = img_crop.shape[:-1]
                if h < img_size and w < img_size:
                    img_crop = np.pad(
                        img_crop, ((0, img_size-h), (0, img_size-w), (0, 0)))

                # keep only TP coordinates
                keep_xyxy = keep_xyxy[tpfp.flatten(
                )[bboxkeep] != false_positive]
                keep_xyxy[:, 0] -= x1
                keep_xyxy[:, 2] -= x1
                keep_xyxy[:, 1] -= y1
                keep_xyxy[:, 3] -= y1
                keep_xyxy[:, [0, 2]] = np.clip(keep_xyxy[:, [0, 2]], 0, w)
                keep_xyxy[:, [1, 3]] = np.clip(keep_xyxy[:, [1, 3]], 0, h)

                if rem_xyxy.size > 0:
                    # first column contains cls, add 1
                    rem_xyxy[:, 0+1] -= x1
                    rem_xyxy[:, 2+1] -= x1
                    rem_xyxy[:, 1+1] -= y1
                    rem_xyxy[:, 3+1] -= y1

                    rem_mask = np.zeros_like(img_crop[:, :, 0], dtype=np.uint8)
                    keep_mask = rem_mask.copy()

                    for cls, xmi, ymi, xma, yma in rem_xyxy.astype(int):
                        if cls == true_positive:
                            cv2.rectangle(rem_mask, (xmi, ymi),
                                          (xma, yma), 1, -1)
                    for xmi, ymi, xma, yma in keep_xyxy.astype(int):
                        cv2.rectangle(keep_mask, (xmi, ymi), (xma, yma), 1, -1)

                    rem_mask = rem_mask.astype(int)
                    keep_mask = keep_mask.astype(int)
                    mask = (keep_mask - rem_mask)
                    mask[mask >= 0] = 1
                    mask[mask < 0] = 0
                    mask = mask.astype(np.uint8)
                    img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask)

                new_xywh = xyxy2xywh(keep_xyxy).astype(np.float32)
                new_xywh[:, [0, 2]] = new_xywh[:, [0, 2]] / w
                new_xywh[:, [1, 3]] = new_xywh[:, [1, 3]] / h

                new_xywh = np.hstack(
                    (np.zeros_like(new_xywh[:, [0]]), new_xywh))
                if draw_labels:
                    for xmi, ymi, xma, yma in keep_xyxy.astype(int):
                        cv2.rectangle(img_crop, (xmi, ymi),
                                      (xma, yma), (5, 5, 245), 1)

                fmt = ("%d", "%1.6f", "%1.6f", "%1.6f", "%1.6f")
                txt_file = f"crop_{index:02d}_{os.path.basename(txt_path)}"
                np.savetxt(txt_file, new_xywh, fmt=fmt)

                img_file = f"crop_{index:02d}_{os.path.basename(img_path)}"
                cv2.imwrite(img_file, img_crop, [
                            cv2.IMWRITE_JPEG_QUALITY, 100])

                index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--images", type=str,
                        metavar="<path-to-images>", default=".")
    parser.add_argument("--labels", type=str,
                        metavar="<path-to-txt-labels>", default=".")
    parser.add_argument("--img-size", type=int, metavar="<px>", default=640)
    parser.add_argument("--draw-labels", action="store_true")

    opt = parser.parse_args()

    images = os.path.join(opt.images, "*.jpg")

    for img in glob.glob(images):
        lbl = os.path.basename(img).replace(".jpg", ".txt")
        lbl = os.path.join(opt.labels, lbl)

        crop(img, lbl, img_size=opt.img_size, draw_labels=opt.draw_labels)

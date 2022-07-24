import argparse
import glob
import os

import cv2
import numpy as np

from .utils.general import xywh2xyxy, xyxy2xywh


def crop(img_path, txt_path, img_size=640, draw_labels=False):

    img_np = cv2.imread(img_path) # load image
    H, W, _ = img_np.shape # image dimensions
    TP, FP, FN = 0, 1, 2 # true positive, false positive, false negative
    input_xyxy = np.loadtxt(txt_path, ndmin=2) # load labels

    xywh = xyxy2xywh(input_xyxy[:, 1:]) # remove cls column
    xywh[:, [0, 2]] /= W # get abs x-coords
    xywh[:, [1, 3]] /= H # get abs y-coords
    xywh = np.hstack([input_xyxy[:, [0]], xywh]) # append cls column

    # assign all FN to TP
    clsmask = xywh[:, 0] == FN
    xywh[clsmask, 0] = TP
    tpfp = xywh[:, [0]].copy()
    xywh = xywh[:, 1::].copy()

    # converting to pixel coordinates
    xywh[:, [0, 2]] = xywh[:, [0, 2]] * W
    xywh[:, [1, 3]] = xywh[:, [1, 3]] * H
    xywh = xywh.round().astype(np.int32)
    xyxy = xywh2xyxy(xywh)

    ovrlapH, ovrlapW = xywh[:, [2,3]].max(axis=0)

    #corners of the bboxes
    xmin, ymin = xyxy[:,0], xyxy[:,1]
    xmax, ymax = xyxy[:,2], xyxy[:,3]

    # bbox centers
    xc, yc = xywh[:,0], xywh[:,1]
    i = 1

    for hstep in range(0, H, img_size-ovrlapH):
        for wstep in range(0, W, img_size-ovrlapW):
            # corners of the window
            x1, y1 = wstep, hstep
            x2, y2 = wstep + img_size, hstep + img_size

            bboxkeep = np.array([xc >= x1, yc >= y1, xc <= x2, yc <= y2]).all(axis=0)
            bboxrem = np.array([xmax > x1, ymax > y1, xmin < x2, ymin < y2]).all(axis=0)
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
                    img_crop = np.pad(img_crop, ((0, img_size-h), (0, img_size-w), (0, 0)))

                # keep only TP coordinates
                keep_xyxy = keep_xyxy[tpfp.flatten()[bboxkeep] != FP]
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
                        if cls == TP:
                            cv2.rectangle(rem_mask, (xmi, ymi), (xma, yma), 1, -1)
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

                new_xywh = np.hstack((np.zeros_like(new_xywh[:, [0]]), new_xywh))

                if draw_labels:
                    for xmi, ymi, xma, yma in keep_xyxy.astype(int):
                        cv2.rectangle(img_crop, (xmi, ymi), (xma, yma), (5, 5, 245), 1)

                np.savetxt(f'crop_{i:02d}_{os.path.basename(txt_path)}', new_xywh, fmt=('%d', '%1.6f', '%1.6f', '%1.6f', '%1.6f'))
                cv2.imwrite(f'crop_{i:02d}_{os.path.basename(img_path)}', img_crop, [cv2.IMWRITE_JPEG_QUALITY, 100])

                i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--images", type=str, metavar="<path-to-images>", default=".")
    parser.add_argument("--labels", type=str, metavar="<path-to-txt-labels>", default=".")
    parser.add_argument("--img-size", type=int, metavar="<px>", default=640)
    parser.add_argument("--draw-labels", action='store_true')

    opt = parser.parse_args()

    images = os.path.join(opt.images, "*.jpg")

    for img in glob.glob(images):
        lbl = os.path.basename(img).replace(".jpg", ".txt")
        lbl = os.path.join(opt.labels, lbl)

        crop(img, lbl, img_size=opt.img_size, draw_labels=opt.draw_labels)

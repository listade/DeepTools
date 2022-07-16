import cv2
import numpy as np
import os
import torch

from PIL import Image



def xywh2xyxy(x):
    # Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right)
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    # Transform box coordinates from [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right) to [x, y, w, h] 
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def crop_window(path2img, path2txt, crop=640, img_dir='.', txt_dir='.'):
    with Image.open(path2img) as f:
        img = np.asarray(f)

    H, W, _ = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # cv2 saves BGR to RGB

    TP, FP, FN = 0, 1, 2
    input_xyxy = np.loadtxt(path2txt, ndmin=2)

    if input_xyxy.size == 0:
        return

    xywh = xyxy2xywh(input_xyxy[:, 1:]) # remove cls column
    xywh[:, [0, 2]] /= W # get abs x-coords 
    xywh[:, [1, 3]] /= H # get abs y-coords
    xywh = np.hstack([input_xyxy[:, [0]], xywh]) # append cls column

    if xywh.size == 0:
        return None
    
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
    
    for hstep in range(0, H, crop-ovrlapH):
        for wstep in range(0, W, crop-ovrlapW):
            # corners of the window
            x1, y1 = wstep, hstep
            x2, y2 = wstep + crop, hstep + crop

            bboxkeep = np.array([xc >= x1, yc >= y1, xc <= x2, yc <= y2]).all(axis=0)
            bboxrem = np.array([xmax > x1, ymax > y1, xmin < x2, ymin < y2]).all(axis=0)
            bboxrem = ~bboxkeep
            keep_xyxy = xyxy[bboxkeep]
            rem_xyxy = np.hstack((tpfp[bboxrem], xyxy[bboxrem]))

            # the window contains at least one bbox (full or crossed)
            if keep_xyxy.size > 0:
                cropImg = img.copy()[y1:y2, x1:x2, :]
                if cropImg.size == 0:
                    return None
                h, w = cropImg.shape[:-1]
                if h < crop and w < crop:
                    cropImg = np.pad(cropImg, ((0, crop-h), (0, crop-w), (0, 0)))

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

                    remMask = np.zeros_like(cropImg[:, :, 0], dtype=np.uint8)
                    keepMask = remMask.copy()
                
                    for cls, xmi, ymi, xma, yma in rem_xyxy.astype(int):
                        if cls == TP:
                            cv2.rectangle(remMask, (xmi, ymi), (xma, yma), 1, -1)
                    for xmi, ymi, xma, yma in keep_xyxy.astype(int):
                        cv2.rectangle(keepMask, (xmi, ymi), (xma, yma), 1, -1)

                    remMask = remMask.astype(int)
                    keepMask = keepMask.astype(int)
                    mask = (keepMask - remMask)
                    mask[mask >= 0] = 1
                    mask[mask < 0] = 0
                    mask = mask.astype(np.uint8)
                    cropImg = cv2.bitwise_and(cropImg, cropImg, mask=mask)
                
                new_xywh = xyxy2xywh(keep_xyxy).astype(np.float32)
                new_xywh[:, [0, 2]] = new_xywh[:, [0, 2]] / w
                new_xywh[:, [1, 3]] = new_xywh[:, [1, 3]] / h
                
                new_xywh = np.hstack((np.zeros_like(new_xywh[:, [0]]), new_xywh))

                for xmi, ymi, xma, yma in keep_xyxy.astype(int):
                    cv2.rectangle(cropImg, (xmi, ymi), (xma, yma), (5, 5, 245), 1)

                basename = os.path.basename(path2img).split('.')[0]

                tile_txt = os.path.join(txt_dir, f'{basename}_crop_{i:02d}.txt')
                tile_jpg = os.path.join(img_dir, f'{basename}_crop_{i:02d}.jpg')

                np.savetxt(tile_txt, new_xywh, fmt=('%d', '%1.6f', '%1.6f', '%1.6f', '%1.6f'))
                cv2.imwrite(tile_jpg, cropImg, [cv2.IMWRITE_JPEG_QUALITY, 100])

                i += 1

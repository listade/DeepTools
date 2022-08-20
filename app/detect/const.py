"""The module of global constans"""


hyp_meta = { # hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
    "lr0": (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2)
    "momentum": (0.1, 0.6, 0.98),  # SGD momentum
    "weight_decay": (1, 0.0, 0.001),  # optimizer weight decay
    "giou": (1, 0.02, 0.2),  # GIoU loss gain
    "cls": (1, 0.2, 4.0),  # cls loss gain
    "cls_pw": (1, 0.5, 2.0),  # cls BCELoss positive_weight
    "obj": (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
    "obj_pw": (1, 0.5, 2.0),  # obj BCELoss positive_weight
    "iou_t": (0, 0.1, 0.7),  # IoU training threshold
    "anchor_t": (1, 2.0, 8.0),  # anchor-multiple threshold
    "fl_gamma": (0, 0.0, 2.0), # focal loss gamma (efficientDet default gamma=1.5)
    "hsv_h": (1, 0.0, 0.1), # image HSV-Hue augmentation (fraction)
    "hsv_s": (1, 0.0, 0.9), # image HSV-Saturation augmentation (fraction)
    "hsv_v": (1, 0.0, 0.9), # image HSV-Value augmentation (fraction)
    "degrees": (1, 0.0, 45.0),  # image rotation (+/- deg)
    "translate": (1, 0.0, 0.9),  # image translation (+/- fraction)
    "scale": (1, 0.0, 0.9),  # image scale (+/- gain)
    "shear": (1, 0.0, 10.0),  # image shear (+/- deg)
    "perspective": (1, 0.0, 0.001), # image perspective (+/- fraction), range 0-0.001
    "flipud": (0, 0.0, 1.0),  # image flip up-down (probability)
    "fliplr": (1, 0.0, 1.0),  # image flip left-right (probability)
    "mixup": (1, 0.0, 1.0), # image mixup (probability)
}
"""YOLO train module"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch.utils.data
import yaml
from torch import optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import test  # import test.py to get mAP after each epoch
from .models.yolo import Model
from .utils.datasets import create_dataloader
from .utils.general import (check_img_size, compute_loss, fitness,
                            increment_dir, labels_to_class_weights,
                            labels_to_image_weights, plot_evolution,
                            plot_images, plot_labels, plot_results,
                            save_mutation, strip_optimizer)
from .utils.torch_utils import ModelEMA, select_device
from .const import hyp_meta


def train(data: str,
          cfg: str,
          hyp: str,
          device: str,
          epochs: int,
          batch_size: int,
          img_size: int,
          weights: str,
          tb_writer=None,
          log_dir="."):

    """Training weights"""

    log_dir = Path(tb_writer.log_dir) if tb_writer is not None else Path(log_dir) / "evolve"  # logging directory path
    weights_dir = str(log_dir / "weights") + os.sep  # weights directory path in log dir
    last = weights_dir + "last.pt"  # last weights file
    best = weights_dir + "best.pt"  # best weights file
    results_file = str(log_dir / "results.txt")  # results path

    os.makedirs(weights_dir, exist_ok=True)  # create weights directory

    with open(data, encoding="utf-8") as f: # load data dict
        data_dict = yaml.load(f, Loader=yaml.FullLoader)

    with open(hyp, encoding="utf-8") as f: # load hyp dict
        hyp_dict = yaml.load(f, Loader=yaml.FullLoader)

    with open(log_dir / "hyp.yaml", "w", encoding="utf-8") as f: # save hyp dict
        yaml.dump(hyp_dict, f, sort_keys=False)

    torch.manual_seed(2) # speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    cudnn.deterministic = False
    cudnn.benchmark = True

    assert len(data_dict["names"]) == int(data_dict["nc"]) # check names count

    model = Model(cfg, ch=3, nc=int(data_dict["nc"])) # define model
    model = model.to(device) # load model to gpu

    if Path(weights).exists(): # if weights already exists
        state_dict = torch.load(weights, map_location=device)  # load weights
        model.load_state_dict(state_dict)  # load weights state

    nominal_batch_size = 64  # nominal batch size
    accumulate = max(round(nominal_batch_size / batch_size), 1) # accumulate loss before optimizing
    hyp_dict["weight_decay"] *= batch_size * accumulate / nominal_batch_size  # scale weight_decay

    params_group_0 = [] # other
    params_group_1 = [] # weight decay
    params_group_2 = [] # biases

    for k, v in model.named_parameters():
        v.requires_grad = True
        if ".bias" in k:
            params_group_2.append(v)  # biases
        elif ".weight" in k and ".bn" not in k:
            params_group_1.append(v)  # apply weight decay
        else:
            params_group_0.append(v)  # all else

    optimizer = optim.SGD(params_group_0, # other params
                          lr=hyp_dict["lr0"],
                          momentum=hyp_dict["momentum"],
                          nesterov=True)

    optimizer.add_param_group({
        "params": params_group_1,
        "weight_decay": hyp_dict["weight_decay"]
    }) # add pg1 with weight_decay

    optimizer.add_param_group({
        "params": params_group_2
    }) # add pg2 (biases)

    print(f"Optimizer groups: {len(params_group_2)} .bias, {len(params_group_1)} conv.weight, {len(params_group_0)} other")

    def lf(x): return (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf) # adjust learning rate

    start_epoch = 0
    best_fitness = 0.0

    max_stride = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, max_stride) for x in img_size] # verify imgsz are gs-multiples

    ema = ModelEMA(model) # exponential moving average
    train_loader, train_dataset = create_dataloader(path=data_dict["train"],
                                                    imgsz=imgsz,
                                                    batch_size=batch_size,
                                                    stride=max_stride,
                                                    hyp=hyp_dict,
                                                    augment=True,
                                                    cache=True,
                                                    rect=False) # train loader

    max_label_class = np.concatenate(train_dataset.labels, 0)[:, 0].max()  # max label class
    batches_num = len(train_loader)  # num of batches

    assert max_label_class < int(data_dict["nc"])  # check labels num

    ema.updates = start_epoch * batches_num // accumulate  # set EMA updates
    test_loader, _ = create_dataloader(path=data_dict["val"],
                                       imgsz=imgsz_test,
                                       batch_size=batch_size,
                                       stride=max_stride,
                                       hyp=hyp_dict,
                                       augment=False,
                                       cache=True,
                                       rect=True)

    hyp_dict["cls"] *= int(data_dict["nc"]) / 80.  # scale coco-tuned hyp["cls"] to current dataset

    model.nc = int(data_dict["nc"])  # attach number of classes to model
    model.hyp = hyp_dict  # attach hyperparameters to model
    model.gr = 1.0  # attach giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(train_dataset.labels, int(data_dict["nc"])).to(device)  # attach class weights
    model.names = data_dict["names"] # attach names to model

    labels = np.concatenate(train_dataset.labels, 0)  # [?]
    c = torch.tensor(labels[:, 0])  # classes

    plot_labels(labels, save_dir=log_dir)

    if tb_writer:
        tb_writer.add_histogram("classes", c, 0)

    t0 = time.time() # start training
    nw = max(3 * batches_num, 1e3) # number of warmup iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(int(data_dict["nc"]))  # mAP per class

    results = (0, 0, 0, 0, 0, 0, 0) # "P", "R", "mAP", "F1", "val GIoU", "val Objectness", "val Classification"

    scheduler.last_epoch = start_epoch - 1  # do not move
    is_cuda = device.type != "cpu"  # is cuda device
    scaler = amp.GradScaler(enabled=is_cuda)

    print(f"Image sizes {imgsz} train, {imgsz_test} test")
    print(f"Using {train_loader.num_workers} dataloader workers")
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(start_epoch, epochs): # epoch
        model.train()

        if train_dataset.image_weights: # update image weights (optional)
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2
            image_weights = labels_to_image_weights(train_dataset.labels,
                                                    nc=int(data_dict["nc"]),
                                                    class_weights=w)

            train_dataset.indices = random.choices(range(train_dataset.img_num), # generate indices
                                                   weights=image_weights,
                                                   k=train_dataset.img_num)  # rand weighted idx

        mloss = torch.zeros(4, device=device)  # mean losses
        pbar = enumerate(train_loader)

        print(("\n" + "%10s" * 8) % ("Epoch", "gpu_mem", "GIoU", "obj", "cls", "total", "targets", "img_size"))
        pbar = tqdm(pbar, total=batches_num)  # progress bar

        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar: # batch
            ni = i + batches_num * epoch # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            if ni <= nw: # warmup
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nominal_batch_size / batch_size]).round()) # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)

                for j, x in enumerate(optimizer.param_groups):
                    x["lr"] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x["initial_lr"] * lf(epoch)]) # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0

                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [0.9, hyp_dict["momentum"]])

            with amp.autocast(enabled=is_cuda): # autocast
                pred = model.forward(imgs) # forward
                loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size

            scaler.scale(loss).backward() # backward

            if ni % accumulate == 0: # optimize
                scaler.step(optimizer)  # optimizer.step
                scaler.update()

                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ("%10s" * 2 + "%10.4g" * 6) % ("%g/%g" % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])

            pbar.set_description(s)

            if ni < 3:
                f = str(log_dir / ("train_batch%g.jpg" % ni))  # filename
                result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)

                if tb_writer and result is not None:
                    tb_writer.add_image(f, result, dataformats="HWC", global_step=epoch)

        scheduler.step() # scheduler epoch step

        if ema is not None: # mean averate precision
            ema.update_attr(model, include=["yaml", "nc", "hyp", "gr", "names", "stride"])

        final_epoch = epoch + 1 == epochs
        if final_epoch:
            results, maps, _ = test.test(data,
                                         batch_size=batch_size,
                                         imgsz=imgsz_test,
                                         model=ema.ema.module if hasattr(ema.ema, "module") else ema.ema,
                                         single_cls=False,
                                         dataloader=test_loader,
                                         save_dir=log_dir) # calculate mAP

        with open(results_file, "a", encoding="utf-8") as f: # write
            f.write(s + "%10.4g" * 7 % results + "\n") # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        if tb_writer is not None: # if tensorboard enabled
            tags = ["train/giou_loss",
                    "train/obj_loss",
                    "train/cls_loss",
                    "metrics/precision",
                    "metrics/recall",
                    "metrics/mAP_0.5",
                    "metrics/mAP_0.5:0.95",
                    "val/giou_loss",
                    "val/obj_loss",
                    "val/cls_loss"]

            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

            fi = fitness(np.array(results).reshape(1, -1)) # fitness_i = weighted combination of [P, R, mAP, F1]

            if fi > best_fitness:
                best_fitness = fi # update best mAP

            with open(results_file, "r", encoding="utf-8") as f:  # create checkpoint
                state_dict = {"epoch": epoch,
                                "best_fitness": best_fitness,
                                "training_results": f.read(),
                                "model": ema.ema.module if hasattr(ema, "module") else ema.ema,
                                "optimizer": None if final_epoch else optimizer.state_dict()} # save model

            torch.save(state_dict, last) # save last, best and delete

            if epoch >= (epochs-30):
                torch.save(state_dict, last.replace(".pt", "_{:03d}.pt".format(epoch)))

            if best_fitness == fi:
                torch.save(state_dict, best)


    n = "_" # strip optimizers begin
    fresults = "results%s.txt" % n
    flast = weights_dir + "last%s.pt" % n
    fbest = weights_dir + "best%s.pt" % n

    for f1, f2 in zip([weights_dir + "last.pt", weights_dir + "best.pt", "results.txt"], [flast, fbest, fresults]):
        if os.path.exists(f1):
            os.rename(f1, f2)  # rename
            if f2.endswith(".pt"):
                strip_optimizer(f2, f2.replace(".pt", "_strip.pt")) # strip optimizer

    if tb_writer is None:
        plot_results(save_dir=log_dir)  # save as results.png

    print(f"{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600} hours")
    torch.cuda.empty_cache()

    return results


def evolve(data: str,
           cfg: str,
           hyp: str,
           device: str,
           epochs: int,
           batch_size: int,
           weights: str,
           img_size: int):
    """Hyperparaemeters evolution"""

    with open(hyp, encoding="utf-8") as f:
        hyp_dict = yaml.load(f, Loader=yaml.FullLoader)  # load hyp dict

    result_path = Path("evolve/hyp_evolved.yaml")

    for _ in range(100):  # generations to evolve
        if os.path.exists("evolve.txt"): # if evolve.txt exists: select best hyps and mutate

            mutations = np.loadtxt("evolve.txt", ndmin=2)
            n = min(5, len(mutations))  # number of previous results to consider
            mutations = mutations[np.argsort(-fitness(mutations))][:n]  # top n mutations
            w = fitness(mutations) - fitness(mutations).min()  # weights [?]
            mutations = mutations[random.choices(range(n), weights=w)[0]]  # weighted selection
            mutation_probability = .9 # mutation probability
            sigma = .2 # sigma

            np.random.seed(int(time.time()))

            g = np.array([x[0] for x in hyp_meta.values()])  # gains 0-1
            ng = len(hyp_meta)
            v = np.ones(ng)

            while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                v = (g * (np.random.random(ng) < mutation_probability) * np.random.randn(ng) * np.random.random() * sigma + 1).clip(0.3, 3.0)

            for i, k in enumerate(hyp_dict.keys()):  # plt.hist(v.ravel(), 300)
                hyp_dict[k] = float(mutations[i + 7] * v[i])  # mutate

        for k, v in hyp_meta.items(): # constrain to limits
            hyp_dict[k] = max(hyp_dict[k], v[1])  # lower limit
            hyp_dict[k] = min(hyp_dict[k], v[2])  # upper limit
            hyp_dict[k] = round(hyp_dict[k], 5)  # significant digits

        results = train(data=data,
                        cfg=cfg,
                        hyp=hyp,
                        device=device,
                        epochs=epochs,
                        batch_size=batch_size,
                        weights=weights,
                        img_size=img_size) # train mutation

        save_mutation(hyp_dict, results, result_path) # Write mutation results

    plot_evolution(result_path) # plot results

    print(f"""
            Hyperparameter evolution complete.
            Best results saved as: {result_path}
            Command to train a new model with these hyperparameters: $ python train.py --hyp {hyp}
          """)


def main():
    """Entry point"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--hyp", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--img-size", nargs="+", type=int, default=[640, 640])
    parser.add_argument("--rect", action="store_true")
    parser.add_argument("--evolve", action="store_true")
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--logdir", type=str, default=".")

    opt = parser.parse_args()
    device = select_device(opt.device, batch_size=opt.batch_size) # check device

    if opt.evolve:
        evolve(weights=opt.weights,
               data=opt.data,
               cfg=opt.cfg,
               hyp=opt.hyp,
               device=device,
               epochs=opt.epochs,
               batch_size=opt.batch_size,
               img_size=opt.img_size) # evolve hyperparameters
        sys.exit(0)

    print(f"Start Tensorboard with \"tensorboard --logdir {opt.logdir}\", view at http://localhost:6006/")

    log_dir = increment_dir(Path(opt.logdir) / "exp")
    tb_writer = SummaryWriter(log_dir=log_dir)  # runs/exp

    train(weights=opt.weights,
          data=opt.data,
          cfg=opt.cfg,
          hyp=opt.hyp,
          device=device,
          epochs=opt.epochs,
          batch_size=opt.batch_size,
          img_size=opt.img_size,
          tb_writer=tb_writer)


if __name__ == "__main__":
    main()

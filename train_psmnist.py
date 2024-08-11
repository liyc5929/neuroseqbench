import argparse
import json
import os
import logging
import time
from functools import partial
from datetime import datetime
import torch
from torch.cuda import amp

from src.benchmark.framework.utils.tools import set_random_seed, setup_logging, save_checkpoint, AverageMeter, ProgressMeter, accuracy, count_parameters
from src.benchmark.framework.utils.dataset import build_dataset
from src.benchmark.framework.network.trainer import SurrogateGradient
from src.benchmark.framework.network.neuron import LIF
from src.benchmark.framework.network.architecture import SpikingNet

parser = argparse.ArgumentParser()
# datasets
parser.add_argument("--dataset", default="psmnist", type=str, help="dataset: [|psmnist]")
parser.add_argument("--data-path", default="./datasource/MNIST", help="path to dataset,")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)")

parser.add_argument("--net", default="ffsnn", type=str, help="networks")
parser.add_argument("--seed", default=1234, type=int, help="seed for initializing training. ")
parser.add_argument("--amp", action="store_true", help="automatic mixed precision training")
parser.add_argument("--save-path", default="", type=str, help="the directory used to save the trained models")
parser.add_argument("--name", default="", type=str, help="name of experiment")

parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument("-b", "--batch-size", 
    default=128, type=int, metavar="N",
    help = "mini-batch size (default: 128), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel"
)
parser.add_argument("-p", "--print-freq", default=500, type=int, metavar="N", help="print frequency (default: 10)")
parser.add_argument("--save-ckpt", action="store_true", default=True, help="")
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

# optimizer
parser.add_argument("--optim", default="sgd", type=str, help="optimizer (default: sgd)")
parser.add_argument("--lr", "--learning-rate", default=0.1, type=float, metavar="LR", help="initial learning rate", dest="lr")
parser.add_argument("--wd", "--weight-decay", default=0, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")

# Cosine learning rate
parser.add_argument("--cos-lr", action="store_true", default=False, help="whether to use cosine learning rate")
parser.add_argument("--step-lr", action="store_true", default=False, help="whether to use cosine learning rate")
parser.add_argument("--step-size", type=int, default=10, help="")

# args of spiking neural networks
parser.add_argument("--threshold", type=float, default=0.5, help="neuronal threshold (default: 1)")
parser.add_argument("--time-step", type=int, default=20, help="total time steps (default: 10)")
parser.add_argument("--decay", type=float, default=0.5, help="decay factor (default: 5)")
parser.add_argument("--grad-clip", type=float, default=0.)
parser.add_argument("--neuron", default="lif", type=str, help="[lif]")
parser.add_argument("--alpha", type=float, default=1., help="scaling factor of surrogate gradient (default 1.0)")

parser.add_argument("--recurrent", action="store_true", default=False, help="")
parser.add_argument("--surrogate", default="rectangle", type=str, help="[rectangle, triangle, sigmoid, multigauss]")


def main():
    args = parser.parse_args()
    if args.save_path == "":
        save_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = save_path + args.name + "_" + str(args.seed)
    else:
        save_path = args.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Logging settings
    setup_logging(os.path.join(save_path, "log.txt"))
    logging.info("saving to:" + str(save_path))

    is_cuda = torch.cuda.is_available()
    assert is_cuda, "CPU is not supported!"
    device = torch.device("cuda" if is_cuda else "cpu")
    set_random_seed(seed=args.seed)
    torch.backends.cudnn.benchmark = False
    args.gpu = "cuda"

    with open(save_path + "/args.json", "w") as fid:
        json.dump(args.__dict__, fid, indent=2)

    logging.info("args:" + str(args))

    train_dataset, val_dataset, input_channels, num_classes, collate_fn = build_dataset(dataset=args.dataset, data_path=args.data_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        num_workers = args.workers,
        shuffle = True,
        collate_fn = collate_fn,
        pin_memory = True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = args.batch_size,
        num_workers = args.workers,
        collate_fn = collate_fn,
        pin_memory = True,
        shuffle = False
    )

    if args.neuron == "lif":
        surro_grad = SurrogateGradient(func_name=args.surrogate, a=args.alpha)
        exec_mode = "serial"
        spiking_neuron = partial(
            LIF,
            decay = args.decay,
            threshold = args.threshold,
            time_step = args.time_step,
            surro_grad = surro_grad,
            exec_mode = exec_mode,
            recurrent = args.recurrent,
        )
    else:
        raise NotImplementedError

    model = SpikingNet(input_size=1, hidden_size=[64, 256, 256], output_size=num_classes, num_hidden_layers=3, spiking_neuron=spiking_neuron, args=args)

    logging.info(str(model))

    para = count_parameters(model)
    logging.info(f"Parameter number: {para}")

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optim == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    criterion = torch.nn.CrossEntropyLoss()
    best_acc1 = 0

    if args.cos_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    elif args.step_lr:
        gamma = 0.8
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=gamma)
    else:
        scheduler = None

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    model = model.cuda()

    standard_train(train_loader, val_loader, model, criterion, optimizer, scheduler, save_path, best_acc1, scaler, args)


def standard_train(train_loader, val_loader, model, criterion, optimizer, scheduler, save_path, best_acc1, scaler, args):
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_acc1, train_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, scaler, args)
        if scheduler is not None:
            scheduler.step()
        # evaluate on validation set
        val_acc1, val_loss = validate_one_epoch(val_loader, model, criterion, args)


        out_string = "Train Acc. {:.4f} Test Acc. {:.4f} lr {:.4f}\t".format(train_acc1, val_acc1, optimizer.param_groups[0]["lr"])
        logging.info(out_string)
        # remember best acc@1 and save checkpoint

        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        if args.save_ckpt:
            save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
            }, is_best, filename=os.path.join(save_path, "checkpoint.pth.tar"), save_path=save_path)

    logging.info(f"Best accuracy: {best_acc1}")
    logging.info("Finished.")


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, scaler, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    for i, (images, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)
        images = images.cuda(args.gpu, non_blocking=True)
        target = labels.cuda(args.gpu, non_blocking=True)

        if args.dataset in ["psmnist"]:
            images = images.transpose(0, 1).contiguous()  # [T, B, N]
        optimizer.zero_grad()
        if args.amp:
            with amp.autocast():
                output = model(images)  # [T, B, N]
                output = output.mean(0)
                loss = criterion(output, target)
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(images)  # [T, B, N]
            output = output.mean(0)
            loss = criterion(output, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], target.size(0))
        top5.update(acc5[0], target.size(0))
        losses.update(loss.item(), target.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0 or (i + 1) == len(train_loader):
            progress.display(i + 1)

    return top1.avg, losses.avg


def validate_one_epoch(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix="Test: "
    )

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            if args.dataset in ["psmnist"]:
                images = images.transpose(0, 1).contiguous()  # [T, B, N]

            # compute output
            output = model(images)
            output = output.mean(0)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], target.size(0))
            top5.update(acc5[0], target.size(0))
            losses.update(loss.item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0 or (i + 1) == len(val_loader):
                progress.display(i + 1)
    return top1.avg, losses.avg


if __name__ == "__main__":
    main()

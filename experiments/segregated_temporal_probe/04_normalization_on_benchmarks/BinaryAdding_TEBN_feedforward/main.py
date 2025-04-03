import argparse
import json
import os
import sys
import logging
import toml
import time
from datetime import datetime
from functools import partial
import torch
import torch.nn as nn

# Check and add current working directory
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

from src.benchmark.framework.utils.tools import setup_logging, save_checkpoint, AverageMeter, ProgressMeter, accuracy, count_parameters
from src.benchmark.framework.network.trainer import SurrogateGradient
from src.benchmark.framework.network.neuron import ThresholdDependentBatchNorm1d, TemporalEffectiveBatchNorm1d
from src.benchmark.framework.network.neuron import RLIF
from src.benchmark.framework.utils.dataset import AddingProblem


class SpikingNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=1, spiking_neuron=None, bn=None, args=None):
        super(SpikingNet, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.args = args
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size] * num_hidden_layers
        else:
            assert len(hidden_size) == num_hidden_layers

        if bn == "ln":
            bns = [nn.LayerNorm(hidden_size[l]) for l in range(num_hidden_layers)]
        elif bn == "tdbn":
            bns = [ThresholdDependentBatchNorm1d(alpha=1, v_th=spiking_neuron().threshold, num_features=hidden_size[l]) for l in range(num_hidden_layers)]
        elif bn == "tebn":
            bns = [TemporalEffectiveBatchNorm1d(T=spiking_neuron().time_step, num_features=hidden_size[l]) for l in range(num_hidden_layers)]
        elif bn is None:
            bns = [None for _ in range(num_hidden_layers)]
        else:
            raise NotImplementedError

        for hidden_layer_i in range(num_hidden_layers):
            exec("self.fc" + str(hidden_layer_i) + " = nn.Linear(in_features=input_size, out_features=hidden_size[hidden_layer_i])")

            if hidden_layer_i == (num_hidden_layers - 1):
                exec("self.spk" + str(
                    hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size[hidden_layer_i], bn=bns[hidden_layer_i], recurrent=False)")
            else:
                exec("self.spk" + str(hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size[hidden_layer_i], bn=bns[hidden_layer_i])")
            input_size = hidden_size[hidden_layer_i]
        self.classifier = nn.Linear(in_features=input_size, out_features=output_size)

    def single_step_forward(self, x):
        for hidden_layer_i in range(self.num_hidden_layers):
            x = eval("self.fc" + str(hidden_layer_i))(x)
            x = eval("self.spk" + str(hidden_layer_i))(x)
        x = self.classifier(x)
        return x

    def forward(self, x, time_step=None):
        if time_step is None:
            time_step = x.size(0)
        output = self.multi_step_forward(x, time_step)
        return output

    def multi_step_forward(self, x, time_step):
        for hidden_layer_i in range(0, self.num_hidden_layers):
            x = eval("self.fc" + str(hidden_layer_i))(x)
            x = eval("self.spk" + str(hidden_layer_i))(x)
        x = self.classifier(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="GPU device number")
    parser.add_argument("--config", type=str, default="test.toml", help="TOML config file name")
    parser.add_argument("--data_root", type=str, default="/benchmark_data")
    parser.add_argument("--save_ckpt", default=True, action="store_true", help="")
    
    # Get `argparse` arguments
    args = parser.parse_args()

    # Get `toml` arguments
    with open(args.config, "r") as fp:
        config = toml.load(fp)

    # General setting
    args.seed          = config.get("seed", 1234)

    # Hyperparameter setting
    args.max_epoch     = config.get("max_epoch", 100)
    args.time_step     = config.get("time_step", 20)
    args.batch_size    = config.get("batch_size", 128)
    args.learning_rate = config.get("learning_rate", 0.1)
    args.weight_decay  = config.get("weight_decay", 0.0)
    args.momentum      = config.get("momentum", 0.9)
    args.optimizer     = config.get("optimizer", "SGD")
    args.use_cos_lr    = config.get("use_cos_lr", False)
    args.use_step_lr   = config.get("use_step_lr", False)
    args.grad_clip     = config.get("grad_clip", 0.0)

    # Architecture setting
    args.hidden_size   = config.get("hidden_size", 512)
    args.hidden_layers = config.get("hidden_layers", 3)
    args.truncated_t   = config.get("truncated_t", 10000)

    # Neuron setting
    args.neuron_decay  = config.get("neuron_decay", 0.5)
    args.neuron_thresh = config.get("neuron_thresh", 0.5)
    args.surro_alpha   = config.get("surro_alpha", 1.0)
    args.recurrent     = config.get("recurrent", False)

    return args


def main():
    args = parse_args()

    save_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = save_path + "_" + str(args.seed)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Logging settings
    setup_logging(os.path.join(save_path, "log.txt"))
    logging.info("saving to:" + str(save_path))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0")
    print(f"Using device {args.device}.")

    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    with open(save_path + "/args.json", "w") as fid:
        json.dump(args.__dict__, fid, indent=2)
    logging.info("args:" + str(args))

    train_dataset = AddingProblem(
        root       = args.data_root,
        subset     = "train",
        num_data   = 50000,
        seq_length = args.time_step,
        is_binary  = True,
        preprocess = None,
    )
    val_dataset = AddingProblem(
        root       = args.data_root,
        subset     = "test",
        num_data   = 2000,
        seq_length = args.time_step,
        is_binary  = True,
        preprocess = None,
    )
    num_classes = 10
    collate_fn = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=False
    )

    surro_grad = SurrogateGradient(func_name="rectangle", a=args.surro_alpha)
    exec_mode = "serial"
    spiking_neuron = partial(RLIF,
        decay=args.neuron_decay,
        threshold=args.neuron_thresh,
        time_step=args.time_step,
        surro_grad=surro_grad,
        exec_mode=exec_mode,
        recurrent=args.recurrent,
        learning_rule="stbp",
        truncated_t=args.truncated_t,
    )

    model = SpikingNet(input_size=2, hidden_size=args.hidden_size, output_size=num_classes, num_hidden_layers=args.hidden_layers, spiking_neuron=spiking_neuron, bn="tebn", args=args)
    logging.info(str(model))

    para = count_parameters(model)
    logging.info(f"Parameter number: {para}")

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.2)
    best_acc1 = 0
    if args.use_cos_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.max_epoch)
    elif args.use_step_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    else:
        scheduler = None

    model = model.to(device)

    standard_train(train_loader, val_loader, model, criterion, optimizer, scheduler, save_path, best_acc1, device, args)


def standard_train(train_loader, val_loader, model, criterion, optimizer, scheduler, save_path, best_acc1, device, args):
    for epoch in range(0, args.max_epoch):
        # Train for one epoch
        train_acc1, train_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, args)
        if scheduler is not None:
            scheduler.step()
        # Evaluate on validation set
        val_acc1, val_loss = validate_one_epoch(val_loader, model, criterion, device, args)

        out_string = "Train Acc. {:.4f} Test Acc. {:.4f} lr {:.4f}\t".format(train_acc1, val_acc1, optimizer.param_groups[0]["lr"])
        logging.info(out_string)

        # Record best acc@1 and save checkpoint
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


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device, non_blocking=True)
        target = labels.to(device, non_blocking=True)

        images = images.transpose(0, 1).contiguous() # [T, B, N]
        optimizer.zero_grad()
        output = model(images) # [T, B, N]
        output = output[-1]
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

        if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
            progress.display(i + 1)

    return top1.avg, losses.avg


def validate_one_epoch(val_loader, model, criterion, device, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix="Test: ",
    )

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            images = images.transpose(0, 1).contiguous() # [T, B, N]

            output = model(images)
            output = output[-1]
            loss = criterion(output, target)

            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], target.size(0))
            top5.update(acc5[0], target.size(0))
            losses.update(loss.item(), target.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % 100 == 0 or (i + 1) == len(val_loader):
                progress.display(i + 1)
    return top1.avg, losses.avg


if __name__ == "__main__":
    main()

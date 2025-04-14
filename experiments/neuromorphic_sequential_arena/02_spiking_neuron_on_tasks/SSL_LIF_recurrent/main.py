import argparse
import json
import os
import logging
import time
from functools import partial
import h5py
import toml
import torch
import torch.nn as nn
from datetime import datetime

from neuroseqbench.utils.tools import (
    setup_logging, save_checkpoint, AverageMeter, ProgressMeter, accuracy, count_parameters, dump_json
)
from neuroseqbench.network.trainer import SurrogateGradient
from neuroseqbench.network.neuron import Recurrent_LIF, CELIF, SPSN, LTC
from neuroseqbench.network.structure import MergeDimension, SplitDimension
from neuroseqbench.utils.dataset import SLoClas


class FFSNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=1, pool=False, dataset = None, spiking_neuron=None, loss=None, neuron_type=None):
        super(FFSNN, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.flatten = nn.Flatten()
        self.pool = pool
        self.neuron = neuron_type
        self.dataset = dataset
        if self.pool:
            self.max_pool = nn.MaxPool2d(4, 4)
        for hidden_layer_i in range(num_hidden_layers):
            if self.neuron == "dhsnn":
                if dataset in ["PSMNIST", "BinaryAdding", "AL", "HAR", "EEG", "SSL"] and hidden_layer_i == (num_hidden_layers - 1):
                    exec("self.spk" + str(
                        hidden_layer_i) + " = spiking_neuron(input_features=input_size, neuron_num=hidden_size[{0}], recurrent=False)".format
                         (hidden_layer_i))
                else:
                    exec("self.spk" + str(hidden_layer_i) + " = spiking_neuron(input_features=input_size, neuron_num=hidden_size[{0}])".format
                    (hidden_layer_i))
            elif self.neuron in ["SSM", "BinarySSM", "GSU"]:
                exec("self.spk" + str(
                    hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size[{0}])".format(hidden_layer_i))
                if hidden_layer_i == 0:
                    exec("self.fc" + str(hidden_layer_i) + " = nn.Linear(in_features=input_size, out_features=hidden_size[{0}])".format(hidden_layer_i))
            else:
                exec("self.fc" + str(
                    hidden_layer_i) + " = nn.Linear(in_features=input_size, out_features=hidden_size[{0}])".format(
                    hidden_layer_i))
                if dataset in ["PSMNIST","BinaryAdding","AL","HAR","EEG","SSL"] and hidden_layer_i == (num_hidden_layers - 1):
                    exec("self.spk" + str(
                        hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size[{0}], recurrent=False)".format(
                        hidden_layer_i))
                else:
                    exec("self.spk" + str(hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size[{0}])".format(
                        hidden_layer_i))
            input_size = hidden_size[hidden_layer_i]
        self.classifier = nn.Linear(in_features=input_size, out_features=output_size)
        if self.neuron == "celif":
            self.TE = nn.Parameter(torch.zeros(max(hidden_size),self.spk0.time_step))
            nn.init.normal_(self.TE, 0.01, 0.01)
            for hidden_layer_i in range(num_hidden_layers):
                exec("self.spk" + str(hidden_layer_i) + " .TE = self.TE".format(hidden_layer_i))

    def forward(self, x, time_step=None):
        if time_step is None:
            time_step = x.size(0)
        output = self.multi_step_forward(x, time_step)
        if self.dataset in ["BinaryAdding", "EEG"]: # laststep decision
            output=output[-1, ...].unsqueeze(0)
        return output

    def multi_step_forward(self, x, time_step):
        x = MergeDimension()(x)
        if self.pool:
            x = self.max_pool(x)
        x = self.flatten(x)
        x = SplitDimension(time_step)(x)
        for hidden_layer_i in range(self.num_hidden_layers):
            if self.neuron == "dhsnn":
                x = x
            elif self.neuron in ["SSM", "BinarySSM", "GSU"]:
                if hidden_layer_i == 0:
                    x = MergeDimension()(x)
                    x = eval("self.fc" + str(hidden_layer_i))(x)
                    x = SplitDimension(time_step)(x)
                else:
                    x = x
            else:
                x = MergeDimension()(x)
                x = eval("self.fc" + str(hidden_layer_i))(x)
                x = SplitDimension(time_step)(x)
            x = eval("self.spk" + str(hidden_layer_i))(x)
        time_step = x.size(0) # in case the membrane potential output only have T=1
        x = MergeDimension()(x)
        x = self.classifier(x)
        x = SplitDimension(time_step)(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="GPU device number")
    parser.add_argument("--config", type=str, default="test.toml", help="TOML config file name")
    parser.add_argument("--data_root", type=str, default="/benchmark_data")
    parser.add_argument("--save-ckpt", default=True, action="store_true", help="")
    
    # Get `argparse` arguments
    args = parser.parse_args()

    # Get `toml` arguments
    with open(args.config, "r") as fp:
        config = toml.load(fp)

    # General setting
    args.seed          = config.get("seed", 1234)

    # Hyperparameter setting
    args.max_epoch     = config.get("max_epoch", 100)
    args.batch_size    = config.get("batch_size", 64)
    args.time_step     = config.get("time_step", 500)
    args.learning_rate = config.get("learning_rate", 0.1)
    args.weight_decay  = config.get("weight_decay", 0.0)
    args.momentum      = config.get("momentum", 0.9)
    args.optimizer     = config.get("optimizer", "AdamW")
    args.use_cos_lr    = config.get("use_cos_lr", False)
    args.grad_clip     = config.get("grad_clip", 1.0)

    # Architecture setting
    args.hidden_dim    = config.get("hidden_dim", [128, 256, 256])

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

    data_path = args.data_root + "/SSL"
    # Check if preprocessed data exists; if so, load it directly;
    # Otherwise, load raw data, perform preprocessing, and save the results
    train_data = h5py.File(data_path + "/Training_raw_noise.mat")
    test_data = h5py.File(data_path + "/Testing_raw_noise.mat")
    train_dataset, val_dataset = SLoClas(train_data, test_data, sequence_length=500)
    input_channels = 4
    num_classes = 72

    train_pin_memory = True
    test_pin_memory = True
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size = args.batch_size,
        num_workers = 0,
        shuffle = (train_sampler is None),
        pin_memory = train_pin_memory,
        sampler = train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size = args.batch_size,
        num_workers = 0,
        pin_memory = test_pin_memory,
        shuffle = False,
    )

    print("Finish data loading.")
    args.surrogate = "triangle"
    surro_grad = SurrogateGradient(func_name=args.surrogate, a=args.surro_alpha)
    exec_mode = "serial"
    learning_rule = "STBP"
    args.neuron = "lif"
    if args.neuron == "lif":
        spiking_neuron = partial(Recurrent_LIF,
            decay = args.neuron_decay,
            threshold = args.neuron_thresh,
            time_step = args.time_step,
            surro_grad = surro_grad,
            exec_mode = exec_mode,
            recurrent = args.recurrent,
            cut_grad = True if learning_rule == "SDBP" else False,
        )
    elif args.neuron == "ltc":
        spiking_neuron = partial(LTC,
            decay = args.neuron_decay,
            threshold = args.neuron_thresh,
            time_step = args.time_step,
            surro_grad = surro_grad,
            exec_mode = exec_mode,
            recurrent = args.recurrent,
        )
    elif args.neuron == "celif":
        beta = 0.02
        spiking_neuron = partial(CELIF,
            decay = args.neuron_decay,
            threshold = args.neuron_thresh,
            time_step = args.time_step,
            surro_grad = surro_grad,
            exec_mode = exec_mode,
            recurrent = args.recurrent,
            beta = beta,
        )
    elif args.neuron == "spsn":
        spiking_neuron = partial(SPSN,
            decay = args.neuron_decay,
            threshold = args.neuron_thresh,
            time_step = args.time_step,
            surro_grad = surro_grad,
            exec_mode = exec_mode,
            recurrent = args.recurrent,
            k = args.time_step,
        )
    else:
        raise NotImplementedError

    model = FFSNN(input_size=input_channels, hidden_size=args.hidden_dim, output_size=num_classes, num_hidden_layers=len(args.hidden_dim), spiking_neuron=spiking_neuron, dataset="SSL", neuron_type=args.neuron)
    logging.info(str(model))
    para = count_parameters(model)
    logging.info(f"Parameter number: {para}")

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.2)
    best_cri = [0, 100]

    if args.use_cos_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.max_epoch)
    else:
        gamma = 0.8
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)

    model = model.to(device)
    standard_train(train_loader, val_loader, model, criterion, optimizer, scheduler, save_path, best_cri, device, args)


def standard_train(train_loader, val_loader, model, criterion, optimizer, scheduler, save_path, best_cri, device, args):
    loss_train_record  = []
    for epoch in range(0, args.max_epoch):
        # Train for one epoch
        train_acc1, train_loss, loss_train_record, train_mae = train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, args, loss_train_record)
        scheduler.step()
        val_acc1, val_loss, val_mae = validate_one_epoch(val_loader, model, criterion, device, args)
        out_string = "Train Acc. {:.4f} Train MAE {:.4f} Test Acc. {:.4f} Test MAE {:.4f} \n ".format(train_acc1, train_mae, val_acc1, val_mae)
        logging.info(out_string)
        # Record best acc@1 and save checkpoint
        is_best = val_acc1 > best_cri[0]
        best_cri[0] = max(val_acc1, best_cri[0])
        best_cri[1] = min(val_mae, best_cri[1])
        if args.save_ckpt:
            save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_cri": best_cri,
                "optimizer": optimizer.state_dict(),
            }, is_best, filename=os.path.join(save_path, "checkpoint.pth.tar"), save_path=save_path)

        training_record = {
            "loss_train_record": loss_train_record,
        }
        dump_json(training_record, save_path, "loss_train_record.txt")
    logging.info(f"Best accuracy/mae: {best_cri}")


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, args, loss_train_record):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    maes = AverageMeter("MAE", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, maes],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        images = images.transpose(0, 1).contiguous() # [T, B, C]
        target = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(images) # [T, B, N]
        # Average across time
        output_mean = output.mean(0)
        loss = criterion(output_mean, target)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        mae = angular_distance_compute(target, torch.argmax(output_mean,dim=1))
        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output_mean, target, topk=(1, 5))
        top1.update(acc1[0], target.size(0))
        top5.update(acc5[0], target.size(0))

        losses.update(loss.item(), target.size(0))
        loss_train_record.append(loss.item())
        maes.update(mae.item(), target.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % 50 == 0 or (i + 1) == len(train_loader):
            progress.display(i + 1)

    return top1.avg, losses.avg, loss_train_record, maes.avg


def validate_one_epoch(val_loader, model, criterion, device, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    maes = AverageMeter("MAE", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5, maes],
        prefix="Test: ")

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            images = images.transpose(0,1).contiguous()  # [T, B, ..]
            target = target.to(device, non_blocking=True)

            # Compute output
            output = model(images) # [T, B, N]
            output = output.mean(0)
            loss = criterion(output, target)
            mae = angular_distance_compute(target, torch.argmax(output, dim=1))
            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], target.size(0))
            top5.update(acc5[0], target.size(0))
            losses.update(loss.item(), target.size(0))
            maes.update(mae.item(), target.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % 50 == 0 or (i + 1) == len(val_loader):
                progress.display(i + 1)
    return top1.avg, losses.avg, maes.avg


def angular_distance_compute(label, pred):
    mae = []
    for i in range(len(label)):
        result = 180 - abs(abs(label[i] - pred[i]) - 180)
        mae.append(result)
    return sum(mae) / len(mae)


if __name__ == "__main__":
    main()

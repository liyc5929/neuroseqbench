import argparse
import json
import os
import logging
import time
from functools import partial
import h5py

import torch
import torch.nn as nn
from datetime import datetime

from neuroseqbench.utils.tools import (
    setup_logging, save_checkpoint, AverageMeter, ProgressMeter, accuracy, count_parameters, dump_json
)
from neuroseqbench.network.trainer import SurrogateGradient
from neuroseqbench.network.neuron import Recurrent_LIF, CELIF, SPSN, LTC, S4D
from neuroseqbench.network.neuron.s4d import setup_optimizer
from neuroseqbench.network.structure import SSM, TCN, LSTMNet, SpkTransformerNet
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
                if dataset in ["psmnist","biadd","AL","HAR","EEG","SSL"] and hidden_layer_i == (num_hidden_layers - 1):
                    exec("self.spk" + str(
                        hidden_layer_i) + " = spiking_neuron(input_features=input_size, neuron_num=hidden_size[{0}], recurrent=False)".format
                         (hidden_layer_i))
                else:
                    exec("self.spk" + str(hidden_layer_i) + " = spiking_neuron(input_features=input_size, neuron_num=hidden_size[{0}])".format
                    (hidden_layer_i))
            else:
                exec("self.fc" + str(
                    hidden_layer_i) + " = nn.Linear(in_features=input_size, out_features=hidden_size[{0}])".format(
                    hidden_layer_i))
                if dataset in ["psmnist","biadd","AL","HAR","EEG","SSL"] and hidden_layer_i == (num_hidden_layers - 1):
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
        if self.dataset in ["add", "biadd", "EEG"]: # last-step decision
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


parser = argparse.ArgumentParser(description="PyTorch Training")
# args of datasets

parser.add_argument("--dataset", default="SSL", type=str, help="dataset")
parser.add_argument("--data-path", default="/benchmark_data",
                    help="path to dataset,")
parser.add_argument("-j", "--workers", default=0, type=int, metavar="N",
                    help="number of data loading workers (default: 4)")

parser.add_argument("--net", default="ffsnn", type=str,
                    help="networks")
parser.add_argument("--seed", default=1234, type=int,
                    help="seed for initializing training. ")
parser.add_argument("--save-path", default="", type=str, help="the directory used to save the trained models")
parser.add_argument("--name", default="", type=str,
                    help="name of experiment")

parser.add_argument("--epochs", default=100, type=int, metavar="N",
                    help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N",
                    help="manual epoch number (useful on restarts)")
parser.add_argument("-b", "--batch-size", default=64, type=int,
                    metavar="N",
                    help="mini-batch size (default: 256 for add), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel")
parser.add_argument("-p", "--print-freq", default=50, type=int,
                    metavar="N", help="print frequency (default: 10)")
parser.add_argument("--save-ckpt",default=True, action="store_true", help="")

# args of optimizer
parser.add_argument("--optim", default="adam", type=str, help="optimizer (default: adam)")
parser.add_argument("--lr", "--learning-rate", default=1e-1, type=float,
                    metavar="LR", help="initial learning rate", dest="lr")
parser.add_argument("--wd", "--weight-decay", default=0, type=float,
                    metavar="W", help="weight decay (default: 1e-4)",
                    dest="weight_decay")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M",
                    help="momentum")
# Cosine learning rate
parser.add_argument("--cos-lr", action="store_true", default=False,
                    help="whether to use cosine learning rate")

# args of spiking neural networks
parser.add_argument("--threshold", type=float, default=0.5, help="neuronal threshold (default: 0.5)")
parser.add_argument("--time-window", type=int, default=500, help="total time steps (default: 500)")
parser.add_argument("--decay", type=float, default=0.5, help="decay factor (default: 0.5)")
parser.add_argument("--alpha", type=float, default=1., help="scaling factor of surrogate gradient (default 1.0)")
parser.add_argument("--learning-rule", default="STBP", type=str, help="[STBP|SDBP]")
parser.add_argument("--detach-mem", action="store_true", default=False, help="")
parser.add_argument("--detach-reset", action="store_true", default=False, help="")
parser.add_argument("--grad-clip", type=float, default=1.0)

parser.add_argument("--neuron", default="lif", type=str, help="[lif, plif, glif, alif, clif, celif, tclif, spsn, dhsnn, lmh, adlif, pmsn, ltc, ssm, psn]")
parser.add_argument("--recurrent", default=False, action="store_true", help="Feedforward or recurrent")

parser.add_argument("--hidden-dim", nargs= "+", default=[128, 256, 256], type=int, metavar="N",help="model architecture")


# args of celif
parser.add_argument("--beta", type=float, default=0.02, help="beta for the celif")
# args of tcn
parser.add_argument("--ksize", type=int, default=7, help="kernel size (default: 7)")
# args of transformer
parser.add_argument("--nhead", type=int, default=2,
                    help="the number of heads in the encoder/decoder of the transformer model")


def main():
    args = parser.parse_args()
    if args.save_path == "":
        save_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.recurrent:
            save_path = "exp/" + args.dataset  + "/" + args.name + args.net + "_" + args.dataset + "_" + str(args.time_window) + "_" + args.neuron + "_" + "FB_" + str(args.seed) + "_" + save_path
        else:
            save_path = "exp/" + args.dataset  + "/" + args.name + args.net + "_" + args.dataset + "_" + str(args.time_window) + "_" + args.neuron + "_" + "FF_" + str(args.seed) + "_" + save_path
    else:
        save_path = args.save_path
    print(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Logging settings
    setup_logging(os.path.join(save_path, "log.txt"))
    logging.info("saving to:" + str(save_path))


    is_cuda = torch.cuda.is_available()
    assert is_cuda, "CPU is not supported!"
    device = torch.device("cuda" if is_cuda else "cpu")
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
    args.gpu = "cuda"

    with open(save_path + "/args.json", "w") as fid:
        json.dump(args.__dict__, fid, indent=2)

    logging.info("args:" + str(args))

    data_path = args.data_path + "/SSL"
    # Check if preprocessed data exists; if so, load it directly
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
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=(train_sampler is None),
                                               pin_memory=train_pin_memory,
                                               sampler=train_sampler,
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             pin_memory=test_pin_memory,
                                             shuffle=False)

    print("Dataloader finish")
    # TODO: Build spiking model
    args.surrogate = "triangle"
    surro_grad = SurrogateGradient(func_name=args.surrogate, a=args.alpha)
    exec_mode = "serial"
    if args.neuron == "lif":
        spiking_neuron = partial(Recurrent_LIF,
                                 decay=args.decay,
                                 threshold=args.threshold,
                                 time_step=args.time_window,
                                 surro_grad=surro_grad,
                                 exec_mode=exec_mode,
                                 recurrent=args.recurrent,
                                 cut_grad=True if args.learning_rule == "SDBP" else False
                                 )
    elif args.neuron == "ltc":
        spiking_neuron = partial(LTC,
                                 decay=args.decay,
                                 threshold=args.threshold,
                                 time_step=args.time_window,
                                 surro_grad=surro_grad,
                                 exec_mode=exec_mode,
                                 recurrent=args.recurrent
                                 )
    elif args.neuron == "celif":
        beta = args.beta
        spiking_neuron = partial(CELIF,
                                 decay=args.decay,
                                 threshold=args.threshold,
                                 time_step=args.time_window,
                                 surro_grad=surro_grad,
                                 exec_mode=exec_mode,
                                 recurrent=args.recurrent,
                                 beta=beta
                                 )
    elif args.neuron == "spsn":
        spiking_neuron = partial(SPSN,
                                 decay=args.decay,
                                 threshold=args.threshold,
                                 time_step=args.time_window,
                                 surro_grad=surro_grad,
                                 exec_mode=exec_mode,
                                 recurrent=args.recurrent,
                                 k=16
                                 )
    else:
        raise NotImplementedError

    if args.net == "ffsnn":
        model = FFSNN(input_size=input_channels, hidden_size=args.hidden_dim, output_size=num_classes,
                      num_hidden_layers=len(args.hidden_dim),
                      spiking_neuron=spiking_neuron, dataset=args.dataset, neuron_type=args.neuron)
    elif args.net == "tcn":
        model = TCN(input_channels, num_classes, args.hidden_dim, kernel_size=args.ksize, dropout=0.0,
                    spiking_neuron=spiking_neuron, output_last_step=False)
    elif args.net == "gsn":
        model = LSTMNet(input_size=input_channels, hidden_size=args.hidden_dim, output_size=num_classes,
                        rnn_type="gsn", num_hidden_layers=len(args.hidden_dim), spiking_neuron=spiking_neuron)
    elif args.net == "spktransformer":
        model = SpkTransformerNet(input_size=input_channels, hidden_size=args.hidden_dim[0], output_size=num_classes,
                                  nhead=args.nhead, num_hidden_layers=len(args.hidden_dim), dropout=0.,
                                  spiking_neuron=spiking_neuron)
    elif args.net == "binaryssm":
        spiking_neuron = partial(S4D,
                                 dropout=0.1,
                                 lr=min(0.001, args.lr),
                                 binary="binary",
                                 threshold=0.0,
                                 time_step=args.time_window,
                                 surro_grad=surro_grad
                                 )
        model = SSM(input_size=input_channels, hidden_size=args.hidden_dim, output_size=num_classes,
                    num_hidden_layers=len(args.hidden_dim),
                    spiking_neuron=spiking_neuron, dataset=args.dataset, neuron_type=args.neuron)
    elif args.net == "gsnssm":
        spiking_neuron = partial(S4D,
                                 dropout=0.1,
                                 lr=min(0.001, args.lr),
                                 binary="GSN"
                                 )
        model = SSM(input_size=input_channels, hidden_size=args.hidden_dim, output_size=num_classes,
                    num_hidden_layers=len(args.hidden_dim),
                    spiking_neuron=spiking_neuron, dataset=args.dataset, neuron_type=args.neuron)
    else:
        raise NotImplementedError
    logging.info(str(model))
    para = count_parameters(model)
    logging.info(f"Parameter number: {para}")


    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optim == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    if args.net in ["ssm", "binaryssm", "gsnssm"]:
        optimizer, _ = setup_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs, optim=args.optim)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.2)
    best_cri = [0, 100]

    if args.cos_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    else:
        gamma = 0.8
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)

    model = model.to(device)
    standard_train(train_loader, val_loader, model, criterion, optimizer, scheduler, save_path, best_cri, device, args)

def standard_train(train_loader, val_loader, model, criterion, optimizer, scheduler, save_path, best_cri, device, args):
    loss_train_record  = []
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_acc1, train_loss, loss_train_record, train_mae = train_one_epoch(train_loader, model, criterion, optimizer, epoch,
                                                                    device, args, loss_train_record)
        scheduler.step()
        val_acc1,val_loss,val_mae = validate_one_epoch(val_loader, model, criterion, device, args)
        out_string = "Train Acc. {:.4f} Train MAE {:.4f} Test Acc. {:.4f} Test MAE {:.4f} \n ".format(train_acc1,
                                                                                                      train_mae,
                                                                                                      val_acc1,
                                                                                                      val_mae)
        logging.info(out_string)
        # remember best acc@1 and save checkpoint
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
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        images = images.transpose(0, 1).contiguous() # [T, B, C]
        target = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(images) # [T, B, N]
        # average across time
        output_mean = output.mean(0)
        loss = criterion(output_mean, target)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        mae = angular_distance_compute(target, torch.argmax(output_mean,dim=1))
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output_mean, target, topk=(1, 5))
        top1.update(acc1[0], target.size(0))
        top5.update(acc5[0], target.size(0))

        losses.update(loss.item(), target.size(0))
        loss_train_record.append(loss.item())
        maes.update(mae.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0 or (i + 1) == len(train_loader):
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

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            images = images.transpose(0,1).contiguous()  # [T, B, ..]
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images) # [T, B, N]
            output = output.mean(0)
            loss = criterion(output, target)
            mae = angular_distance_compute(target, torch.argmax(output, dim=1))
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], target.size(0))
            top5.update(acc5[0], target.size(0))
            losses.update(loss.item(), target.size(0))
            maes.update(mae.item(), target.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0 or (i + 1) == len(val_loader):
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

import argparse
import json
import os
import logging
import math
import time
import numpy as np
from functools import partial

import torch
from torch.cuda import amp
from datetime import datetime
import matplotlib.pyplot as plt

from src.benchmark.framework.utils.tools import set_random_seed, distributed, setup_logging, save_checkpoint, \
    AverageMeter, ProgressMeter, accuracy, count_parameters
from src.benchmark.framework.utils.dataloader import build_dataset
from src.benchmark.framework.network.trainer import SurrogateGradient
from src.benchmark.framework.network.neuron import LIF, LIFNode, RLIF, CELIF, SPSN, PMSN, LTC
from src.benchmark.framework.network.architecture import FFSNN, DvsGestureSNN, SpikingNet, TCN, LSTMNet, TransformerNet, \
    TextLSTMNet, TextTransformerNet, SpkTransformerNet, DVSLIPNet, SSMNet, StackedLSTM, SpikingGRUCell
from src.benchmark.framework.network.trainer import TriangleSurroGrad, RectangleSurroGrad

def dump_json(obj, fdir, name):
    """
    Dump python object in json
    """
    # if fdir and not os.path.exists(fdir):
    #     os.makedirs(fdir)
    with open(os.path.join(fdir, name), "w") as f:
        json.dump(obj, f, indent=4, sort_keys=False)

parser = argparse.ArgumentParser(description='PyTorch Training')
# args of datasets
parser.add_argument('--dataset', default='add', type=str,
                    help='dataset: [seqcifar10|dvsgesture|add|psmnist]')
parser.add_argument('--data-path', default='/datasets/MNIST',
                    help='path to dataset,')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--net', default='ffsnn', type=str,
                    help='networks')
parser.add_argument('--seed', default=1234, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--use-ddp', action='store_true', default=False, help='')
parser.add_argument('--amp', action='store_true', help='automatic mixed precision training')
parser.add_argument('--save-path', default='', type=str, help='the directory used to save the trained models')
parser.add_argument('--name', default='', type=str,
                    help='name of experiment')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=500, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-ckpt', action='store_true', default=False, help='')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# args of optimizer
parser.add_argument('--optim', default='sgd', type=str, help='optimizer (default: sgd)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
# Cosine learning rate
parser.add_argument('--cos-lr', action='store_true', default=False,
                    help='whether to use cosine learning rate')
parser.add_argument('--step-lr', action='store_true', default=False,
                    help='whether to use cosine learning rate')

# args of spiking neural networks
parser.add_argument('--threshold', type=float, default=0.5, help='neuronal threshold (default: 1)')
parser.add_argument('--time-window', type=int, default=20, help='total time steps (default: 10)')
parser.add_argument('--decay', type=float, default=0.5, help='decay factor (default: 5)')
parser.add_argument('--detach-mem', action='store_true', default=False, help='')
parser.add_argument('--detach-reset', action='store_true', default=False, help='')
parser.add_argument('--grad-clip', type=float, default=0.)
parser.add_argument('--neuron', default='lif', type=str, help='[lif, lifnode]')
parser.add_argument('--alpha', type=float, default=1., help='scaling factor of surrogate gradient (default 1.0)')

parser.add_argument('--hidden-size', type=int, default=512, help='')
parser.add_argument('--step-size', type=int, default=10, help='')
parser.add_argument('--hidden-layers', type=int, default=3, help='')

parser.add_argument('--truncated-t', type=int, default=10000, help='')
parser.add_argument('--learning-rule', default='stbp', type=str, help='[stbp|sdbp|notd|eprop|tbptt]')
parser.add_argument('--recurrent', action='store_true', default=False, help='')
parser.add_argument('--bn', default=None, type=str, help='[bn, tdbn, tebn, ln]')
parser.add_argument('--surrogate', default='rectangle', type=str,
                    help='[rectangle, triangle, sigmoid, multigauss, ASGL]')

parser.add_argument('--ksize', type=int, default=7, help='kernel size (default: 7)')
parser.add_argument('--rnn-type', default='lstm', type=str, help='[lif, lstm, gru]')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')

parser.add_argument('--inference', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--emb-dim', type=int, default=256, help='')
parser.add_argument('--dropout-emb', type=float, default=0.4, help='default: 0.4 on PTB')
parser.add_argument('--dropout', type=float, default=0.1, help='')


parser.add_argument('--final-step-cls', action='store_true', default=False, help='')
parser.add_argument('--data-cache', action='store_true', default=False, help='')

parser.add_argument('--min-length', type=int, default=0, help='')

parser.add_argument('--num-bins', type=int, default=1, help='')
parser.add_argument('--beta', type=float, default=0.02, help='')
parser.add_argument('--t-internal', type=int, default=1, help='')












def main():
    args = parser.parse_args()
    if args.save_path == '':
        save_path = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_path = save_path + args.name + '_' + str(args.seed)
        if args.amp:
            save_path += '_amp'
    else:
        save_path = args.save_path


    if args.use_ddp:
        distributed.init_distributed_mode(args)

    if args.use_ddp:
        torch.distributed.barrier()
        if distributed.is_main_process():
            from pathlib import Path

            Path(save_path).mkdir(parents=True, exist_ok=True)
            # Logging settings
            setup_logging(os.path.join(save_path, 'log.txt'))
            logging.info('saving to:' + str(save_path))
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Logging settings
        setup_logging(os.path.join(save_path, 'log.txt'))
        logging.info('saving to:' + str(save_path))

    if args.use_ddp:
        is_cuda = torch.cuda.is_available()
        assert is_cuda
        device = torch.device('cuda', args.gpu)
        set_random_seed(seed=args.seed, is_ddp=True)
        torch.backends.cudnn.benchmark = True
    else:
        is_cuda = torch.cuda.is_available()
        assert is_cuda, 'CPU is not supported!'
        device = torch.device('cuda' if is_cuda else 'cpu')
        set_random_seed(seed=args.seed, is_ddp=False)
        torch.backends.cudnn.benchmark = False
        args.gpu = 'cuda'

    if not args.use_ddp or (args.use_ddp and distributed.is_main_process()):
        with open(save_path + '/args.json', 'w') as fid:
            json.dump(args.__dict__, fid, indent=2)

    logging.info('args:' + str(args))

    train_dataset, val_dataset, input_channels, num_classes, collate_fn = build_dataset(dataset=args.dataset,
                                                                                        data_path=args.data_path,
                                                                                        seq_length=args.time_window,
                                                                                        data_cache=args.data_cache,
                                                                                        min_length=args.min_length,
                                                                                        num_bins=args.num_bins,
                                                                                        )
    print(f"train dataset: {len(train_dataset)}")
    print(f"val dataset: {len(val_dataset)}")
    if args.dataset == 'dvsgesture' and args.time_window >= 500:
        train_pin_memory = False if args.data_cache else True
        test_pin_memory = True
    else:
        train_pin_memory = True
        test_pin_memory = True
    train_sampler = None  # TODO: train_sampler when using ddp
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=(train_sampler is None),
                                               collate_fn=collate_fn,
                                               pin_memory=train_pin_memory,
                                               sampler=train_sampler,
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             collate_fn=collate_fn,
                                             pin_memory=test_pin_memory,
                                             shuffle=False)

    # TODO: Build spiking model
    if args.neuron == 'lif':
        surro_grad = SurrogateGradient(func_name=args.surrogate, a=args.alpha)
        exec_mode = "serial"
        spiking_neuron = partial(RLIF,
                                 decay=args.decay,
                                 threshold=args.threshold,
                                 time_step=args.time_window,
                                 surro_grad=surro_grad,
                                 exec_mode=exec_mode,
                                 recurrent=args.recurrent,
                                 learning_rule=args.learning_rule,
                                 truncated_t=args.truncated_t,
                                 )
        args.multi_step = True
    elif args.neuron == 'ltc':
        surro_grad = SurrogateGradient(func_name=args.surrogate, a=args.alpha)
        exec_mode = "serial"
        b_j0 = 0.2
        spiking_neuron = partial(LTC,
                                 decay=args.decay,
                                 threshold=args.threshold,
                                 time_step=args.time_window,
                                 surro_grad=surro_grad,
                                 exec_mode=exec_mode,
                                 recurrent=args.recurrent,
                                 b_j0=b_j0
                                 )
        args.multi_step = True
    elif args.neuron == 'celif':
        surro_grad = SurrogateGradient(func_name=args.surrogate, a=args.alpha)
        exec_mode = "serial"
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
        args.multi_step = True
    elif args.neuron == 'spsn':
        surro_grad = SurrogateGradient(func_name=args.surrogate, a=args.alpha)
        exec_mode = "serial"
        spiking_neuron = partial(SPSN,
                                 decay=args.decay,
                                 threshold=args.threshold,
                                 time_step=args.time_window,
                                 surro_grad=surro_grad,
                                 exec_mode=exec_mode,
                                 recurrent=args.recurrent
                                 )
        args.multi_step = True
    elif args.neuron == 'pmsn':
        surro_grad = SurrogateGradient(func_name=args.surrogate, a=args.alpha)
        exec_mode = "serial"
        spiking_neuron = partial(PMSN,
                                 decay=args.decay,
                                 threshold=args.threshold,
                                 time_step=args.time_window,
                                 surro_grad=surro_grad,
                                 exec_mode=exec_mode,
                                 recurrent=args.recurrent
                                 )
        args.multi_step = True
    elif args.neuron == 'lifnode':
        spiking_neuron = partial(LIFNode,
                                 decay_factor=args.decay,
                                 threshold=args.threshold,
                                 surrogate_function=TriangleSurroGrad.apply,
                                 hard_reset=True,
                                 detach_reset=args.detach_reset,
                                 detach_mem=args.detach_mem,
                                 )
        args.multi_step = False
    elif args.neuron == 'ann':
        spiking_neuron = None
        args.multi_step = False
    else:
        raise NotImplementedError

    if args.net == 'ffsnn':
        if args.dataset in ['dvsgesture', 'dvsslr']:
            if args.dataset == 'dvsgesture':
                input_size = 2048
            elif args.dataset == 'dvsslr':
                input_size = 11180
            model = DvsGestureSNN(input_size=input_size, hidden_size=args.hidden_size, output_size=num_classes, num_hidden_layers=args.hidden_layers,
                                  spiking_neuron=spiking_neuron, bn=args.bn, final_step_cls=args.final_step_cls, args=args)
        elif args.dataset == 'add':
            model = SpikingNet(input_size=2, hidden_size=args.hidden_size, output_size=1,
                               num_hidden_layers=args.hidden_layers,
                               spiking_neuron=spiking_neuron, bn=args.bn, recurrent=args.recurrent, args=args)
        elif args.dataset == 'binadd':
            model = SpikingNet(input_size=2, hidden_size=args.hidden_size, output_size=num_classes,
                               num_hidden_layers=args.hidden_layers,
                               spiking_neuron=spiking_neuron, bn=args.bn, recurrent=args.recurrent, args=args)
        elif args.dataset in ['gsc', 'ssc', 'ssc2', 'ecg']:
            model = SpikingNet(input_size=input_channels, hidden_size=args.hidden_size, output_size=num_classes,
                               num_hidden_layers=args.hidden_layers,
                               spiking_neuron=spiking_neuron, bn=args.bn, recurrent=args.recurrent, args=args, dropout=args.dropout)
        elif args.dataset in ['psmnist', 'smnist']:
            model = SpikingNet(input_size=1, hidden_size=[64, 256, 256], output_size=num_classes, num_hidden_layers=3,
                               spiking_neuron=spiking_neuron, bn=args.bn, recurrent=args.recurrent, args=args)
        elif args.dataset == 'dvslip':
            input_size = 88 * 88 * 2
            # model = DVSLIPNet(num_classes=num_classes, spiking_neuron=spiking_neuron, args=args)
            model = SpikingNet(input_size=input_size, hidden_size=args.hidden_size, output_size=num_classes,
                               num_hidden_layers=args.hidden_layers,
                               spiking_neuron=spiking_neuron, bn=args.bn, recurrent=args.recurrent, args=args,
                              )
    elif args.net == 'tcn':
        if args.dataset in ['add', 'binadd']:
            channel_sizes = [args.hidden_size] * args.hidden_layers
            model = TCN(2, num_classes, channel_sizes, kernel_size=args.ksize, dropout=0.0, spiking_neuron=spiking_neuron,
                        output_last_step=False, t_internal=args.t_internal)
        elif args.dataset == 'psmnist':
            channel_sizes = [args.hidden_size] * args.hidden_layers
            model = TCN(1, num_classes, channel_sizes, kernel_size=args.ksize, dropout=0.0,
                        spiking_neuron=spiking_neuron, output_last_step=False, t_internal=args.t_internal)
        elif args.dataset == 'dvslip':
            channel_sizes = [args.hidden_size] * args.hidden_layers
            model = TCN(88 * 88 * 2, num_classes, channel_sizes, kernel_size=args.ksize, dropout=0.0,
                        spiking_neuron=spiking_neuron, output_last_step=False, use_flatten=True)
        elif args.dataset == 'dvsgesture':
            channel_sizes = [args.hidden_size] * args.hidden_layers
            model = TCN(2048, num_classes, channel_sizes, kernel_size=args.ksize, dropout=0.0,
                        spiking_neuron=spiking_neuron, output_last_step=False, use_pool=True)
    elif args.net == 'lstm':
        if args.dataset == 'dvsgesture':
            model = LSTMNet(input_size=2048, hidden_size=args.hidden_size, output_size=num_classes, rnn_type=args.rnn_type,
                            num_hidden_layers=args.hidden_layers, spiking_neuron=spiking_neuron, spiking=False, dvs_pooling=True)
        elif args.dataset == 'add':
            model = LSTMNet(input_size=2, hidden_size=args.hidden_size, output_size=1, rnn_type=args.rnn_type,
                            num_hidden_layers=args.hidden_layers, spiking_neuron=spiking_neuron, spiking=False)
        elif args.dataset == 'binadd':
            model = LSTMNet(input_size=2, hidden_size=args.hidden_size, output_size=num_classes, rnn_type=args.rnn_type,
                            num_hidden_layers=args.hidden_layers, spiking_neuron=spiking_neuron, spiking=False)
        elif args.dataset == 'psmnist':
            model = LSTMNet(input_size=1, hidden_size=[64, 64, args.hidden_size], output_size=num_classes, rnn_type=args.rnn_type,
                            num_hidden_layers=args.hidden_layers, spiking_neuron=spiking_neuron, spiking=False)
    elif args.net == 'spklstm':
        if args.dataset == 'add':
            model = LSTMNet(input_size=2, hidden_size=args.hidden_size, output_size=1, rnn_type=args.rnn_type,
                            num_hidden_layers=args.hidden_layers, spiking_neuron=spiking_neuron, spiking=True)
        elif args.dataset == 'binadd':
            model = LSTMNet(input_size=2, hidden_size=args.hidden_size, output_size=num_classes, rnn_type=args.rnn_type,
                            num_hidden_layers=args.hidden_layers, spiking_neuron=spiking_neuron, spiking=True)
        elif args.dataset == 'psmnist':
            model = LSTMNet(input_size=1, hidden_size=args.hidden_size, output_size=num_classes, rnn_type=args.rnn_type,
                            num_hidden_layers=args.hidden_layers, spiking_neuron=spiking_neuron, spiking=True)
        elif args.dataset == 'dvslip':
            input_size = 88 * 88 * 2
            # model = DVSLIPNet(num_classes=num_classes, spiking_neuron=spiking_neuron, args=args)
            model = LSTMNet(input_size=input_size, hidden_size=args.hidden_size, output_size=num_classes, rnn_type=args.rnn_type,
                            num_hidden_layers=args.hidden_layers, spiking_neuron=spiking_neuron, spiking=True, use_flatten=True)
        elif args.dataset == 'dvsgesture':
            model = LSTMNet(input_size=2048, hidden_size=args.hidden_size, output_size=num_classes, rnn_type=args.rnn_type,
                            num_hidden_layers=args.hidden_layers, spiking_neuron=spiking_neuron, spiking=True, dvs_pooling=True)
    elif args.net == 'transformer':
        if args.dataset == 'psmnist':
            model = TransformerNet(input_size=1, hidden_size=args.hidden_size, output_size=num_classes,
                                   nhead=args.nhead, num_hidden_layers=args.hidden_layers, dropout=0.0)
        elif args.dataset == 'binadd':
            model = TransformerNet(input_size=2, hidden_size=args.hidden_size, output_size=num_classes, nhead=args.nhead,
                                   num_hidden_layers=args.hidden_layers, dropout=0.)
        elif args.dataset == 'add':
            model = TransformerNet(input_size=2, hidden_size=args.hidden_size, output_size=1, nhead=args.nhead,
                                   num_hidden_layers=args.hidden_layers, dropout=0.)
        if args.dataset in ['dvsgesture', 'cifar10dvs' ]:
            model = TransformerNet(input_size=2048, hidden_size=args.hidden_size, output_size=num_classes, nhead=args.nhead,
                                   num_hidden_layers=args.hidden_layers, dropout=0., use_pool=True)
    elif args.net == 'spktransformer':
        if args.dataset == 'add':
            model = SpkTransformerNet(input_size=2, hidden_size=args.hidden_size, output_size=1, nhead=args.nhead,
                                   num_hidden_layers=args.hidden_layers, dropout=0., spiking_neuron=spiking_neuron)
        elif args.dataset == 'binadd':
            model = SpkTransformerNet(input_size=2, hidden_size=args.hidden_size, output_size=num_classes, nhead=args.nhead,
                                   num_hidden_layers=args.hidden_layers, dropout=0., spiking_neuron=spiking_neuron, T=args.t_internal)
        elif args.dataset == 'psmnist':
            model = SpkTransformerNet(input_size=1, hidden_size=args.hidden_size, output_size=num_classes, nhead=args.nhead,
                                   num_hidden_layers=args.hidden_layers, dropout=0., spiking_neuron=spiking_neuron, T=args.t_internal)
        elif args.dataset == 'dvslip':
            model = SpkTransformerNet(input_size=88 * 88 * 2, hidden_size=args.hidden_size, output_size=num_classes, nhead=args.nhead,
                                   num_hidden_layers=args.hidden_layers, dropout=0., spiking_neuron=spiking_neuron, T=1, use_flatten=True)
        elif args.dataset in ['dvsgesture', 'cifar10dvs']:
            model = SpkTransformerNet(input_size=2048, hidden_size=args.hidden_size, output_size=num_classes,
                                      nhead=args.nhead,
                                      num_hidden_layers=args.hidden_layers, dropout=0., spiking_neuron=spiking_neuron, use_pool=True)
    elif args.net == 'binaryssm':
        from src.benchmark.framework.network.neuron.S4DModule import S4D
        surro_grad = SurrogateGradient(func_name=args.surrogate, a=args.alpha)
        spiking_neuron = partial(S4D,
                                 dropout=args.dropout,
                                 lr=min(0.001, args.lr),
                                 binary='binary',
                                 threshold=0.,
                                 time_step=args.time_window,
                                 surro_grad=surro_grad
                                 )
        if args.dataset == 'dvslip':
            input_size = 88 * 88 * 2
        else:
            raise NotImplementedError
        model = SSMNet(input_size=input_size, hidden_size=args.hidden_size, output_size=num_classes,
                      num_hidden_layers=args.hidden_layers,
                      spiking_neuron=spiking_neuron)
    elif args.net == 'gsussm':
        from src.benchmark.framework.network.neuron.S4DModule import S4D
        surro_grad = SurrogateGradient(func_name=args.surrogate, a=args.alpha)
        spiking_neuron = partial(S4D,
                                 dropout=args.dropout,
                                 lr=min(0.001, args.lr),
                                 binary='GSU'
                                 )
        if args.dataset == 'dvslip':
            input_size = 88 * 88 * 2
        else:
            raise NotImplementedError
        model = SSMNet(input_size=input_size, hidden_size=args.hidden_size, output_size=num_classes,
                      num_hidden_layers=args.hidden_layers,
                      spiking_neuron=spiking_neuron)
    else:
        raise NotImplementedError
    logging.info(str(model))

    # TODO: Model profile
    # model.eval()
    para = count_parameters(model)
    # logging.info(f"Parameter number: {para}")

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=args.momentum, nesterov=True)
    elif args.optim == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    # assert args.cos_lr
    if args.net in ['binaryssm', 'gsussm']:
        from src.benchmark.framework.network.neuron.S4DModule import setup_optimizer
        optimizer, _ = setup_optimizer(
            model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs, optim=args.optim)
    # define loss function (criterion) and optimizer

    if args.dataset == 'add':
        criterion = torch.nn.MSELoss()
        best_acc1 = float('inf')
    elif args.dataset == 'binadd':
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.2)
        best_acc1 = 0
    else:
        criterion = torch.nn.CrossEntropyLoss()
        best_acc1 = 0
    if args.cos_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    elif args.step_lr:
        if args.dataset == 'add':
            gamma = 0.9
        else:
            gamma = 0.8
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=gamma)
    else:
        scheduler = None
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    if args.use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
            find_unused_parameters=True,
        )
    else:
        model = torch.nn.DataParallel(model).cuda()

    standard_train(train_loader, val_loader, model, criterion, optimizer, scheduler, save_path, best_acc1, scaler,
                       args)


def standard_train(train_loader, val_loader, model, criterion, optimizer, scheduler, save_path, best_acc1, scaler,
                   args):
    all_val_res = []
    loss_train_record = []

    for epoch in range(args.start_epoch, args.epochs):
        if args.use_ddp:
            train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train_acc1, train_loss, loss_train_record = train_one_epoch(train_loader, model, criterion, optimizer, epoch, scaler, args, loss_train_record)
        if scheduler is not None:
            scheduler.step()
        if not args.use_ddp or (args.use_ddp and distributed.is_main_process()):
            # evaluate on validation set
            val_acc1, val_loss = validate_one_epoch(val_loader, model, criterion, args)

            out_string = 'Train Acc. {:.4f} Test Acc. {:.4f} lr {:.4f}\t'.format(train_acc1, val_acc1,
                                                                                 optimizer.param_groups[0]["lr"])
            all_val_res.append(val_acc1.cpu())
            logging.info(out_string)

            is_best = val_acc1 > best_acc1
            best_acc1 = max(val_acc1, best_acc1)

            if args.save_ckpt:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, filename=os.path.join(save_path, 'checkpoint.pth.tar'), save_path=save_path)
    np.savetxt(os.path.join(save_path, 'val_res.txt'), all_val_res, delimiter=',', fmt='%.4f')
    training_record = {
        'loss_train_record': loss_train_record,
    }
    dump_json(training_record, save_path, 'loss_train_record.txt')
    logging.info(f'Best accuracy: {best_acc1}')

    logging.info("Finished.")




def train_one_epoch(train_loader, model, criterion, optimizer, epoch, scaler, args, loss_train_record):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.cuda(args.gpu, non_blocking=True)  # dvsgesture: [B, T, 2, 128, 128]
        target = labels.cuda(args.gpu, non_blocking=True)

        if args.dataset in ['dvsgesture', 'cifar10dvs', 'dvsslr', 'dvslip'] and not args.data_cache:
            images = images.permute(1, 0, 2, 3, 4).contiguous()  # [T, B, 2, 128, 128]
        elif args.dataset in ['add', 'psmnist', 'smnist', 'imdb', 'binadd', 'ecg', '20news']:
            images = images.transpose(0, 1).contiguous()  # [T, B, N]
        optimizer.zero_grad()
        if args.amp:
            with amp.autocast():
                output = model(images, multi_step=args.multi_step)  # [T, B, N]
                if args.final_step_cls:
                    output = output[-1]
                else:
                    output = output.mean(0)
                loss = criterion(output, target)

                scaler.scale(loss).backward()

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(images, multi_step=args.multi_step)  # [T, B, N]
            if args.final_step_cls:
                output = output[-1]
            else:
                output = output.mean(0)
            loss = criterion(output, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], target.size(0))
        top5.update(acc5[0], target.size(0))
        losses.update(loss.item(), target.size(0))
        loss_train_record.append(loss.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0 or (i + 1) == len(train_loader):
            progress.display(i + 1)

    return top1.avg, losses.avg, loss_train_record


def validate_one_epoch(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            if args.dataset in ['dvsgesture', 'cifar10dvs', 'dvsslr', 'dvslip'] and not args.data_cache:
                images = images.permute(1, 0, 2, 3, 4).contiguous()  # [T, B, 2, 128, 128]
            elif args.dataset in ['add', 'psmnist', 'smnist', 'imdb', 'binadd', 'ecg', '20news']:
                images = images.transpose(0, 1).contiguous()  # [T, B, N]

            # compute output
            output = model(images, multi_step=args.multi_step)

            if args.final_step_cls:
                output = output[-1]
            else:
                output = output.mean(0)
            loss = criterion(output, target)

            # measure accuracy and record loss
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

if __name__ == '__main__':
    main()

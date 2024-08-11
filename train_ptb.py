import argparse
import json
import os
import logging
import math
import time
from functools import partial
import torch
from datetime import datetime

from src.benchmark.framework.utils.tools import set_random_seed, setup_logging, save_checkpoint, AverageMeter, ProgressMeter
from src.benchmark.framework.utils.dataset import build_lm_dataloader, get_batch
from src.benchmark.framework.network.trainer import SurrogateGradient
from src.benchmark.framework.network.neuron import LIF
from src.benchmark.framework.network.architecture import LMSNN

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default="PTB", type=str, help="dataset: [PTB|WT2]")
parser.add_argument("--data-path", default="./datasource", help="path to dataset,")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)")

parser.add_argument("--seed", default=1234, type=int, help="seed for initializing training. ")
parser.add_argument("--save-path", default="", type=str, help="the directory used to save the trained models")
parser.add_argument("--name", default="", type=str, help="name of experiment")

parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument( "-b", "--batch-size", 
    default=20, type=int, metavar="N",
    help = "mini-batch size (default: 128), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel"
)
parser.add_argument("-p", "--print-freq", default=100, type=int, metavar="N", help="print frequency (default: 10)")
parser.add_argument("--save-ckpt", action="store_true", default=True, help="")
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

# args of optimizer
parser.add_argument("--optim", default="sgd", type=str, help="optimizer (default: sgd)")
parser.add_argument("--lr", "--learning-rate", default=3, type=float, metavar="LR", help="initial learning rate", dest="lr")
parser.add_argument("--wd", "--weight-decay", default=1.2e-6, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")

# args of spiking neural networks
parser.add_argument("--threshold", type=float, default=0.5, help="neuronal threshold (default: 1)")
parser.add_argument("--time-step", type=int, default=70, help="total time steps (default: 10)")
parser.add_argument("--nlayers", type=int, default=2, help="total time steps (default: 10)")
parser.add_argument("--decay", type=float, default=0.5, help="decay factor (default: 5)")
parser.add_argument("--detach-mem", action="store_true", default=False, help="")
parser.add_argument("--detach-reset", action="store_true", default=False, help="")
parser.add_argument("--neuron", default="lif", type=str, help="[lif]")
parser.add_argument("--alpha", type=float, default=1., help="scaling factor of surrogate gradient (default 1.0)")

parser.add_argument("--grad-clip", type=float, default=0.25)

parser.add_argument("--dropout-emb", type=float, default=0.4, help="default: 0.4 on PTB")
parser.add_argument("--dropout-words", type=float, default=0.1, help="default: 0.1 on PTB")
parser.add_argument("--dropout-forward", type=float, default=0.25, help="default: 0.25 on PTB")
parser.add_argument("--dropout", type=float, default=0.4, help="default: 0.4 on PTB")
parser.add_argument("--rnn-type", default="lif", type=str, help="[lif, lstm, gru]")
parser.add_argument("--surrogate", default="rectangle", type=str, help="[rectangle|triangle|multigauss]")
parser.add_argument("--emb-dim", type=int, default=400, help="total time steps (default: 10)")
parser.add_argument("--hidden-dim", type=int, default=1100, help="total time steps (default: 10)")
parser.add_argument("--recurrent", action="store_true", default=False, help="")


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
    device = torch.device("cuda")
    set_random_seed(seed=args.seed)
    torch.backends.cudnn.benchmark = False
    args.gpu = "cuda"

    with open(save_path + "/args.json", "w") as fid:
        json.dump(args.__dict__, fid, indent=2)

    logging.info("args:" + str(args))

    train_dataset, val_dataset, test_dataset, vocab_size = build_lm_dataloader(dataset=args.dataset, data_path=args.data_path, train_batch_size=args.batch_size)
    logging.info(f"Dataset {args.dataset} has {vocab_size} tokens")

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

    model = LMSNN(
        rnn_type = args.rnn_type,
        nlayers = args.nlayers,
        emb_dim = args.emb_dim,
        hidden_dim = args.hidden_dim,
        vocab_size = vocab_size,
        dropout_words = args.dropout_words,
        dropout_embedding = args.dropout_emb,
        dropout_forward = args.dropout_forward,
        dropout = args.dropout,
        spiking_neuron = spiking_neuron,
        args = args
    )

    logging.info(str(model))
    logging.info(f"Model number of parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1024 / 1024):.4f} M")

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.0)
    elif args.optim == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.0, 0.999))
    else:
        raise NotImplementedError

    criterion = torch.nn.CrossEntropyLoss()

    scheduler = None
    scaler = None

    best_val_ppl = float("inf")

    model = model.cuda()

    try:
        standard_train(train_dataset, val_dataset, model, criterion, optimizer, scheduler, save_path, best_val_ppl, scaler, vocab_size, args)
    except KeyboardInterrupt:
        logging.info("-" * 89)
        logging.info("Exiting from training early")
    # Evaluate the best model on the test dataset
    best_model_checkpoint = torch.load(os.path.join(save_path, "model_best.pth.tar"))
    model.load_state_dict(best_model_checkpoint["state_dict"])
    test_ppl = validate_one_epoch(test_dataset, model, criterion, vocab_size, 1, args)
    logging.info("=" * 89)
    logging.info(f"| End of training | test ppl {test_ppl:8.2f}")
    logging.info("=" * 89)


def standard_train(train_loader, val_loader, model, criterion, optimizer, scheduler, save_path, best_val_ppl, scaler, vocab_size, args):
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_ppl, train_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, scaler, vocab_size, args)

        # evaluate on validation set
        val_ppl = validate_one_epoch(val_loader, model, criterion, vocab_size, 10, args)
        out_string = "Train ppl. {:8.2f} Val ppl {:8.2f} \t".format(train_ppl, val_ppl)
        logging.info(out_string)
        # remember best acc@1 and save checkpoint
        is_best = val_ppl < best_val_ppl
        best_val_ppl = min(val_ppl, best_val_ppl)
        if args.save_ckpt:
            save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_val_ppl": best_val_ppl,
                "optimizer": optimizer.state_dict(),
            }, is_best, filename=os.path.join(save_path, "checkpoint.pth.tar"), save_path=save_path)
    logging.info(f"Best best_val_ppl: {best_val_ppl}")


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is not None:
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, scaler, ntokens, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    seq_length = args.time_step
    num_batches = (train_loader.size(0) - 1) // seq_length

    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )

    # Switch to train mode
    model.train()
    end = time.time()

    hidden = model.init_hidden(args.batch_size)

    for batch_index, i in enumerate(range(0, train_loader.size(0) - 1, seq_length)):
        data, targets = get_batch(train_loader, i, seq_len=seq_length, batch_first=False)
        # Measure data loading time
        data_time.update(time.time() - end)
        data = data.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
        optimizer.zero_grad()

        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)

        loss = criterion(output.view(-1, ntokens), targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        losses.update(loss.item(), data.numel())

        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_index + 1) % args.print_freq == 0 or (batch_index + 1) == num_batches:
            progress.display(batch_index + 1)

    return math.exp(losses.avg), losses.avg


def validate_one_epoch(val_loader, model, criterion, ntokens, eval_batch_size, args):
    losses = AverageMeter("Loss", ":.4e")
    # switch to evaluate mode
    model.eval()
    seq_length = args.time_step
    iter_range = range(0, val_loader.size(0) - 1, seq_length)
    with torch.no_grad():
        # initialize hidden states
        hidden = model.init_hidden(eval_batch_size)
        # iterate evaluation data
        for num_iter, index in enumerate(iter_range):
            data, targets = get_batch(val_loader, index, seq_len=seq_length, batch_first=False)
            data = data.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
            losses.update(loss.item(), data.numel())
    return math.exp(losses.avg)


if __name__ == "__main__":
    main()
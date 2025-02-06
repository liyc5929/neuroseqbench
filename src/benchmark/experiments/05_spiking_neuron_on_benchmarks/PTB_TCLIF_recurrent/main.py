import argparse
import json
import os
import sys
import logging
import math
import time
import torch
import toml
from functools import partial
from datetime import datetime

# Check and add current working directory
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

from src.benchmark.framework.utils.tools import setup_logging, save_checkpoint, AverageMeter, ProgressMeter
from src.benchmark.framework.network.trainer import SurrogateGradient
from src.benchmark.framework.network.neuron import TCLIF
from src.benchmark.framework.network.structure import LMSNN
from src.benchmark.framework.utils.dataset import PennTreebank


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="GPU device number")
    parser.add_argument("--config", type=str, default="config.toml", help="TOML config file name")
    parser.add_argument("--data_root", type=str, default="/benchmark_data")


    parser.add_argument("--save-path", default="", type=str, help="the directory used to save the trained models")
    parser.add_argument("--save-ckpt", action="store_true", default=True, help="")
    
    # args of spiking neural networks
    parser.add_argument("--detach-mem", action="store_true", default=False, help="")
    parser.add_argument("--detach-reset", action="store_true", default=False, help="")
    
    parser.add_argument("--dropout-emb", type=float, default=0.4, help="default: 0.4 on PTB")
    parser.add_argument("--dropout-words", type=float, default=0.1, help="default: 0.1 on PTB")
    parser.add_argument("--dropout-forward", type=float, default=0.25, help="default: 0.25 on PTB")
    parser.add_argument("--dropout", type=float, default=0.4, help="default: 0.4 on PTB")
    

    # Get `argparse` arguments
    args = parser.parse_args()

    # Get `toml` arguments
    with open(args.config, "r") as fp:
        config = toml.load(fp)

    # General setting
    args.seed          = config.get("seed", 1234)

    # Hyperparameter setting
    args.max_epoch     = config.get("max_epoch", 100)
    args.batch_size    = config.get("batch_size", 20)
    args.time_step     = config.get("time_step", 70)
    args.learning_rate = config.get("learning_rate", 3.0)
    args.weight_decay  = config.get("weight_decay", 1.2e-6)
    args.optimizer     = config.get("optimizer", "SGD")
    args.grad_clip     = config.get("grad_clip", 0.25)

    # Architecture setting
    args.embedding_dim = config.get("embedding_dim", 400)
    args.hidden_dim    = config.get("hidden_dim", 1100)
    args.num_layers    = config.get("num_layers", 2)

    # Neuron setting
    args.neuron_decay  = config.get("neuron_decay", 0.5)
    args.neuron_thresh = config.get("neuron_thresh", 0.5)
    args.recurrent     = config.get("recurrent", False)

    return args


def main():
    args = parse_args()

    if args.save_path == "":
        save_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = save_path + "_" + str(args.seed)
    else:
        save_path = args.save_path

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

    T = args.time_step
    B = args.batch_size
    train_dataset = PennTreebank(root=os.path.join(args.data_root, "PennTreebank"), subset="train", time_step=T, chunk_num=B, device=device)
    val_dataset   = PennTreebank(root=os.path.join(args.data_root, "PennTreebank"), subset="valid", time_step=T, chunk_num=10, device=device)
    test_dataset  = PennTreebank(root=os.path.join(args.data_root, "PennTreebank"), subset="test",  time_step=T, chunk_num=1, device=device)
    vocab_size = 10000

    surro_grad = SurrogateGradient(func_name="rectangle", a=1.0)
    exec_mode = "serial"
    beta1 = 1.0
    beta2 = -1.0
    spiking_neuron = partial(TCLIF,
        decay=args.neuron_decay,
        threshold=args.neuron_thresh,
        time_step=args.time_step,
        surro_grad=surro_grad,
        exec_mode=exec_mode,
        recurrent=args.recurrent,
        beta1=beta1,
        beta2=beta2
    )

    model = LMSNN(
        rnn_type="tclif",
        nlayers=2,
        emb_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=vocab_size,
        dropout_words=args.dropout_words,
        dropout_embedding=args.dropout_emb,
        dropout_forward=args.dropout_forward,
        dropout=args.dropout,
        spiking_neuron=spiking_neuron,
        args=args,
    )
    logging.info(str(model))
    logging.info(f"Model number of parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1024 / 1024):.4f} M")

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.0)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.0, 0.999))
    else:
        raise NotImplementedError

    # Define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = None
    scaler = None
    best_val_ppl = float("inf")

    model = model.to(device)
    try:
        standard_train(train_dataset, val_dataset, model, criterion, optimizer, scheduler, save_path, best_val_ppl, scaler, vocab_size, device, args)
    except KeyboardInterrupt:
        logging.info("-" * 89)
        logging.info("Exiting from training early")
    best_model_checkpoint = torch.load(os.path.join(save_path, "model_best.pth.tar"))
    model.load_state_dict(best_model_checkpoint["state_dict"])
    test_ppl = validate_one_epoch(test_dataset, model, criterion, vocab_size, 1, device)
    logging.info("=" * 89)
    logging.info(f"| End of training | test ppl {test_ppl:8.2f}")
    logging.info("=" * 89)


def standard_train(train_loader, val_loader, model, criterion, optimizer, scheduler, save_path, best_val_ppl, scaler, vocab_size, device, args):
    for epoch in range(0, args.max_epoch):
        # train for one epoch
        train_ppl, train_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, scaler, vocab_size, device, args)
        # Evaluate on validation set
        val_ppl = validate_one_epoch(val_loader, model, criterion, vocab_size, 10, device)
        out_string = "Train ppl. {:8.2f} Val ppl {:8.2f} \t".format(train_ppl, val_ppl)
        logging.info(out_string)
        # Remember best acc@1 and save checkpoint
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


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, scaler, ntokens, device, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    num_batches = len(train_loader)

    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # Switch to train mode
    model.train()
    end = time.time()
    hidden = model.init_hidden(args.batch_size)

    for batch_index, (data, targets) in enumerate(train_loader): 
        # Measure data loading time
        data_time.update(time.time() - end)
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        output, hidden = model(data, hidden)

        loss = criterion(output.view(-1, ntokens), targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_rec = criterion(output.view(-1, ntokens), targets)

        losses.update(loss_rec.item(), data.numel())

        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_index + 1) % 100 == 0 or (batch_index + 1) == num_batches:
            progress.display(batch_index + 1)

    return math.exp(losses.avg), losses.avg


def validate_one_epoch(val_loader, model, criterion, ntokens, eval_batch_size, device):
    losses = AverageMeter("Loss", ":.4e")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        # Initialize hidden states
        hidden = model.init_hidden(eval_batch_size)
        for _, (data, targets) in enumerate(val_loader): 
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output, hidden = model(data, hidden) # [T, B, N]
            loss = criterion(output.view(-1, ntokens), targets)
            losses.update(loss.item(), data.numel())
    return math.exp(losses.avg)


if __name__ == "__main__":
    main()

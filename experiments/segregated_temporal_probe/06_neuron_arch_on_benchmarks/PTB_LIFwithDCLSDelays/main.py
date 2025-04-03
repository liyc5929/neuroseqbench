import argparse
import json
import os
import logging
import math
import time
import toml
import torch
from datetime import datetime

from torch.nn import Module, Sequential, Embedding, ConstantPad1d, BatchNorm1d
from DCLS.construct.modules import Dcls1d
from neuroseqbench.utils.tools import setup_logging, save_checkpoint, AverageMeter, ProgressMeter
from neuroseqbench.utils.dataset import PennTreebank
from neuroseqbench.network.trainer import SurrogateGradient
from neuroseqbench.network.neuron import LIF, NonSpikingLIF
from neuroseqbench.network.structure import Permute, ANNSequential


class PTB_SNNDelay(Module):
    def __init__(self, 
        input_size, 
        hidden_size,
        output_size, 
        time_step,
        max_delay       = 25,
        spiking_neuron  = None,
    ):
        super(PTB_SNNDelay, self).__init__()
        # Register hyperparameters
        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.output_size    = output_size
        self.time_step      = time_step
        self.max_delay      = max_delay
        self.left_padding   = self.max_delay - 1
        self.right_padding  = (self.max_delay - 1) // 2
        self.spiking_neuron = spiking_neuron
        self.final_neuron   = NonSpikingLIF(
            decay      = self.spiking_neuron.decay,
            time_step  = self.spiking_neuron.time_step,
            exec_mode  = self.spiking_neuron.exec_mode,
        )

        class LockedDropout(Module):
            def __init__(self):
                super().__init__()
        
            def forward(self, x, dropout=0.5):
                if not self.training or not dropout:
                    return x
                m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
                mask = torch.autograd.Variable(m, requires_grad=False) / (1 - dropout)
                mask = mask.expand_as(x)
                return mask * x

        self.dropout              = 0.1
        self.vocabulary_size      = output_size
        self.embedded_dim         = self.input_size
        self.embedding_layer      = Embedding(self.vocabulary_size, self.embedded_dim)
        self.locked_dropout_layer = LockedDropout()

        # Delay learning model
        self.features = Sequential(
            # Calculate delays
            Permute(1, 2, 0),
            ConstantPad1d(padding=(self.left_padding, self.right_padding), value=0),
            Dcls1d(self.input_size, self.hidden_size, kernel_count=1, groups=1, dilated_kernel_size=self.max_delay, bias=False, version="gauss"),
            Permute(2, 0, 1),
            # Calculate neurons
            ANNSequential(BatchNorm1d(self.hidden_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),),
            self.spiking_neuron,

            # Calculate delays
            Permute(1, 2, 0),
            ConstantPad1d(padding=(self.left_padding, self.right_padding), value=0),
            Dcls1d(self.hidden_size, self.hidden_size, kernel_count=1, groups=1, dilated_kernel_size=self.max_delay, bias=False, version="gauss"),
            Permute(2, 0, 1),
            # Calculate neurons
            ANNSequential(BatchNorm1d(self.hidden_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),),
            self.spiking_neuron,

            # Calculate delays
            Permute(1, 2, 0),
            ConstantPad1d(padding=(self.left_padding, self.right_padding), value=0),
            Dcls1d(self.hidden_size, self.output_size, kernel_count=1, groups=1, dilated_kernel_size=self.max_delay, bias=False, version="gauss"),
            Permute(2, 0, 1),
            # Calculate neurons
            self.final_neuron,
        )

    def embedded_dropout(self, embed, words, dropout=0.1, scale=None):
        if dropout:
            mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
            masked_embed_weight = mask * embed.weight
        else:
            masked_embed_weight = embed.weight
        if scale:
            masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight
    
        padding_idx = embed.padding_idx
        if padding_idx is None:
            padding_idx = -1
    
        X = torch.nn.functional.embedding(
            words, masked_embed_weight,
            padding_idx, embed.max_norm, embed.norm_type,
            embed.scale_grad_by_freq, embed.sparse
        )
        return X

    def forward(self, inputs: torch.Tensor): # shape (T, B)
        embedded = self.embedded_dropout(self.embedding_layer, inputs, dropout=self.dropout if self.training else 0)
        tx       = self.locked_dropout_layer(embedded, dropout=self.dropout) # shape (T, B, self.embedded_dim)
        ty       = self.features(tx)
        predict  = self.locked_dropout_layer(ty, self.dropout)
        return predict[-self.time_step:].view(-1, self.vocabulary_size)


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
    args.seed          = config.get("seed", 42)

    # Hyperparameter setting
    args.max_epoch     = config.get("max_epoch", 100)
    args.time_step     = config.get("time_step", 70)
    args.batch_size    = config.get("batch_size", 20)
    args.learning_rate = config.get("learning_rate", 3)
    args.weight_decay  = config.get("weight_decay", 1.2e-6)
    args.momentum      = config.get("momentum", 0.9)
    args.optimizer     = config.get("optimizer", "SGD")
    args.grad_clip     = config.get("grad_clip", 0.25)

    # Architecture setting
    args.input_size    = config.get("input_size", 400)
    args.hidden_size   = config.get("hidden_size", 32)
    args.output_size   = config.get("output_size", 10000)
    args.max_delay     = config.get("max_delay", 25)

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

    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
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

    surro_grad     = SurrogateGradient(func_name="rectangle", a=1.0)
    spiking_neuron = LIF(
        decay      = 0.9, 
        threshold  = 0.5, 
        time_step  = 70, 
        surro_grad = surro_grad,
        exec_mode  = "serial"
    )
    model = PTB_SNNDelay(
        input_size     = args.input_size,
        hidden_size    = args.hidden_size,
        output_size    = args.output_size,
        time_step      = args.time_step,
        max_delay      = args.max_delay,
        spiking_neuron = spiking_neuron,
    ).to(device=device)

    logging.info(str(model))
    logging.info(f"Model number of parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1024 / 1024):.4f} M")

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.0)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.0, 0.999))
    else:
        raise NotImplementedError

    criterion = torch.nn.CrossEntropyLoss()
    best_val_ppl = float("inf")
    model = model.to(device)

    try:
        standard_train(train_dataset, val_dataset, model, criterion, optimizer, save_path, best_val_ppl, vocab_size, device, args)
    except KeyboardInterrupt:
        logging.info("-" * 89)
        logging.info("Exiting from training early")

    # Evaluate the best model on the test dataset
    best_model_checkpoint = torch.load(os.path.join(save_path, "model_best.pth.tar"), weights_only=True)
    model.load_state_dict(best_model_checkpoint["state_dict"])
    test_ppl = validate_one_epoch(test_dataset, model, criterion, vocab_size, device)
    logging.info("=" * 89)
    logging.info(f"| End of training | test ppl {test_ppl:8.2f}")
    logging.info("=" * 89)


def standard_train(train_loader, val_loader, model, criterion, optimizer, save_path, best_val_ppl, vocab_size, device, args):
    for epoch in range(0, args.max_epoch):
        # Train for one epoch
        train_ppl, train_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, vocab_size, device, args)

        # Evaluate on validation set
        val_ppl = validate_one_epoch(val_loader, model, criterion, vocab_size, device)
        out_string = "Train ppl. {:8.2f} Val ppl {:8.2f} \t".format(train_ppl, val_ppl)
        logging.info(out_string)
        # Record best acc@1 and save checkpoint
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


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, ntokens, device, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    num_batches = len(train_loader)

    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )

    # Switch to train mode
    model.train()
    end = time.time()

    for batch_index, (data, targets) in enumerate(train_loader): 
        # Measure data loading time
        data_time.update(time.time() - end)
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        losses.update(loss.item(), data.numel())

        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_index + 1) % 100 == 0 or (batch_index + 1) == num_batches:
            progress.display(batch_index + 1)

    return math.exp(losses.avg), losses.avg


def validate_one_epoch(val_loader, model, criterion, ntokens, device):
    losses = AverageMeter("Loss", ":.4e")
    # Switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for _, (data, targets) in enumerate(val_loader): 
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output = model(data)
            loss = criterion(output.view(-1, ntokens), targets)
            losses.update(loss.item(), data.numel())
    return math.exp(losses.avg)


if __name__ == "__main__":
    main()

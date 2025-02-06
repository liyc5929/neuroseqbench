import argparse
import json
import logging
import math
import time
from functools import partial

import os
import sys
import toml

# Check and add current working directory
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

import torch
import torch.nn as nn
from datetime import datetime

from src.benchmark.framework.utils.tools import setup_logging, save_checkpoint, AverageMeter, ProgressMeter
from src.benchmark.framework.network.neuron import ASGL_LIF
from src.benchmark.framework.network.trainer.ASGL_surrogate import MutiStepNoisyRateScheduler


class LMSNN(nn.Module):
    def __init__(self,
        nlayers,
        emb_dim,
        hidden_dim,
        vocab_size,
        dropout_words,
        dropout_embedding,
        dropout_forward,
        dropout,
        spiking_neuron=None,
        args=None,
    ):
        super(LMSNN, self).__init__()

        self.args = args
        # Language model specifics
        self.nlayers = nlayers
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Dropout initializations
        self.dropout_words = dropout_words
        self.dropout_embedding = dropout_embedding
        self.dropout_forward = dropout_forward
        self.dropout = dropout

        class LockedDropout(nn.Module):
            def __init__(self):
                super().__init__()
        
            def forward(self, x, dropout=0.5):
                if not self.training or not dropout:
                    return x
                m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
                mask = torch.autograd.Variable(m, requires_grad=False) / (1 - dropout)
                mask = mask.expand_as(x)
                return mask * x

        # input and output layers
        self.locked_dropout = LockedDropout()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.decoder = nn.Linear(emb_dim, vocab_size)
        self.return_state = True

        self.init_weights(initrange=0.1)

        # Tie weights of embedding and decoder
        self.decoder.weight = self.embeddings.weight

        bns = [None for _ in range(nlayers)]
        # RNN model definition
        self.linears = [
            nn.Linear(emb_dim if l == 0 else hidden_dim, emb_dim if l == nlayers - 1 else hidden_dim,)
            for l in range(nlayers)
        ]
        self.snns = [spiking_neuron(neuron_num=emb_dim if l == nlayers - 1 else hidden_dim) for l in range(nlayers)]
        self.linears = nn.ModuleList(self.linears)
        self.snns = nn.ModuleList(self.snns)


    def init_weights(self, initrange=0.1):
        nn.init.uniform_(self.embeddings.weight, -initrange, initrange)
        self.decoder.bias.data.fill_(0)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return [
            (
                weight.new_zeros(batch_size, self.emb_dim if l == self.nlayers - 1 else self.hidden_dim), 
                weight.new_zeros(batch_size, self.emb_dim if l == self.nlayers - 1 else self.hidden_dim)
            )  # v, y
            for l in range(self.nlayers)
        ]

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
            embed.scale_grad_by_freq, embed.sparse,
        )
        return X

    def forward(self, inputs, state): # inputs: [T, B, N]
        embedded = self.embedded_dropout(self.embeddings, inputs, dropout=self.dropout_words if self.training else 0)
        embedded = self.locked_dropout(embedded, dropout=self.dropout_embedding)

        # RNN forward
        new_states = []
        hiddens = embedded
        self.loss = []
        for l, (linear, snn) in enumerate(zip(self.linears, self.snns)):
            hiddens = linear(hiddens)
            hiddens, final_states = snn(hiddens, state[l])
            new_states.append(final_states)
            if l != self.nlayers - 1:
                hiddens = self.locked_dropout(hiddens, dropout=self.dropout_forward)

        # Decoder forward
        hiddens = self.locked_dropout(hiddens, self.dropout)
        decoded = self.decoder(hiddens)
        return decoded, new_states


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
    args.seed              = config.get("seed", 1234)

    # Hyperparameter setting
    args.max_epoch         = config.get("max_epoch", 100)
    args.time_step         = config.get("time_step", 70)
    args.batch_size        = config.get("batch_size", 20)
    args.learning_rate     = config.get("learning_rate", 3)
    args.weight_decay      = config.get("weight_decay", 1.2e-6)
    args.momentum          = config.get("momentum", 0.9)
    args.optimizer         = config.get("optimizer", "SGD")
    args.use_cos_lr        = config.get("use_cos_lr", True)
    args.grad_clip         = config.get("grad_clip", 0.25)

    # Architecture setting
    args.embedding_dim     = config.get("embedding_dim", 400)
    args.hidden_dim        = config.get("hidden_dim", 1100)
    args.dropout_embedding = config.get("dropout_embedding", 0.4)
    args.dropout_words     = config.get("dropout_words", 0.1)
    args.dropout_forward   = config.get("dropout_forward", 0.25)
    args.dropout           = config.get("dropout", 0.4)

    # Neuron setting
    args.neuron_decay      = config.get("neuron_decay", 0.5)
    args.neuron_thresh     = config.get("neuron_thresh", 0.5)
    args.surro_alpha       = config.get("surro_alpha", 1.0)
    args.recurrent         = config.get("recurrent", False)

    return args


def main():
    args = parse_args()

    save_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = save_path + "_" + str(args.seed)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Log settings
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

    from src.benchmark.framework.utils.dataset import PennTreebank
    T = args.time_step
    B = args.batch_size
    train_dataset = PennTreebank(root=os.path.join(args.data_root, "PennTreebank"), subset="train", time_step=T, chunk_num=B, device=device)
    val_dataset   = PennTreebank(root=os.path.join(args.data_root, "PennTreebank"), subset="valid", time_step=T, chunk_num=10, device=device)
    test_dataset  = PennTreebank(root=os.path.join(args.data_root, "PennTreebank"), subset="test",  time_step=T, chunk_num=1, device=device)
    vocab_size = 10000

    exec_mode = "serial"
    spiking_neuron = partial(ASGL_LIF,
        decay=args.neuron_decay,
        threshold=args.neuron_thresh,
        time_step=args.time_step,
        exec_mode=exec_mode,
        a=1.0,
        recurrent=args.recurrent,
    )

    model = LMSNN(
        nlayers=2,
        emb_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=vocab_size,
        dropout_words=args.dropout_words,
        dropout_embedding=args.dropout_embedding,
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
    assert args.use_cos_lr

    criterion = torch.nn.CrossEntropyLoss()
    best_val_ppl = float("inf")

    model = model.to(device)
    try:
        standard_train(train_dataset, val_dataset, model, criterion, optimizer, save_path, best_val_ppl, device, vocab_size, args)
    except KeyboardInterrupt:
        logging.info("-" * 89)
        logging.info("Exiting from training early")
    # Evaluate the best model on the test dataset
    best_model_checkpoint = torch.load(os.path.join(save_path, "model_best.pth.tar"))
    model.load_state_dict(best_model_checkpoint["state_dict"])
    test_ppl = validate_one_epoch(test_dataset, model, criterion, vocab_size, device, 1)
    logging.info("=" * 89)
    logging.info(f"| End of training | test ppl {test_ppl:8.2f}")
    logging.info("=" * 89)


def standard_train(train_loader, val_loader, model, criterion, optimizer, save_path, best_val_ppl, device, vocab_size, args):
    rate_scheduler = MutiStepNoisyRateScheduler(init_p=0.1, reduce_ratio=0.9, num_epoch=args.max_epoch, start_epoch=0)
    rate_scheduler.set_noisy_rate(0.1, model)
    for epoch in range(0, args.max_epoch):

        train_ppl, train_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, vocab_size, args)
        val_ppl = validate_one_epoch(val_loader, model, criterion, vocab_size, device, 10)
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
        rate_scheduler(epoch, model)
    logging.info(f"Best best_val_ppl: {best_val_ppl}")


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is not None:
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, ntokens, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    num_batches = len(train_loader)

    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()
    hidden = model.init_hidden(args.batch_size)

    for batch_index, (data, targets) in enumerate(train_loader): 
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


def validate_one_epoch(val_loader, model, criterion, ntokens, device, eval_batch_size):
    losses = AverageMeter("Loss", ":.4e")

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

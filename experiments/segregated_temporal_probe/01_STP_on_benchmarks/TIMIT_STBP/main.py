import os
import time
import argparse
import toml
import torch
from torch.nn import Module, Sequential, CrossEntropyLoss, Linear, Dropout, BatchNorm1d
from torch.utils.data import DataLoader

from neuroseqbench.network.neuron import LIF
from neuroseqbench.network.structure import MergeDimension, SplitDimension
from neuroseqbench.network.trainer import SurrogateGradient
from neuroseqbench.utils.dataset import TIMIT


class TIMIT_MLP(Module):
    def __init__(self, in_dim=8, hidden=128, out_dim=20, time_step=250, spiking_neuron=None, drop=0.0):
        super().__init__()
        self.__doc__ = f"{in_dim}--{hidden}--dropout{drop}--{hidden}--dropout{drop}--{hidden}--BN1d--{hidden}--BN1d--{hidden}--BN1d--{out_dim}"

        self.time_step = time_step
        self.features = Sequential(
            MergeDimension(),
            Linear(in_dim, hidden),
            Dropout(drop),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            Linear(hidden, hidden),
            Dropout(drop),
            SplitDimension(self.time_step),
            spiking_neuron,
            
            MergeDimension(),
            Linear(hidden, hidden),
            BatchNorm1d(hidden, track_running_stats=False),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            Linear(hidden, hidden),
            BatchNorm1d(hidden, track_running_stats=False),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            Linear(hidden, hidden),
            BatchNorm1d(hidden, track_running_stats=False),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            Linear(hidden, out_dim),
            SplitDimension(self.time_step)
        )

    def forward(self, tx):
        ty = self.features(tx)
        return ty 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="GPU device number")
    parser.add_argument("--config", type=str, default="test.toml", help="toml config file name")
    parser.add_argument("--data_root", type=str, default="/benchmark_data")

    # Get `argparse` arguments
    args = parser.parse_args()

    # Get `toml` arguments
    with open(args.config, "r") as fp:
        config = toml.load(fp)

    # General setting
    args.seed              = config.get("seed")

    # Hyperparameter setting
    args.max_epoch         = config.get("max_epoch", 200)
    args.time_step         = config.get("time_step")
    args.batch_size        = config.get("batch_size", 256)
    args.learning_rate     = config.get("learning_rate", 5e-4)
    args.weight_decay      = config.get("weight_decay", 0.0)

    # Architecture setting
    args.mlp_hidden        = config.get("mlp_hidden", 128) # eg. 128, 512, 1024

    # Neuron setting
    args.neuron_decay      = config.get("neuron_decay", 0.9)
    args.neuron_thresh     = config.get("neuron_thresh", 0.5)
    args.neuron_surrogate  = config.get("neuron_surrogate", "rectangle")
    args.neuron_reset_mode = config.get("neuron_reset_mode", "soft")
    args.neuron_exec_mode  = config.get("neuron_exec_mode", "serial")

    return args


def get_dataloader(task_name: str, data_root: str, time_step: int, batch_size: int, device):
    if time_step in (100,):
        TIMIT_T = time_step
    else: raise NotImplementedError(f"Undefined time steps for task `{task_name}`.")

    train_set = TIMIT(
        root            = os.path.join(data_root, "TIMIT"),
        subset          = "train",
        saved_data_file = f"preprocessed_T{time_step}",
        time_step       = TIMIT_T,
        device          = device
    )
    valid_set = TIMIT(
        root            = os.path.join(data_root, "TIMIT"),
        subset          = "valid",
        saved_data_file = f"preprocessed_T{time_step}",
        time_step       = TIMIT_T,
        device          = device
    )
    test_set = TIMIT(
        root            = os.path.join(data_root, "TIMIT"),
        subset          = "test",
        saved_data_file = f"preprocessed_T{time_step}",
        time_step       = TIMIT_T,
        device          = device
    )

    def TIMIT_collate_fn(item):
        batchs = [batch for batch, _ in item]
        labels = [label for _, label in item]
        return torch.stack(batchs).transpose(0, 1), torch.stack(labels).transpose(0, 1)
    train_loader = DataLoader(
        dataset    = train_set,
        batch_size = batch_size,
        collate_fn = TIMIT_collate_fn,
        shuffle    = True,
        pin_memory = False,
    )
    valid_loader = DataLoader(
        dataset    = valid_set,
        batch_size = batch_size,
        collate_fn = TIMIT_collate_fn,
        shuffle    = True,
        pin_memory = False,
    )
    test_loader = DataLoader(
        dataset    = test_set,
        batch_size = batch_size,
        collate_fn = TIMIT_collate_fn,
        shuffle    = True,
        pin_memory = False,
    )

    return train_loader, valid_loader, test_loader


def main():
    args = parse_args()
    print(args)
    print(f"Pid: {os.getpid()}")

    # Set Random Seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.deterministic = True

    # Set Device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0")
    print(f"Using device {args.device}.")

    # Get dataloader
    train_loader, valid_loader, test_loader = get_dataloader(
        task_name  = "TIMIT",
        data_root  = args.data_root, 
        time_step  = args.time_step,
        batch_size = args.batch_size,
        device     = device,
    )
    
    # Define model
    surro_grad = SurrogateGradient(func_name=args.neuron_surrogate, a=1.0)
    neuron     = LIF(
        decay      = args.neuron_decay, 
        threshold  = args.neuron_thresh, 
        time_step  = args.time_step, 
        surro_grad = surro_grad,
        prop_mode  = "STBP",
        reset_mode = args.neuron_reset_mode,
        exec_mode  = args.neuron_exec_mode
    )
    model  = TIMIT_MLP(in_dim=39, out_dim=61, hidden=args.mlp_hidden, time_step=args.time_step, spiking_neuron=neuron).to(device=device)

    criterion  = CrossEntropyLoss()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print(f"model: {model.__doc__}")
    params = sum([param.nelement() for param in model.parameters()])
    print(f"Parameters: {params / 1e3} K.")
    print(model)
    
    WEIGHT_PATH = "./weights"
    os.makedirs(WEIGHT_PATH, exist_ok=True)
    valid_best_acc = 0.0
    time_start = time.time()
    for epoch in range(args.max_epoch):
        # Training
        running_loss  = 0.0
        train_total   = 0
        train_correct = 0
        model.train()
        for index, (inputs, labels) in enumerate(train_loader, start=1):
            inputs = inputs.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs)

            loss = 0.0
            for y, y_ in zip(logits, labels):
                loss += criterion(y, y_)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            if index % 100 == 0 or (len(train_loader) < 100 and index == len(train_loader)):
                print(
                    f"Epoch: [{epoch}/{args.max_epoch}], "
                    f"step: [{index}/{len(train_loader)}], "
                    f"lr: {optimizer.param_groups[0]['lr']:.6f}, "
                    f"running loss: {running_loss:.6f}, "
                    f"time elapsed: {time.time() - time_start:.6f}",
                    f"pid: {os.getpid()}", 
                )
                running_loss = 0.0
                time_start = time.time()

            _, predicted = torch.max(logits.data, dim=2)
            train_total += predicted.numel()
            train_correct += (predicted == labels).sum().item()
        train_acc = 100 * train_correct / train_total
        print(f"\tTrain Accuracy: {train_acc:.4f} %")

        # Validation
        valid_total   = 0
        valid_correct = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True)
                logits = model(inputs)

                _, predicted = torch.max(logits.data, dim=2)
                valid_total += predicted.numel()
                valid_correct += (predicted == labels).sum().item()
            valid_acc = 100 * valid_correct / valid_total
            if valid_acc > valid_best_acc: 
                valid_best_acc = valid_acc
        print(f"\tValid Accuracy: {valid_acc:.4f} %, Valid Best Accuracy: {valid_best_acc:.4f} %")

        # Testing
        if test_loader is not None:
            test_total   = 0
            test_correct = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device=device, non_blocking=True)
                    labels = labels.to(device=device, non_blocking=True)
                    logits = model(inputs)
    
                    _, predicted = torch.max(logits.data, dim=2)
                    test_total  += predicted.numel()
                    test_correct += (predicted == labels).sum().item()
                test_acc = 100 * test_correct / test_total
            print(f"\tTest  Accuracy: {test_acc:.4f} %")

    print("Finished.")


if __name__ == "__main__":
    main()

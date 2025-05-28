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
from neuroseqbench.utils.dataset import SpikingSpeechCommands


class SSC_MLP(Module):
    def __init__(self, in_dim=8, hidden=128, out_dim=20, time_step=250, spiking_neuron=None, drop=0.0):
        super().__init__()
        self.__doc__ = f"{in_dim}--{hidden}--dropout--{hidden}--dropout--{hidden}--BN1d--{hidden}--BN1d--{hidden}--BN1d--{out_dim}"

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
            BatchNorm1d(hidden),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            Linear(hidden, hidden),
            BatchNorm1d(hidden),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            Linear(hidden, hidden),
            BatchNorm1d(hidden),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            Linear(hidden, out_dim),
            SplitDimension(self.time_step)
        )

    def forward(self, tx):
        ty = self.features(tx)
        return ty.sum(dim=0)


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
    if time_step in (50, 100, 150, 200, 250, 1000):
        SSC_T = time_step
    else: raise NotImplementedError(f"Undefined time steps for task `{task_name}`.")

    def SSC_preprocess(times, units, label):
        """
        Hanle Zheng \emph{et al.} Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics, \textit{nature communications}, 2023.
        """
        import numpy as np
        data_label = torch.tensor(label, dtype=torch.int64)
        max_unit   = 700
        max_time   = 1
        dt         = 1 / SSC_T
        time_step  = int(max_time / dt)
        list_input = []
        for i in range(time_step):
            indexs = np.argwhere(times <= i * dt).flatten()
            vals   = units[indexs]; vals = vals[vals > 0]
            vector = np.zeros(max_unit); vector[max_unit - vals] = 1
            times  = np.delete(times, indexs)
            units  = np.delete(units, indexs)
            list_input.append(vector)
        data_input = torch.tensor(np.array(list_input), dtype=torch.float32)
        return data_input, data_label

    train_set = SpikingSpeechCommands(
        root            = os.path.join(data_root, "SpikingSpeechCommands"),
        subset          = "train",
        preprocess      = SSC_preprocess,
        saved_data_file = f"preprocessed_T{time_step}",
        device          = device
    )
    valid_set = SpikingSpeechCommands(
        root            = os.path.join(data_root, "SpikingSpeechCommands"),
        subset          = "valid",
        preprocess      = SSC_preprocess,
        saved_data_file = f"preprocessed_T{time_step}",
        device          = "cpu"
    )
    test_set = SpikingSpeechCommands(
        root            = os.path.join(data_root, "SpikingSpeechCommands"),
        subset          = "test",
        preprocess      = SSC_preprocess,
        saved_data_file = f"preprocessed_T{time_step}",
        device          = "cpu"
    )

    def SSC_collate_fn(item):
        batchs = [batch for batch, _ in item]
        labels = [label for _, label in item]
        return torch.stack(batchs).transpose(0, 1), torch.tensor(labels)
    train_loader = DataLoader(
        dataset    = train_set,
        batch_size = batch_size,
        collate_fn = SSC_collate_fn,
        shuffle    = True,
        pin_memory = False,
    )
    valid_loader = DataLoader(
        dataset    = valid_set,
        batch_size = batch_size,
        collate_fn = SSC_collate_fn,
        shuffle    = False,
        pin_memory = True,
    )
    test_loader    = DataLoader(
        dataset    = test_set,
        batch_size = batch_size,
        collate_fn = SSC_collate_fn,
        shuffle    = False,
        pin_memory = True,
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
        task_name  = "SpikingSpeechCommands", 
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
        prop_mode  = "SDBP",
        reset_mode = args.neuron_reset_mode,
        exec_mode  = args.neuron_exec_mode
    )

    model = SSC_MLP(in_dim=700, out_dim=35, hidden=args.mlp_hidden, time_step=args.time_step, spiking_neuron=neuron).to(device=device)

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

            loss = criterion(logits, labels)
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

            _, predicted = torch.max(logits.data, dim=1)
            train_total += labels.size(0)
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

                _, predicted = torch.max(logits.data, dim=1)
                valid_total += labels.size(0)
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
    
                    _, predicted = torch.max(logits.data, dim=1)
                    test_total  += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                test_acc = 100 * test_correct / test_total
            print(f"\tTest  Accuracy: {test_acc:.4f} %")

    print("Finished.")


if __name__ == "__main__":
    main()

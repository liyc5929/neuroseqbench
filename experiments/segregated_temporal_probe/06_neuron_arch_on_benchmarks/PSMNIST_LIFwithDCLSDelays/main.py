import os
import time
import toml
import argparse
import torch

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from neuroseqbench.utils.dataset import PSMNIST
from neuroseqbench.network.neuron import LIF
from neuroseqbench.network.structure import DCLS_Delays 
from neuroseqbench.network.trainer import SurrogateGradient


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
    args.max_epoch     = config.get("max_epoch", 200)
    args.time_step     = config.get("time_step", 784)
    args.batch_size    = config.get("batch_size", 256)
    args.learning_rate = config.get("learning_rate", 1e-3)
    args.weight_decay  = config.get("weight_decay", 0.0)

    # Architecture setting
    args.input_size    = config.get("input_size", 2)
    args.hidden_size   = config.get("hidden_size", 110)
    args.output_size   = config.get("output_size", 10)
    args.max_delay     = config.get("max_delay", 100)
    args.output_mode   = config.get("output_mode", "last_time_step")

    # Neuron setting
    args.neuron_decay  = config.get("neuron_decay", 0.9)
    args.neuron_thresh = config.get("neuron_thresh", 0.5)

    return args


def main():
    args = parse_args()
    print(args)
    print(f"Pid: {os.getpid()}")

    # Set Random Seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True

    # Set Device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0")
    print(f"Using device {args.device}.")

    # Define dataset
    if args.time_step != 784:
        raise NotImplementedError(f"Undefined time steps.")
    train_set = PSMNIST(
        root            = args.data_root,
        train           = True,
        download        = True,
        time_step       = args.time_step,
        device          = device
    )
    valid_set = PSMNIST(
        root            = args.data_root,
        train           = False,
        download        = True,
        time_step       = args.time_step,
        device          = device
    )
    def PSMNIST_collate_fn(item):
        batchs = [batch for batch, _ in item]
        labels = [label for _, label in item]
        return torch.stack(batchs).transpose(0, 1).contiguous(), torch.tensor(labels)
    train_loader = DataLoader(
        dataset    = train_set,
        batch_size = args.batch_size,
        collate_fn = PSMNIST_collate_fn,
        shuffle    = True,
        pin_memory = False,
    )
    valid_loader = DataLoader(
        dataset    = valid_set,
        batch_size = args.batch_size,
        collate_fn = PSMNIST_collate_fn,
        shuffle    = False,
        pin_memory = False,
    )
    test_loader = None

    # Define model
    surro_grad = SurrogateGradient(func_name="rectangle", a=1.0)
    neuron     = LIF(
        decay      = args.neuron_decay, 
        threshold  = args.neuron_thresh, 
        time_step  = args.time_step, 
        surro_grad = surro_grad,
        prop_mode  = "STBP",
        reset_mode = "soft",
        exec_mode  = "serial"
    )
    model = DCLS_Delays(
        input_size     = args.input_size,
        hidden_size    = args.hidden_size,
        output_size    = args.output_size,
        time_step      = args.time_step,
        max_delay      = args.max_delay,
        output_mode    = args.output_mode,
        spiking_neuron = neuron,
    ).to(device=device)

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5 * args.learning_rate, total_steps=args.max_epoch)
    print(f"model: {model.__doc__}")
    print(model)
    params = sum([param.nelement() for param in model.parameters()])
    print(f"Parameters: {params / 1e3} K ({params / 1e6} M).")

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
        scheduler.step()

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
            # model.load_state_dict(torch.load(os.path.join(WEIGHT_PATH, weight_name), map_location=device), strict=False, weights_only=True)
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

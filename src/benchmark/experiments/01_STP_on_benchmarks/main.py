import os
import sys
import time
import argparse
import toml
import torch
from torch.nn import (
    Module, Sequential, CrossEntropyLoss,
    Linear, Dropout, Conv1d, Conv2d, AvgPool2d, Flatten, ReLU, BatchNorm1d, 
)
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import DataLoader

# Check and add current working directory
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory) 

from src.benchmark.framework.network.snn_layer import LIF
from src.benchmark.framework.network.structure import MergeDimension, SplitDimension, Permute
from src.benchmark.framework.network.trainer import SurrogateGradient
from src.benchmark.framework.utils.dataset import (
    SMNIST, GoogleSpeechCommands, SpikingHeidelbergDigits, SpikingSpeechCommands, TIMIT
)


class SMNIST_MLP1(Module):
    def __init__(self, in_dim=8, time_step=784, spiking_neuron=None):
        super(SMNIST_MLP1, self).__init__()
        self.__doc__ = f"{in_dim}--64--256--256--10"
        self.in_dim = in_dim
        self.time_step = time_step
        self.features = Sequential(
            MergeDimension(),
            Linear(in_dim, 64),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            Linear(64, 256),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            Linear(256, 256),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            Linear(256, 10),
            SplitDimension(self.time_step)
        )

    def forward(self, tx): # (T, B, 1)
        tx = tx.reshape(tx.shape[1] // self.in_dim, -1, self.in_dim)
        ty = self.features(tx)
        return ty.sum(dim=0)


class GSC_MLP1(Module):
    """
    40--300--dropout--300--dropout--12
    """
    def __init__(self, in_dim=40, time_step=101, spiking_neuron=None, drop=0.3):
        super().__init__()
        self.time_step = time_step
        self.features = Sequential(
            MergeDimension(),
            Linear(in_dim, 300),
            Dropout(drop),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            Linear(300, 300),
            Dropout(drop),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            Linear(300, 12),
            SplitDimension(self.time_step)
        )

    def forward(self, tx):
        ty = self.features(tx)
        return ty.sum(dim=0)


class SampleSCNN1(Module):
    def __init__(self, in_dim=40, hidden=128, out_dim=12, time_step=101, spiking_neuron=None):
        super(SampleSCNN1, self).__init__()
        self.__doc__ = f"{in_dim}--conv1d({hidden})--dropout--conv1d({hidden})--dropout--{out_dim}"

        self.time_step = time_step
        self.features = Sequential(
            Permute(1, 2, 0),
            Conv1d(in_channels=in_dim, out_channels=hidden, kernel_size=5, stride=1, padding=2),
            ReLU(),
            Dropout(0.3),
            Permute(2, 0, 1),
            spiking_neuron,

            Permute(1, 2, 0),
            Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=5, stride=1, padding=2),
            ReLU(),
            Dropout(0.3),
            Permute(2, 0, 1),
            spiking_neuron,

            MergeDimension(),
            Linear(in_features=hidden, out_features=out_dim),
            SplitDimension(self.time_step),
        )
        
    def forward(self, tx: torch.Tensor): # tx: (T, B, C)
        ty = self.features(tx)
        return ty.sum(dim=0)


class SampleSCNN2(Module):
    def __init__(self, in_dim=2, hidden=512, out_dim=11, time_step=50, spiking_neuron=None):
        super(SampleSCNN2, self).__init__()
        self.__doc__ = f"{in_dim}--Conv2d(32)--AvgPool2d--Conv2d(32)--AvgPool2d--Conv2d(16)--AvgPool2d--Conv2d(8)--{hidden}--128--{out_dim}"

        self.time_step = time_step
        self.features = Sequential(
            MergeDimension(),
            Conv2d(in_channels=2, out_channels=32, stride=1, padding=0, kernel_size=5),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            AvgPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, stride=1, padding=0, kernel_size=5),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            AvgPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=16, stride=1, padding=0, kernel_size=5),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            AvgPool2d(kernel_size=2),
            Conv2d(in_channels=16, out_channels=8, stride=1, padding=0, kernel_size=5),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            Flatten(start_dim=1, end_dim=-1), # for shape (B, C, H, W)
            Linear(hidden, 128),
            SplitDimension(self.time_step),
            spiking_neuron,

            MergeDimension(),
            Linear(128, out_dim),
            SplitDimension(self.time_step),
        )
        
    def forward(self, tx: torch.Tensor): # tx: (T, B, C, H, W)
        ty = self.features(tx)
        return ty.sum(dim=0)


class SHD_SSC_MLP1(Module):
    def __init__(self, in_dim=8, hidden=128, out_dim=20, time_step=250, spiking_neuron=None, drop=0.0):
        super().__init__()
        self.__doc__ = f"{in_dim}--{hidden}--dropout--{hidden}--dropout--{out_dim}"

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
            Linear(hidden, out_dim),
            SplitDimension(self.time_step)
        )

    def forward(self, tx):
        ty = self.features(tx)
        return ty.sum(dim=0)


class SHD_SSC_MLP2(Module):
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

class TIMIT_MLP2(Module):
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
    args.seed = config.get("seed", 42)
    args.task = config.get("task") # SMNIST, GSC, SHD, SSC, DvsGesture or TIMIT
    args.max_epoch = config.get("max_epoch", 200)
    args.time_step = config.get("time_step")
    args.batch_size = config.get("batch_size", 256)
    args.learning_rate = config.get("learning_rate", 5e-4)
    args.weight_decay = config.get("weight_decay", 0)
    args.arch = config.get("arch", "MLP1") # MLP1, MLP2, ..., CNN1, CNN2, ...
    args.mlp_hidden = config.get("mlp_hidden", 128) # eg. 128, 512, 1024
    args.neuron_decay = config.get("neuron_decay", 0.9)
    args.neuron_thresh = config.get("neuron_thresh", 0.5)
    args.neuron_surrogate = config.get("neuron_surrogate", "rectangle")
    args.neuron_surro_hyperparam = config.get("neuron_surro_hyperparam", 1.0) # Hyperparameter `a` for rectangle function
    args.neuron_reset_mode = config.get("neuron_reset_mode", "soft")
    args.neuron_prop_mode = config.get("neuron_prop_mode", "STBP") # "STBP, SDBP, noTD"

    return args


if __name__ == "__main__":
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
    if args.task == "SMNIST":
        if args.time_step != 784:
            raise NotImplementedError(f"Undefined time steps for task `{args.task}`.")
        train_set = SMNIST(
            root            = args.data_root,
            train           = True,
            download        = True,
            saved_data_file = f"preprocessed_T{args.time_step}",
            time_step       = args.time_step,
            device          = device
        )
        valid_set = SMNIST(
            root            = args.data_root,
            train           = False,
            download        = True,
            saved_data_file = f"preprocessed_T{args.time_step}",
            time_step       = args.time_step,
            device          = device
        )
        def SMNIST_collate_fn(item):
            batchs = [batch for batch, _ in item]
            labels = [label for _, label in item]
            return torch.stack(batchs).transpose(0, 1), torch.tensor(labels)
        train_loader = DataLoader(
            dataset    = train_set,
            batch_size = args.batch_size,
            collate_fn = SMNIST_collate_fn,
            shuffle    = True,
            pin_memory = False,
        )
        valid_loader = DataLoader(
            dataset    = valid_set,
            batch_size = args.batch_size,
            collate_fn = SMNIST_collate_fn,
            shuffle    = False,
            pin_memory = False,
        )
    elif args.task == "GSC":
        # Preprocess argument selection
        if args.time_step == 51:
            mel_spectrogram_kwargs = {
                "sample_rate": 16000,
                "n_fft":       480,
                "hop_length":  320,
                "n_mels":      40
            }
        elif args.time_step == 101: 
            mel_spectrogram_kwargs = {
                "sample_rate": 16000,
                "n_fft":       480,
                "hop_length":  160,
                "n_mels":      40
            }
        elif args.time_step == 150:
            mel_spectrogram_kwargs = {
                "sample_rate": 16000,
                "n_fft":       480,
                "hop_length":  107,
                "n_mels":      40
            }
        else:
            raise NotImplementedError(f"Undefined time steps for task `{args.task}`.")
        GSC_T = args.time_step

        GSCmdV2Categs = {
            "unknown": 0,
            "silence": 1,
            "_unknown_": 0,
            "_silence_": 1,
            "_background_noise_": 1,
            "yes": 2,
            "no": 3,
            "up": 4,
            "down": 5,
            "left": 6,
            "right": 7,
            "on": 8,
            "off": 9,
            "stop": 10,
            "go": 11
        }
        def GSC_preprocess(waveform, _, label, *__):
            data_label      = torch.tensor(GSCmdV2Categs.get(label, 0), dtype=torch.int64)
            mel_spectrogram = MelSpectrogram(**mel_spectrogram_kwargs)(waveform).squeeze(0).transpose(0, 1)
            mean            = mel_spectrogram.mean()
            std             = mel_spectrogram.std()
            data_input      = (mel_spectrogram - mean) / std
            data_input      = torch.nn.functional.pad(data_input, (0, 0, 0, GSC_T - data_input.shape[0]))
            placeholder     = torch.tensor([0], dtype=torch.int64)
            return (
                data_input, 
                placeholder, 
                data_label, 
                placeholder, 
                placeholder 
            )
        train_set = GoogleSpeechCommands(
            root            = args.data_root,
            url             = "speech_commands_v0.02",
            download        = True,
            subset          = "training",
            preprocess      = GSC_preprocess,
            saved_data_file = f"preprocessed_T{args.time_step}",
            device          = device
        )
        valid_set = GoogleSpeechCommands(
            root            = args.data_root,
            url             = "speech_commands_v0.02",
            download        = True,
            subset          = "validation",
            preprocess      = GSC_preprocess,
            saved_data_file = f"preprocessed_T{args.time_step}",
            device          = device
        )

        def GSC_collate_fn(item): # [waveform, sample_rate, label, speaker_id, utterance_number]
            batchs = [batch for batch, *_ in item]
            labels = [label for _, _, label, *_ in item]
            return torch.stack(batchs).transpose(0, 1), torch.tensor(labels)
        train_loader = DataLoader(
            dataset    = train_set,
            batch_size = args.batch_size,
            collate_fn = GSC_collate_fn,
            shuffle    = True,
            pin_memory = False,
        )
        valid_loader = DataLoader(
            dataset    = valid_set,
            batch_size = args.batch_size,
            collate_fn = GSC_collate_fn,
            shuffle    = False,
            pin_memory = False,
        )
    elif args.task == "GSCv1":
        # Preprocess argument selection
        if args.time_step == 51:
            mel_spectrogram_kwargs = {
                "sample_rate": 16000,
                "n_fft":       480,
                "hop_length":  320,
                "n_mels":      40
            }
        elif args.time_step == 101: 
            mel_spectrogram_kwargs = {
                "sample_rate": 16000,
                "n_fft":       480,
                "hop_length":  160,
                "n_mels":      40
            }
        elif args.time_step == 150:
            mel_spectrogram_kwargs = {
                "sample_rate": 16000,
                "n_fft":       480,
                "hop_length":  107,
                "n_mels":      40
            }
        else:
            raise NotImplementedError(f"Undefined time steps for task `{args.task}`.")
        GSCv1_T = args.time_step

        GSCmdV1Categs = {
            "unknown": 0, "_unknown_": 0, 
            "silence": 1, "_silence_": 1, "_background_noise_": 1,
            "yes": 2, "no": 3, "up": 4, "down": 5, "left": 6, "right": 7, 
            "on": 8, "off": 9, "stop": 10, "go": 11,
            "bed": 12, "bird": 13, "cat": 14, "dog": 15,
            "happy": 16, "house": 17, "marvin": 18, "sheila": 19, 
            "zero": 20, "one": 21, "two": 22, "three": 23, "four": 24, 
            "five": 25, "six": 26, "seven": 27, "eight": 28, "nine": 29,
            "tree": 30, "wow": 31
        }
        def GSC_preprocess(waveform, _, label, *__):
            data_label      = torch.tensor(GSCmdV1Categs.get(label, 0), dtype=torch.int64)
            mel_spectrogram = MelSpectrogram(**mel_spectrogram_kwargs)(waveform).squeeze(0).transpose(0, 1)
            mean            = mel_spectrogram.mean()
            std             = mel_spectrogram.std()
            data_input      = (mel_spectrogram - mean) / std
            data_input      = torch.nn.functional.pad(data_input, (0, 0, 0, GSCv1_T - data_input.shape[0]))
            placeholder     = torch.tensor([0], dtype=torch.int64)
            return (
                data_input, 
                placeholder, 
                data_label, 
                placeholder, 
                placeholder 
            )
        train_set = GoogleSpeechCommands(
            root            = args.data_root,
            url             = "speech_commands_v0.01",
            download        = True,
            subset          = "training",
            preprocess      = GSC_preprocess,
            saved_data_file = f"v1_preprocessed_T{args.time_step}",
            device          = device
        )
        valid_set = GoogleSpeechCommands(
            root            = args.data_root,
            url             = "speech_commands_v0.01",
            download        = True,
            subset          = "validation",
            preprocess      = GSC_preprocess,
            saved_data_file = f"v1_preprocessed_T{args.time_step}",
            device          = device
        )

        def GSC_collate_fn(item): # [waveform, sample_rate, label, speaker_id, utterance_number]
            batchs = [batch for batch, *_ in item]
            labels = [label for _, _, label, *_ in item]
            return torch.stack(batchs).transpose(0, 1), torch.tensor(labels)
        train_loader = DataLoader(
            dataset    = train_set,
            batch_size = args.batch_size,
            collate_fn = GSC_collate_fn,
            shuffle    = True,
            pin_memory = False,
        )
        valid_loader = DataLoader(
            dataset    = valid_set,
            batch_size = args.batch_size,
            collate_fn = GSC_collate_fn,
            shuffle    = False,
            pin_memory = False,
        )
    elif args.task == "GSCv2":
        # Preprocess argument selection
        if args.time_step == 51:
            mel_spectrogram_kwargs = {
                "sample_rate": 16000,
                "n_fft":       480,
                "hop_length":  320,
                "n_mels":      40
            }
        elif args.time_step == 101: 
            mel_spectrogram_kwargs = {
                "sample_rate": 16000,
                "n_fft":       480,
                "hop_length":  160,
                "n_mels":      40
            }
        elif args.time_step == 150:
            mel_spectrogram_kwargs = {
                "sample_rate": 16000,
                "n_fft":       480,
                "hop_length":  107,
                "n_mels":      40
            }
        else:
            raise NotImplementedError(f"Undefined time steps for task `{args.task}`.")
        GSCv2_T = args.time_step

        full_GSCmdV2Categs = {
            "unknown": 0, "_unknown_": 0, 
            "silence": 1, "_silence_": 1, "_background_noise_": 1,
            "yes": 2, "no": 3, "up": 4, "down": 5, "left": 6, "right": 7, 
            "on": 8, "off": 9, "stop": 10, "go": 11,
            "forward": 12, "backward": 13, "follow": 14, "learn": 15, 
            "bed": 16, "bird": 17, "cat": 18, "dog": 19,
            "zero": 20, "one": 21, "two": 22, "three": 23, "four": 24, 
            "five": 25, "six": 26, "seven": 27, "eight": 28, "nine": 29,
            "happy": 30, "house": 31, "marvin": 32, "sheila": 33, "tree": 34, "wow": 35,
            "visual": 36
        }
        def GSC_preprocess(waveform, _, label, *__):
            data_label      = torch.tensor(full_GSCmdV2Categs.get(label, 0), dtype=torch.int64)
            mel_spectrogram = MelSpectrogram(**mel_spectrogram_kwargs)(waveform).squeeze(0).transpose(0, 1)
            mean            = mel_spectrogram.mean()
            std             = mel_spectrogram.std()
            data_input      = (mel_spectrogram - mean) / std
            data_input      = torch.nn.functional.pad(data_input, (0, 0, 0, GSCv2_T - data_input.shape[0]))
            placeholder     = torch.tensor([0], dtype=torch.int64)
            return (
                data_input, 
                placeholder, 
                data_label, 
                placeholder, 
                placeholder 
            )
        train_set = GoogleSpeechCommands(
            root            = args.data_root,
            url             = "speech_commands_v0.02",
            download        = True,
            subset          = "training",
            preprocess      = GSC_preprocess,
            saved_data_file = f"v2_preprocessed_T{args.time_step}",
            device          = device
        )
        valid_set = GoogleSpeechCommands(
            root            = args.data_root,
            url             = "speech_commands_v0.02",
            download        = True,
            subset          = "validation",
            preprocess      = GSC_preprocess,
            saved_data_file = f"v2_preprocessed_T{args.time_step}",
            device          = device
        )

        def GSC_collate_fn(item): # [waveform, sample_rate, label, speaker_id, utterance_number]
            batchs = [batch for batch, *_ in item]
            labels = [label for _, _, label, *_ in item]
            return torch.stack(batchs).transpose(0, 1), torch.tensor(labels)
        train_loader = DataLoader(
            dataset    = train_set,
            batch_size = args.batch_size,
            collate_fn = GSC_collate_fn,
            shuffle    = True,
            pin_memory = False,
        )
        valid_loader = DataLoader(
            dataset    = valid_set,
            batch_size = args.batch_size,
            collate_fn = GSC_collate_fn,
            shuffle    = False,
            pin_memory = False,
        )
    elif args.task == "SHD":
        if args.time_step in (30, 50, 100, 150, 200, 250, 1000):
            SHD_T = args.time_step
        else: raise NotImplementedError(f"Undefined time steps for task `{args.task}`.")

        def old_SHD_preprocess(times, units, label): # Rate coding based preprocess
            data_label        = torch.tensor(label, dtype=torch.int64)
            max_unit          = 700
            max_time          = 1.4
            time_step         = SHD_T 
            data_input        = torch.zeros((time_step, max_unit))
            time_step_bins    = torch.linspace(0, max_time, time_step)
            time_step_indexes = torch.searchsorted(time_step_bins, torch.tensor(times, dtype=torch.float32)) - 1
            for time_step_index, unit_index in zip(time_step_indexes, units):
                data_input[time_step_index, unit_index] += 1
            data_input -= data_input.mean(axis=(0, 1), keepdims=True)
            data_input /= data_input.std(axis=(0, 1), keepdims=True) 
            return data_input, data_label

        def SHD_preprocess(times, units, label):
            """
            Hanle Zheng \emph{et al.} Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics, \textit{nature communications}, 2023.
            """
            import numpy as np
            data_label = torch.tensor(label, dtype=torch.int64)
            max_unit   = 700
            max_time   = 1
            dt         = 1 / SHD_T  # standard: 1e-3 or 4e-3
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

        train_set = SpikingHeidelbergDigits(
            root            = f"{args.data_root}/SpikingHeidelbergDigits",
            subset          = "train",
            preprocess      = SHD_preprocess,
            saved_data_file = f"preprocessed_T{args.time_step}",
            device          = device
        )
        valid_set = SpikingHeidelbergDigits(
            root            = f"{args.data_root}/SpikingHeidelbergDigits",
            subset          = "test",
            preprocess      = SHD_preprocess,
            saved_data_file = f"preprocessed_T{args.time_step}",
            device          = device
        )

        def SHD_collate_fn(item):
            batchs = [batch for batch, _ in item]
            labels = [label for _, label in item]
            return torch.stack(batchs).transpose(0, 1), torch.tensor(labels)
        train_loader = DataLoader(
            dataset    = train_set,
            batch_size = args.batch_size,
            collate_fn = SHD_collate_fn,
            shuffle    = True,
            pin_memory = False,
        )
        valid_loader = DataLoader(
            dataset    = valid_set,
            batch_size = args.batch_size,
            collate_fn = SHD_collate_fn,
            shuffle    = False,
            pin_memory = False,
        )
        test_loader = None
    elif args.task == "SSC":
        if args.time_step in (50, 100, 150, 200, 250, 1000):
            SSC_T = args.time_step
        else: raise NotImplementedError(f"Undefined time steps for task `{args.task}`.")

        def old_SSC_preprocess(times, units, label): # Rate coding based preprocess
            data_label        = torch.tensor(label, dtype=torch.int64)
            max_unit          = 700
            max_time          = 1.4
            time_step         = SSC_T 
            data_input        = torch.zeros((time_step, max_unit))
            time_step_bins    = torch.linspace(0, max_time, time_step)
            time_step_indexes = torch.searchsorted(time_step_bins, torch.tensor(times, dtype=torch.float32)) - 1
            for time_step_index, unit_index in zip(time_step_indexes, units):
                data_input[time_step_index, unit_index] += 1
            data_input -= data_input.mean(axis=(0, 1), keepdims=True)
            data_input /= data_input.std(axis=(0, 1), keepdims=True) 
            return data_input, data_label

        def SSC_preprocess(times, units, label):
            """
            Hanle Zheng \emph{et al.} Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics, \textit{nature communications}, 2023.
            """
            import numpy as np
            data_label = torch.tensor(label, dtype=torch.int64)
            max_unit   = 700
            max_time   = 1
            dt         = 1 / SSC_T  # standard: 1e-3 or 4e-3
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
            root            = f"{args.data_root}/SpikingSpeechCommands",
            subset          = "train",
            preprocess      = SSC_preprocess,
            saved_data_file = f"preprocessed_T{args.time_step}",
            device          = device
        )
        valid_set = SpikingSpeechCommands(
            root            = f"{args.data_root}/SpikingSpeechCommands",
            subset          = "valid",
            preprocess      = SSC_preprocess,
            saved_data_file = f"preprocessed_T{args.time_step}",
            device          = "cpu"
        )
        test_set = SpikingSpeechCommands(
            root            = f"{args.data_root}/SpikingSpeechCommands",
            subset          = "test",
            preprocess      = SSC_preprocess,
            saved_data_file = f"preprocessed_T{args.time_step}",
            device          = "cpu"
        )

        def SSC_collate_fn(item):
            batchs = [batch for batch, _ in item]
            labels = [label for _, label in item]
            return torch.stack(batchs).transpose(0, 1), torch.tensor(labels)
        train_loader = DataLoader(
            dataset    = train_set,
            batch_size = args.batch_size,
            collate_fn = SSC_collate_fn,
            shuffle    = True,
            pin_memory = False,
        )
        valid_loader = DataLoader(
            dataset    = valid_set,
            batch_size = args.batch_size,
            collate_fn = SSC_collate_fn,
            shuffle    = False,
            pin_memory = True,
        )
        test_loader    = DataLoader(
            dataset    = test_set,
            batch_size = args.batch_size,
            collate_fn = SSC_collate_fn,
            shuffle    = False,
            pin_memory = True,
        )
    elif args.task == "DvsGesture":
        if args.time_step in (50, 100, 150, 300):
            DvsGesture_train_device = device
            DvsGesture_valid_device = device
            DvsGesture_train_pin    = False
            DvsGesture_valid_pin    = False
        elif args.time_step in (500,):
            DvsGesture_train_device = device
            DvsGesture_valid_device = "cpu"
            DvsGesture_train_pin    = False
            DvsGesture_valid_pin    = True
        else: 
            raise NotImplementedError(f"Undefined time steps for task `{args.task}`.")
        DvsGesture_T = args.time_step

        from src.benchmark.framework.utils.dataset.dvs_gesture import DVS128Gesture
        train_set = DVS128Gesture(
            f"{args.data_root}/DvsGesture", data_type="frame", train=True, frames_number=DvsGesture_T, split_by="number",
            device=DvsGesture_train_device
        )
        valid_set = DVS128Gesture(
            f"{args.data_root}/DvsGesture", data_type="frame", train=False, frames_number=DvsGesture_T, split_by="number",
            device=DvsGesture_valid_device
        )
        def DvsGesture_collate_fn(item):
            batchs = [batch for batch, _ in item]
            labels = [label for _, label in item]
            return torch.stack(batchs).transpose(0, 1), torch.tensor(labels)
        train_loader = DataLoader(
            dataset    = train_set,
            batch_size = args.batch_size,
            collate_fn = DvsGesture_collate_fn,
            shuffle    = True,
            pin_memory = DvsGesture_train_pin,
        )
        valid_loader = DataLoader(
            dataset    = valid_set,
            batch_size = args.batch_size,
            collate_fn = DvsGesture_collate_fn,
            shuffle    = False,
            pin_memory = DvsGesture_valid_pin,
        )
    elif args.task == "TIMIT":
        if args.time_step in (100,):
            TIMIT_T = args.time_step
        else: raise NotImplementedError(f"Undefined time steps for task `{args.task}`.")

        train_set = TIMIT(
            root            = f"{args.data_root}/TIMIT",
            subset          = "train",
            saved_data_file = f"preprocessed_T{args.time_step}",
            time_step       = TIMIT_T,
            device          = device
        )
        valid_set = TIMIT(
            root            = f"{args.data_root}/TIMIT",
            subset          = "valid",
            saved_data_file = f"preprocessed_T{args.time_step}",
            time_step       = TIMIT_T,
            device          = device
        )
        test_set = TIMIT(
            root            = f"{args.data_root}/TIMIT",
            subset          = "test",
            saved_data_file = f"preprocessed_T{args.time_step}",
            time_step       = TIMIT_T,
            device          = device
        )

        def TIMIT_collate_fn(item):
            batchs = [batch for batch, _ in item]
            labels = [label for _, label in item]
            return torch.stack(batchs).transpose(0, 1), torch.stack(labels).transpose(0, 1)
        train_loader = DataLoader(
            dataset    = train_set,
            batch_size = args.batch_size,
            collate_fn = TIMIT_collate_fn,
            shuffle    = True,
            pin_memory = False,
        )
        valid_loader = DataLoader(
            dataset    = valid_set,
            batch_size = args.batch_size,
            collate_fn = TIMIT_collate_fn,
            shuffle    = True,
            pin_memory = False,
        )
        test_loader = DataLoader(
            dataset    = test_set,
            batch_size = args.batch_size,
            collate_fn = TIMIT_collate_fn,
            shuffle    = True,
            pin_memory = False,
        )
    else:
        raise NotImplementedError(f"Undefined task `{args.task}`.")

    # Define model
    surro_grad = SurrogateGradient(func_name=args.neuron_surrogate, a=1.0)
    neuron     = LIF(
        decay      = args.neuron_decay, 
        threshold  = args.neuron_thresh, 
        time_step  = args.time_step, 
        surro_grad = surro_grad,
        prop_mode  = args.neuron_prop_mode,
        reset_mode = args.neuron_reset_mode,
        exec_mode  = "serial"
    )
    if args.arch == "MLP1":
        if args.task == "SMNIST":
            model  = SMNIST_MLP1(in_dim=1, time_step=args.time_step, spiking_neuron=neuron).to(device=device)
        elif args.task == "GSC":
            model  = GSC_MLP1(in_dim=40, time_step=args.time_step, spiking_neuron=neuron).to(device=device)
        elif args.task == "SHD":
            model  = SHD_SSC_MLP1(in_dim=700, out_dim=20, hidden=128, time_step=args.time_step, spiking_neuron=neuron).to(device=device)
        elif args.task == "SSC":
            model  = SHD_SSC_MLP1(in_dim=700, out_dim=35, hidden=128, time_step=args.time_step, spiking_neuron=neuron).to(device=device)
    if args.arch == "MLP2":
        if args.task == "SHD":
            model  = SHD_SSC_MLP2(in_dim=700, out_dim=20, hidden=args.mlp_hidden, time_step=args.time_step, spiking_neuron=neuron).to(device=device)
        elif args.task == "SSC":
            model  = SHD_SSC_MLP2(in_dim=700, out_dim=35, hidden=args.mlp_hidden, time_step=args.time_step, spiking_neuron=neuron).to(device=device)
        elif args.task == "TIMIT":
            model  = TIMIT_MLP2(in_dim=39, out_dim=61, hidden=args.mlp_hidden, time_step=args.time_step, spiking_neuron=neuron).to(device=device)
    elif args.arch == "CNN1":
        if args.task == "GSC":
            in_dim = 40; hidden = 128; out_dim = 12
        if args.task == "GSCv1":
            in_dim = 40; hidden = 128; out_dim = 32
        if args.task == "GSCv2":
            in_dim = 40; hidden = 128; out_dim = 37
        elif args.task == "SHD":
            in_dim = 700; hidden = 32; out_dim = 20
        elif args.task == "SSC":
            in_dim = 700; hidden = 32; out_dim = 35
        model = SampleSCNN1(in_dim=in_dim, hidden=hidden, out_dim=out_dim, time_step=args.time_step, spiking_neuron=neuron).to(device=device)
    elif args.arch == "CNN2":
        if args.task == "DvsGesture":
            model = SampleSCNN2(in_dim=2, hidden=512, out_dim=11, time_step=args.time_step, spiking_neuron=neuron).to(device=device)
    else:
        raise NotImplementedError(f"Undefined architecture `{args.arch}`.")

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

            if args.task == "TIMIT":
                loss = 0.0
                for y, y_ in zip(logits, labels):
                    loss += criterion(y, y_)
            else: 
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

            if args.task == "TIMIT":
                _, predicted = torch.max(logits.data, dim=2)
                train_total += predicted.numel()
            else:
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

                if args.task == "TIMIT":
                    _, predicted = torch.max(logits.data, dim=2)
                    valid_total += predicted.numel()
                else:
                    _, predicted = torch.max(logits.data, dim=1)
                    valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
            valid_acc = 100 * valid_correct / valid_total
            if valid_acc > valid_best_acc: 
                valid_best_acc = valid_acc
                # # Save model weights
                # weight_name = f"{args.task}_{args.arch}_epoch{epoch}_{best_acc:.4f}.pth"
                # torch.save(model.state_dict(), os.path.join(WEIGHT_PATH, weight_name))
        print(f"\tValid Accuracy: {valid_acc:.4f} %, Valid Best Accuracy: {valid_best_acc:.4f} %")

        # Testing
        if test_loader is not None:
            # model.load_state_dict(torch.load(os.path.join(WEIGHT_PATH, weight_name), map_location=device), strict=False)
            test_total   = 0
            test_correct = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device=device, non_blocking=True)
                    labels = labels.to(device=device, non_blocking=True)
                    logits = model(inputs)
    
                    if args.task == "TIMIT":
                        _, predicted = torch.max(logits.data, dim=2)
                        test_total  += predicted.numel()
                    else:
                        _, predicted = torch.max(logits.data, dim=1)
                        test_total  += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                test_acc = 100 * test_correct / test_total
            print(f"\tTest  Accuracy: {test_acc:.4f} %")

    print("Finished.")

import os.path

import torch
import torchvision
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import transforms
from torchaudio.transforms import MelSpectrogram

from ..dataset.dvs_gesture import DVS128Gesture, DVS128Gesture_Cached
from ..dataset.augmentation import RandomMixup, RandomCutmix, ClassificationPresetTrain
from ..dataset.imdb import create_lra_imdb_classification_dataset, get_imdb_word_dataloader, zero_words_in_embedding
from ..dataset.cifardvs.cifar10_dvs import CIFAR10DVS
from ..dataset.google_speech_commands import GoogleSpeechCommands
from ..dataset.spiking_heidelberg_digits import SpikingHeidelbergDigits
from ..dataset.spiking_speech_commands import SpikingSpeechCommands
from ..dataset.adding_problem import AddingProblem
from ..dataset.newsgroups import create_20newsgroup_dataset
from ..dataset.dvs_slr import create_dvsslr_dataset
from ..dataset.dvslip.dvs_lip import DVS_Lip, DVSLipDataset
from ..dataset.cifardvs.augmentation import ToPILImage, Resize, ToTensor, Roll  # , Cutout
import numpy as np
from torch.utils.data import Dataset
from functools import partial

from ..dataset.electrocardiogram import Electrocardiogram
from ..dataset import integrate_events_segment_to_frame





def build_dataset(dataset, data_path, seq_length=100, data_cache=False, min_length=0, num_bins=1):
    if dataset == 'seqcifar10':
        mixup_transforms = []
        mixup_transforms.append(RandomMixup(num_classes=10, p=1.0, alpha=0.2))
        mixup_transforms.append(RandomCutmix(num_classes=10, p=1.0, alpha=1.))
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*default_collate(batch))

        train_transform = ClassificationPresetTrain(mean=(0.4914, 0.4822, 0.4465),
                                                    std=(0.2023, 0.1994, 0.2010),
                                                    interpolation=InterpolationMode('bilinear'),
                                                    auto_augment_policy='ta_wide',
                                                    random_erase_prob=0.1)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=True,
            transform=train_transform,
            download=False)

        test_dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=False,
            transform=test_transform,
            download=False)
        n_classes = 10
        input_channels = 3

    elif dataset == 'dvsgesture':
        if data_cache:
            train_dataset = DVS128Gesture_Cached(data_path, data_type='frame', train=True, frames_number=seq_length,
                          split_by='number')
            test_dataset = DVS128Gesture_Cached(data_path, data_type='frame', train=False, frames_number=seq_length,
                                      split_by='number')
            n_classes = 11
            input_channels = 2
            def DvsGesture_collate_fn(item):
                batchs = [batch for batch, _ in item]
                labels = [label for _, label in item]
                return torch.stack(batchs).transpose(0, 1), torch.tensor(labels)
            collate_fn = DvsGesture_collate_fn
        else:
            train_dataset = DVS128Gesture(data_path, data_type='frame', train=True, frames_number=seq_length,
                                                 split_by='number')
            test_dataset = DVS128Gesture(data_path, data_type='frame', train=False, frames_number=seq_length,
                                                split_by='number')
            n_classes = 11
            input_channels = 2
            collate_fn = None
    elif dataset == 'psmnist':
        permutation = np.random.permutation(int(784))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(1, 784).t()),
            transforms.Lambda(lambda x: x[permutation])
        ])
        transform_train = transform_test = transform

        train_dataset = torchvision.datasets.MNIST(
            root=data_path, train=True, download=False, transform=transform_train)
        test_dataset = torchvision.datasets.MNIST(
            root=data_path, train=False, download=False, transform=transform_test)
        n_classes = 10
        input_channels = 1
        collate_fn = None
    elif dataset == 'smnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(1, 784).t()),
        ])
        transform_train = transform_test = transform

        train_dataset = torchvision.datasets.MNIST(
            root=data_path, train=True, download=False, transform=transform_train)
        test_dataset = torchvision.datasets.MNIST(
            root=data_path, train=False, download=False, transform=transform_test)
        n_classes = 10
        input_channels = 1
        collate_fn = None
    elif dataset == 'add':
        # X_train, Y_train = data_generator(50000, seq_length)  # [N,1,Seq_len],[N,2,Seq_len]
        # X_test, Y_test = data_generator(1000, seq_length)
        # train_dataset = AddingProblemDataset(X_train, Y_train)
        # test_dataset = AddingProblemDataset(X_test, Y_test)
        # n_classes = None
        # input_channels = 1
        # collate_fn = None
        n_classes = None
        input_channels = 1
        collate_fn = None
        train_dataset = AddingProblem(
            root=data_path,
            subset="train",
            num_data=50000,
            seq_length=100,
            is_binary=False,
            preprocess=None
        )
        test_dataset = AddingProblem(
            root=data_path,
            subset="test",
            num_data=1000,
            seq_length=100,
            is_binary=False,
            preprocess=None
        )
    elif dataset == 'binadd':
        # X_train, Y_train = data_generator(50000, seq_length, binary=True)  # [N,1,Seq_len],[N,2,Seq_len]
        # X_test, Y_test = data_generator(1000, seq_length, binary=True)
        # train_dataset = AddingProblemDataset(X_train, Y_train)
        # test_dataset = AddingProblemDataset(X_test, Y_test)
        # n_classes = 10
        # input_channels = 1
        # collate_fn = None
        data_path = '/datasets/BinaryAddingProblem'
        train_dataset = AddingProblem(
            root=data_path,
            subset="train",
            num_data=50000,
            seq_length=seq_length,
            is_binary=True,
            preprocess=None
        )
        test_dataset = AddingProblem(
            root=data_path,
            subset="test",
            num_data=2000,
            seq_length=seq_length,
            is_binary=True,
            preprocess=None
        )
        n_classes = 10
        input_channels = 1
        collate_fn = None
    elif dataset == 'imdb':
        train_dataset, test_dataset, n_classes, VOCAB_SIZE, collate_fn = create_lra_imdb_classification_dataset(data_dir=data_path, seq_length=seq_length)
        input_channels = VOCAB_SIZE
    elif dataset == 'cifar10dvs':
        # transform_train = transforms.Compose([
        #     ToPILImage(),
        #     # Resize(48),
        #     ToTensor(),
        # ])
        # transform_test = transforms.Compose([
        #     ToPILImage(),
        #     # Resize(48),
        #     ToTensor(),
        # ])
        train_dataset = CIFAR10DVS(data_path, train=True, use_frame=True, frames_num=seq_length,
                                   split_by='number',
                                   normalization=None)
        test_dataset = CIFAR10DVS(data_path, train=False, use_frame=True, frames_num=seq_length,
                                  split_by='number',
                                  normalization=None)

        n_classes = 10
        input_channels = 2
        collate_fn = None

    elif dataset == 'gsc':
        if seq_length == 51:
            mel_spectrogram_kwargs = {
                "sample_rate": 16000,
                "n_fft":       480,
                "hop_length":  320,
                "n_mels":      40
            }
        elif seq_length == 101:
            mel_spectrogram_kwargs = {
                "sample_rate": 16000,
                "n_fft":       480,
                "hop_length":  160,
                "n_mels":      40
            }
        elif seq_length == 150:
            mel_spectrogram_kwargs = {
                "sample_rate": 16000,
                "n_fft":       480,
                "hop_length":  107,
                "n_mels":      40
            }
        else:
            raise NotImplementedError


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
            data_input      = torch.nn.functional.pad(data_input, (0, 0, 0, seq_length - data_input.shape[0]))
            placeholder     = torch.tensor([0], dtype=torch.int64)
            return (
                data_input,
                placeholder,
                data_label,
                placeholder,
                placeholder
            )
        train_dataset = GoogleSpeechCommands(
            root            = data_path,
            url             = "speech_commands_v0.02",
            download        = True,
            subset          = "training",
            preprocess      = GSC_preprocess,
            saved_data_file = f"preprocessed_T{seq_length}",
        )
        test_dataset = GoogleSpeechCommands(
            root            = data_path,
            url             = "speech_commands_v0.02",
            download        = True,
            subset          = "testing",
            preprocess      = GSC_preprocess,
            saved_data_file = f"preprocessed_T{seq_length}",
        )

        def GSC_collate_fn(item): # [waveform, sample_rate, label, speaker_id, utterance_number]
            batchs = [batch for batch, *_ in item]
            labels = [label for _, _, label, *_ in item]
            return torch.stack(batchs).transpose(0, 1), torch.tensor(labels)

        n_classes = 12
        input_channels = 40
        collate_fn = GSC_collate_fn

    elif dataset == 'shd':
        assert seq_length in [30, 50, 100, 150, 200, 250, 1000]

        def SHD_preprocess(times, units, label):  # SHD_T == 1000
            """
            Hanle Zheng \emph{et al.} Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics, \textit{nature communications}, 2023.
            """
            import numpy as np
            data_label = torch.tensor(label, dtype=torch.int64)
            max_unit = 700
            max_time = 1
            dt = 1e-3
            time_step = int(max_time / dt)
            list_input = []
            for i in range(time_step):
                indexs = np.argwhere(times <= i * dt).flatten()
                vals = units[indexs]
                vals = vals[vals > 0]
                vector = np.zeros(max_unit)
                vector[max_unit - vals] = 1
                times = np.delete(times, indexs)
                units = np.delete(units, indexs)
                list_input.append(vector)
            data_input = torch.tensor(np.array(list_input))
            return data_input, data_label

        train_dataset = SpikingHeidelbergDigits(
            root=f"{data_path}/SpikingHeidelbergDigits",
            subset="train",
            preprocess=SHD_preprocess,
            saved_data_file=f"preprocessed_T{seq_length}",
        )
        test_dataset = SpikingHeidelbergDigits(
            root=f"{data_path}/SpikingHeidelbergDigits",
            subset="test",
            preprocess=SHD_preprocess,
            saved_data_file=f"preprocessed_T{seq_length}",
        )

        def SHD_collate_fn(item):
            batchs = [batch for batch, _ in item]
            labels = [label for _, label in item]
            return torch.stack(batchs).transpose(0, 1), torch.tensor(labels)

        n_classes = 20
        input_channels = 700
        collate_fn = SHD_collate_fn

    elif dataset == 'ssc':
        assert seq_length in [50, 100, 150, 200, 250, 1000]
        def SSC_preprocess(times, units, label): # SSC_T == 1000
            """
            Hanle Zheng \emph{et al.} Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics, \textit{nature communications}, 2023.
            """
            import numpy as np
            data_label = torch.tensor(label, dtype=torch.int64)
            max_unit   = 700
            max_time   = 1
            dt         = 4e-3
            time_step  = int(max_time / dt)
            assert time_step == seq_length
            list_input = []
            for i in range(time_step):
                indexs = np.argwhere(times <= i * dt).flatten()
                vals   = units[indexs]
                vals = vals[vals > 0]
                vector = np.zeros(max_unit)
                vector[max_unit - vals] = 1
                times  = np.delete(times, indexs)
                units  = np.delete(units, indexs)
                list_input.append(vector)
            data_input = torch.tensor(np.array(list_input))
            return data_input, data_label

        train_dataset = SpikingSpeechCommands(
            root            = f"{data_path}/SpikingSpeechCommands",
            subset          = "train",
            preprocess      = SSC_preprocess,
            saved_data_file = f"preprocessed_T{seq_length}",
        )
        test_dataset = SpikingSpeechCommands(
            root            = f"{data_path}/SpikingSpeechCommands",
            subset          = "test",
            preprocess      = SSC_preprocess,
            saved_data_file = f"preprocessed_T{seq_length}",
        )

        def SSC_collate_fn(item):
            batchs = [batch for batch, _ in item]
            labels = [label for _, label in item]
            return torch.stack(batchs).transpose(0, 1), torch.tensor(labels)

        n_classes = 35
        input_channels = 700
        collate_fn = SSC_collate_fn

    elif dataset == 'ssc2':
        assert seq_length in [50, 100, 150, 200, 250, 1000]

        def SSC_preprocess(times, units, label):  # Rate coding based preprocess
            data_label = torch.tensor(label, dtype=torch.int64)
            max_unit = 700
            max_time = 1.4
            time_step = seq_length
            data_input = torch.zeros((time_step, max_unit))
            time_step_bins = torch.linspace(0, max_time, time_step)
            time_step_indexes = torch.searchsorted(time_step_bins, torch.tensor(times, dtype=torch.float32)) - 1
            for time_step_index, unit_index in zip(time_step_indexes, units):
                data_input[time_step_index, unit_index] += 1
            data_input -= data_input.mean(axis=(0, 1), keepdims=True)
            data_input /= data_input.std(axis=(0, 1), keepdims=True)
            return data_input, data_label

        train_dataset = SpikingSpeechCommands(
            root            = f"{data_path}/SpikingSpeechCommands",
            subset          = "train",
            preprocess      = SSC_preprocess,
            saved_data_file = f"preprocessed2_T{seq_length}",
        )
        test_dataset = SpikingSpeechCommands(
            root            = f"{data_path}/SpikingSpeechCommands",
            subset          = "test",
            preprocess      = SSC_preprocess,
            saved_data_file = f"preprocessed2_T{seq_length}",
        )

        def SSC_collate_fn(item):
            batchs = [batch for batch, _ in item]
            labels = [label for _, label in item]
            return torch.stack(batchs).transpose(0, 1), torch.tensor(labels)

        n_classes = 35
        input_channels = 700
        collate_fn = SSC_collate_fn

    elif dataset == 'ecg':
        train_dataset = Electrocardiogram(subset="train")
        test_dataset = Electrocardiogram(subset="test")
        input_channels = 4
        n_classes = 6
        collate_fn = None

    elif dataset == '20news':
        train_dataset, val_dataset, test_dataset, n_classes, input_channels = create_20newsgroup_dataset(max_length=seq_length, min_length=min_length)
        collate_fn = None
    elif dataset == 'dvsslr':
        train_dataset, test_dataset, n_classes = create_dvsslr_dataset(root_dir='/datasets/DVSSLR', seq_length=seq_length)
        input_channels = 2
        collate_fn = None
    elif dataset == 'dvslip':
        data_path = '/datasets/dvslip/extract/DVS-Lip'
        # data_path = '/datasets/dvsgesture'
        train_dataset = DVSLipDataset(data_root=data_path, train=True, augment_spatial=True, T=seq_length)
        test_dataset = DVSLipDataset(data_root=data_path, train=False, augment_spatial=False, T=seq_length)

        # train_dataset = DVS_Lip(data_path=data_path, train=True, seq_len=seq_length, num_bins=num_bins)
        # test_dataset = DVS_Lip(data_path=data_path, train=False, seq_len=seq_length, num_bins=num_bins)
        input_channels = 2
        n_classes = 100
        collate_fn = None
    else:
        raise NotImplementedError

    return train_dataset, test_dataset, input_channels, n_classes, collate_fn

def build_imdb_dataloader(batch_size, emb_dim, data_path):
    train_iterator, valid_iterator, test_iterator, text_field = get_imdb_word_dataloader(batch_size, emb_dim, data_path)
    ninp = len(text_field.vocab)
    pad_idx = text_field.vocab.stoi[text_field.pad_token]
    zero_words_embedding = partial(zero_words_in_embedding,
            embedding_size=emb_dim,
            text=text_field,
            pad_idx=pad_idx,
            )
    return train_iterator, valid_iterator, test_iterator, ninp, pad_idx, zero_words_embedding

class DefaultDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        encoder,
        device                  = None,
        batch_size              = 1,
        shuffle                 = None,
        sampler                 = None,
        batch_sampler           = None,
        num_workers             = 0,
        collate_fn              = None,
        pin_memory              = False,
        drop_last               = False,
        timeout                 = 0,
        worker_init_fn          = None,
        multiprocessing_context = None,
        generator               = None,
        *,
        prefetch_factor         = None,
        persistent_workers      = False,
        # pin_memory_device       = "",
    ):
        super(DefaultDataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            # prefetch_factor    = prefetch_factor,
            persistent_workers = persistent_workers,
            # pin_memory_device  = pin_memory_device,
        )
        self.encoder = encoder
        self.device  = device

    def __iter__(self):
        for batch in super(DefaultDataLoader, self).__iter__():
            inputs, labels = batch
            encoded_inputs = self.encoder(inputs)

            if self.device is None:
                yield encoded_inputs, labels
            else:
                encoded_inputs = encoded_inputs.to(self.device)
                labels         = labels.to(self.device)
                yield encoded_inputs, labels




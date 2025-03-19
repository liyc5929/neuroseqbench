import logging
import os.path

import torch
import torchvision
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import transforms

from ..dataset.dvs_gesture import DVS128Gesture, DVS128Gesture_Cached
from ..dataset.augmentation import RandomMixup, RandomCutmix, ClassificationPresetTrain
from ..dataset.adding_problem import AddingProblem
from ..dataset.dvslip.dvs_lip import DVSLipDataset
import numpy as np
from torch.utils.data import Dataset
from functools import partial





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

    elif dataset == 'dvslip':
        from ..dataset.dvslip.dvs_lip import DVSLipDataset
        data_path = '/datasets/dvslip/extract/DVS-Lip'
        # data_path = '/datasets/dvsgesture'
        train_dataset = DVSLipDataset(data_root=data_path, train=True, augment_spatial=True, T=seq_length)
        test_dataset = DVSLipDataset(data_root=data_path, train=False, augment_spatial=False, T=seq_length)
        input_channels = 2
        n_classes = 100
        collate_fn = None
    else:
        raise NotImplementedError

    return train_dataset, test_dataset, input_channels, n_classes, collate_fn


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




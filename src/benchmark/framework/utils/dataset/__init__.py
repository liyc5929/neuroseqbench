from .penn_treebank import PennTreebank

import torchvision
from torchvision.transforms import transforms
import numpy as np
import os
from collections import Counter
from typing import Tuple
import torch


def build_dataset(dataset, data_path):
    if dataset == "psmnist":
        permutation = np.random.permutation(int(784))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(1, 784).t()),
            transforms.Lambda(lambda x: x[permutation])
        ])
        transform_train = transform_test = transform

        train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform_train)
        test_dataset  = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform_test)
        n_classes = 10
        input_channels = 1
        collate_fn = None
    else:
        raise NotImplementedError

    return train_dataset, test_dataset, input_channels, n_classes, collate_fn


def batchify(data: torch.Tensor, bsz: int) -> torch.Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size
        device: torch device to load data

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data


def get_batch(source: torch.Tensor, i: int, seq_len: int, batch_first: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int starting point in the source tensor
        seq_len: backpropagation through time steps
        batch_first: if True, function return shape (BS, seq_len)

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len]

    # map to shape (BS, SeqLen, ntokens)
    if batch_first:
        data = torch.transpose(data, 0, 1)
        target = torch.transpose(target, 0, 1)
    return data, target.flatten()


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path), f'File does not exist: {path}'
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


def build_lm_dataloader(dataset, data_path, train_batch_size, val_batch_size=10, test_batch_size=1):
    """
    Returns Wiki-Text-2 train, val and test split as well as number of tokens

    Args:
        root: directory to store and load the dataset
        dset: choice of WT2 or PTB (Wikitext-2 or PennTreeBank)
        batch_size: batch size
        device: a torch device, e.g. 'cpu' or 'cuda'
    """
    if dataset == 'WT2': # WikiText-2
        corpus = Corpus(os.path.join(data_path, 'wikitext-2'))
    elif dataset == 'PTB': # PennTreebank
        corpus = Corpus(os.path.join(data_path, 'PennTreebank'))
    elif dataset == 'WT103': # WikiText-103
        corpus = Corpus(os.path.join(data_path, 'wikitext-103'))
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented. Choose either WT2 or PTB!')

    train_data = batchify(corpus.train, train_batch_size)
    val_data = batchify(corpus.valid, val_batch_size)
    test_data = batchify(corpus.test, test_batch_size)

    return train_data, val_data, test_data, len(corpus.dictionary)
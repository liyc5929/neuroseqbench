
import os
import torch
from typing import Union
from torch.utils.data import Dataset
from collections import Counter


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
        self.data = {
            "train": self.tokenize(os.path.join(path, "train.txt")),
            "valid": self.tokenize(os.path.join(path, "valid.txt")),
            "test":  self.tokenize(os.path.join(path, "test.txt")),
        }

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path), f"File does not exist: {path}"
        # Add words to the dictionary
        with open(path, "r") as f:
            tokens = 0
            for line in f:
                words = line.split() + ["<eos>"]
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, "r") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids


class PennTreebank(Dataset):
    """
    item size: (time_step, chunk_num)
    """
    def __init__(self, 
        root,
        subset                           = "train", # "train", "valid" or "test"
        data_source                      = "src/benchmark/framework/utils/datasource/PennTreebank", 
        time_step: int                   = 1,
        chunk_num: int                   = 1,       # generally be the batch size
        device: Union[str, torch.device] = "cpu"
    ):
        self.root      = root
        self.subset    = subset
        self.time_step = time_step
        self.chunk_num = chunk_num

        os.makedirs(os.path.join(root), exist_ok=True)
        scratch_data_root = os.path.join(root, "scratch")
        # Check or generate scratch data and save to the `scratch_data_root`
        if not os.path.exists(scratch_data_root):
            print(f"The `scratch_data_root` does not exist, `train`, `valid` and `test` scratch corpus data generating of `{self.__class__.__name__}` start.")
            os.makedirs(scratch_data_root)

            # Fetching and cutting original data
            corpus = Corpus(path=data_source)
            for filename in ("train", "valid", "test"):
                torch.save(corpus.data[filename], os.path.join(scratch_data_root, f"{filename}.pt"))

            # Get subset data
            corpus_data = corpus.data[subset]
        else:
            print(f"The `scratch_data_root` exists, data preloading of `{self.__class__.__name__}` from path `{scratch_data_root}` start.")
            corpus_data = torch.load(os.path.join(scratch_data_root, f"{subset}.pt"))

        chunk_len      = corpus_data.size(0) // chunk_num
        corpus_data    = corpus_data[: chunk_len * chunk_num].view(chunk_num, chunk_len).t().contiguous()
        data_len       = (corpus_data.size(0) - 1) // self.time_step
        self.inputs    = corpus_data[:  data_len * self.time_step    ].view(data_len, self.time_step, -1).contiguous().to(device)
        self.labels    = corpus_data[1: data_len * self.time_step + 1].view(data_len, -1).contiguous().to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, n):
        return (self.inputs[n], self.labels[n])

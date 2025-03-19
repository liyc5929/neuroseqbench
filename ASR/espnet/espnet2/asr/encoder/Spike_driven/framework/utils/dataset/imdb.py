from pathlib import Path
from typing import Sequence
from functools import partial
# import torchtext
# from datasets import load_dataset, DatasetDict, Value
from torch import nn
import logging
import pickle
# from torchtext import data
# from torchtext import datasets
import torch

def is_list(x):
    return isinstance(x, Sequence) and not isinstance(x, str)


class SequenceDataset:
    registry = {}
    _name_ = NotImplementedError("Dataset must have shorthand name")

    # Since subclasses do not specify __init__ which is instead handled by this class
    # Subclasses can provide a list of default arguments which are automatically registered as attributes
    # TODO apparently there is a python 3.8 decorator that basically does this
    @property
    def init_defaults(self):
        return {}

    # https://www.python.org/dev/peps/pep-0487/#subclass-registration
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls._name_] = cls

    def __init__(
            self,
            _name_,
            seq_length,
            data_dir=None,
            **dataset_cfg,
    ):
        assert _name_ == self._name_
        self.l_max = seq_length
        self.data_dir = Path(data_dir).absolute() if data_dir is not None else None


        # Add all arguments to self
        init_args = self.init_defaults
        init_args.update(
            dataset_cfg
        )  # TODO this overrides the default dict which is bad
        for k, v in init_args.items():
            setattr(self, k, v)

        self.init()  # Extra init stuff if desired # TODO get rid of this

        # train, val, test datasets must be set by class instantiation
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def init(self):
        pass

    def setup(self):
        """This method should set self.dataset_train, self.dataset_val, and self.dataset_test"""
        raise NotImplementedError

    def split_train_val(self, val_split):
        train_len = int(len(self.dataset_train) * (1.0 - val_split))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset_train,
            (train_len, len(self.dataset_train) - train_len),
            generator=torch.Generator().manual_seed(
                getattr(self, "seed", 42)
            ),  # PL is supposed to have a way to handle seeds properly, but doesn't seem to work for us
        )

    @staticmethod
    def collate_fn(batch, resolution=1):
        """batch: list of (x, y) pairs"""

        def _collate(batch, resolution=1):
            # From https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
            elem = batch[0]
            if isinstance(elem, torch.Tensor):
                out = None
                if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum(x.numel() for x in batch)
                    storage = elem.storage()._new_shared(numel)
                    out = elem.new(storage)
                x = torch.stack(batch, dim=0, out=out)
                if resolution is not None:
                    x = x[:, ::resolution]  # assume length is first axis after batch
                return x
            else:
                return torch.tensor(batch)

        x, y = zip(*batch)
        # Drop every nth sample
        # x = torch.stack(x, dim=0)[:, ::resolution]
        # y = torch.LongTensor(y)
        # y = torch.tensor(y)
        # y = torch.stack(y, dim=0)
        x = _collate(x, resolution=resolution)
        y = _collate(y, resolution=None)
        return x, y

    def train_dataloader(self, train_resolution=None, eval_resolutions=None, **kwargs):
        if train_resolution is None:
            train_resolution = [1]
        if not is_list(train_resolution):
            train_resolution = [train_resolution]
        assert len(train_resolution) == 1, "Only one train resolution supported for now"

        return self._dataloader(
            self.dataset_train,
            resolutions=train_resolution,
            shuffle=True,
            **kwargs,
        )[0]

    def val_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_val, **kwargs)

    def test_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_test, **kwargs)

    def _eval_dataloader(self, dataset, train_resolution=None, eval_resolutions=None, **kwargs):
        if eval_resolutions is None:
            eval_resolutions = [1]
        if not is_list(eval_resolutions):
            eval_resolutions = [eval_resolutions]

        kwargs["shuffle"] = False if "shuffle" not in kwargs else kwargs["shuffle"]
        dataloaders = self._dataloader(
            dataset,
            resolutions=eval_resolutions,
            # shuffle=False,
            **kwargs,
        )

        return (
            {
                str(res) if res > 1 else None: dl
                for res, dl in zip(eval_resolutions, dataloaders)
            }
            if dataloaders is not None
            else None
        )

    def _dataloader(self, dataset, resolutions, **loader_args):
        if dataset is None:
            return None

        # if self.tbptt:
        #     DataLoader = partial(
        #         TBPTTDataLoader, chunk_len=self.chunk_len, overlap_len=self.overlap_len
        #     )
        # else:
        DataLoader = torch.utils.data.DataLoader

        return [
            DataLoader(
                dataset=dataset,
                collate_fn=partial(self.collate_fn, resolution=resolution)
                if self.collate_fn is not None
                else None,
                **loader_args,
            )
            for resolution in resolutions
        ]

    def __str__(self):
        return self._name_


class IMDB(SequenceDataset):
    _name_ = "imdb"
    d_output = 2
    l_output = 0

    @property
    def init_defaults(self):
        return {
            # "l_max": 4096,
            "level": "char",
            "min_freq": 15,
            "seed": 42,
            "val_split": 0.0,
            "append_bos": False,
            "append_eos": True,
            # 'max_vocab': 135,
            "n_workers": 4,  # Only used for tokenizing dataset before caching
        }

    @property
    def n_tokens(self):
        return len(self.vocab)

    def init(self):
        """If cache_dir is not None, we'll cache the processed dataset there."""
        self.data_dir = self.data_dir  # or default_data_path / self._name_
        self.cache_dir = self.data_dir / "cache"
        assert self.level in [
            "word",
            "char",
        ], f"level {self.level} not supported"

    def prepare_data(self):
        if self.cache_dir is None:  # Just download the dataset
            load_dataset(self._name_, cache_dir=self.data_dir)
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        print(
            f"IMDB {self.level} level | min_freq {self.min_freq} | vocab size {len(self.vocab)}"
        )
        dataset.set_format(type="torch", columns=["input_ids", "label"])

        # Create all splits
        dataset_train, self.dataset_test = dataset["train"], dataset["test"]
        if self.val_split == 0.0:
            # Use test set as val set, as done in the LRA paper
            self.dataset_train, self.dataset_val = dataset_train, None
        else:
            train_val = dataset_train.train_test_split(
                test_size=self.val_split, seed=self.seed
            )
            self.dataset_train, self.dataset_val = (
                train_val["train"],
                train_val["test"],
            )

        def collate_batch(batch, resolution=1):
            xs, ys = zip(*[(data["input_ids"], data["label"]) for data in batch])
            xs, ys = list(xs), list(ys)

            lengths = torch.tensor([len(x) for x in xs])
            # assert torch.is_tensor(xs[0])
            # assert len(xs[0].shape) == 1
            # xs[0] = torch.cat((xs[0], torch.tensor([self.vocab["<pad>"]] * (self.l_max - xs[0].size(0)), dtype=xs[0].dtype)), 0)
            # assert xs[0].size(0) == self.l_max
            xs = nn.utils.rnn.pad_sequence(
                xs, padding_value=self.vocab["<pad>"], batch_first=True
            )
            # print(f"padding_value: {self.vocab['<pad>']}")
            ys = torch.tensor(ys)
            return xs, ys, lengths

        self.collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)
        dataset = load_dataset(self._name_, cache_dir=self.data_dir)
        dataset = DatasetDict(train=dataset["train"], test=dataset["test"])
        if self.level == "word":
            tokenizer = torchtext.data.utils.get_tokenizer(
                "spacy", language="en_core_web_sm"
            )
        else:  # self.level == 'char'
            tokenizer = list  # Just convert a string to a list of chars
        # Account for <bos> and <eos> tokens
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {"tokens": tokenizer(example["text"])[:l_max]}
        dataset = dataset.map(
            tokenize,
            remove_columns=["text"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens"],
            min_freq=self.min_freq,
            specials=(
                    ["<pad>", "<unk>"]
                    + (["<bos>"] if self.append_bos else [])
                    + (["<eos>"] if self.append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        numericalize = lambda example: {
            "input_ids": vocab(
                (["<bos>"] if self.append_bos else [])
                + example["tokens"]
                + (["<eos>"] if self.append_eos else [])
            )
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-level-{self.level}-min_freq-{self.min_freq}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"


def create_lra_imdb_classification_dataloader(batch_size=50,
                                           data_dir='./datasets',
                                           ):
    """

    :param cache_dir:		(str):		Not currently used.
    :param bsz:				(int):		Batch size.
    :param seed:			(int)		Seed for shuffling data.
    :return:
    """
    print("[*] Generating LRA-text (IMDB) Classification Dataset")
    dataset = SequenceDataset.registry['imdb']('imdb', data_dir=data_dir)
    dataset.setup()
    train_loader = dataset.train_dataloader(batch_size=batch_size)
    test_loader = dataset.test_dataloader(batch_size=batch_size, drop_last=False, shuffle=False)

    N_CLASSES = dataset.d_output
    SEQ_LENGTH = dataset.l_max
    VOCAB_SIZE = len(dataset.vocab)
    # IN_DIM = 135  # We should probably stop this from being hard-coded.
    # TRAIN_SIZE = len(dataset.dataset_train)

    return train_loader, test_loader, N_CLASSES, SEQ_LENGTH, VOCAB_SIZE

def create_lra_imdb_classification_dataset(seq_length, data_dir='./datasets',
                                           ):
    """

    :param cache_dir:		(str):		Not currently used.
    :param bsz:				(int):		Batch size.
    :param seed:			(int)		Seed for shuffling data.
    :return:
    """
    print("[*] Generating LRA-text (IMDB) Classification Dataset")
    dataset = SequenceDataset.registry['imdb']('imdb', seq_length=seq_length, data_dir=data_dir)
    dataset.setup()
    train_dataset = dataset.dataset_train
    test_dataset = dataset.dataset_test

    N_CLASSES = dataset.d_output
    # SEQ_LENGTH = dataset.l_max
    VOCAB_SIZE = len(dataset.vocab)
    collate_fn = dataset.collate_fn
    # IN_DIM = 135  # We should probably stop this from being hard-coded.
    # TRAIN_SIZE = len(dataset.dataset_train)

    return train_dataset, test_dataset, N_CLASSES, VOCAB_SIZE, collate_fn







def get_imdb_word_dataloader(batch_size, embedding_size, data_dir='./datasets'):
    text = data.Field(tokenize='spacy', include_lengths=False)
    label = data.LabelField(dtype=torch.float)
    train_data, test_data = datasets.IMDB.splits(text, label, root=data_dir)
    train_data, valid_data = train_data.split()

    max_vocab_size = 25_000
    text.build_vocab(train_data,
                     max_size=max_vocab_size,
                     vectors="glove.6B." + str(embedding_size) + "d",
                     unk_init=torch.Tensor.normal_)
    label.build_vocab(train_data)
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                               batch_size=batch_size, sort=False)

    return train_iterator, valid_iterator, test_iterator, text


def zero_words_in_embedding(model, embedding_size, text, pad_idx):
    pretrained_embeddings = text.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = text.vocab.stoi[text.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_size)
    model.embedding.weight.data[pad_idx] = torch.zeros(embedding_size)


if __name__ == '__main__':
    data_dir = './datasets'
    batch_size = 2
    train_loader, test_loader, N_CLASSES, SEQ_LENGTH, VOCAB_SIZE = create_lra_imdb_classification_dataloader(
        batch_size=batch_size, data_dir=data_dir)
    print(f"vocab size: {VOCAB_SIZE}")
    for batch_idx, (inputs, targets, lengths) in enumerate(train_loader):
        # print(type(batch))
        print(inputs.size())
        print(targets)
        print(lengths)
        exit()
        # prep_batch(batch, SEQ_LENGTH, IN_DIM)

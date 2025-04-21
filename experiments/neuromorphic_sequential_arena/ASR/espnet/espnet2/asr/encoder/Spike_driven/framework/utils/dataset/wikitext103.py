import torch
from torch.utils.data import Dataset
try:
    from datasets import load_dataset
except:
    pass
try:

    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
except BaseException as e:
    get_tokenizer = lambda *args, **kwargs: (args, kwargs)
    build_vocab_from_iterator = lambda *args, **kwargs: (args, kwargs)


class Wikitext103Dataset(Dataset):
    def __init__(self, data_mode="word", split="train", data_dir="./data/wikitext103"):
        assert split in ["train", "valid", "test"], f"Invalid {self.__class__.__name__} split."
        assert data_mode in ["word", "char"], f"Invalid {self.__class__.__name__} data_mode."
        if split == "valid": split = "validation"
        self.split     = split
        self.data_mode = data_mode
        self.tokenizer = get_tokenizer("basic_english") if data_mode == "word" else list
        self.data      = load_dataset("parquet", data_dir=data_dir)[split]["text"]
        self.vocab     = build_vocab_from_iterator(map(self.tokenizer, self.data), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.data = [torch.tensor(self.vocab(self.tokenizer(item)), dtype=torch.long) for item in self.data]
        self.data = torch.cat(self.data) if self.data else self.data
    
    def __len__(self):
        return len(self.data) - 1
    
    def __getitem__(self, idx):
        return self.data[idx], self.data[idx + 1]

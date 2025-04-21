from sklearn.datasets import fetch_20newsgroups
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import re
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def clean_20news_data(text_str):
    """
    Clean up 20NewsGroups text data, from CogLTX: https://github.com/Sleepychord/CogLTX/blob/main/20news/process_20news.py
    // SPDX-License-Identifier: MIT
    :param text_str: text string to clean up
    :return: clean text string
    """
    tmp_doc = []
    for words in text_str.split():
        if ':' in words or '@' in words or len(words) > 60:
            pass
        else:
            c = re.sub(r'[>|-]', '', words)
            # c = words.replace('>', '').replace('-', '')
            if len(c) > 0:
                tmp_doc.append(c)
    tmp_doc = ' '.join(tmp_doc)
    tmp_doc = re.sub(r'\([A-Za-z \.]*[A-Z][A-Za-z \.]*\) ', '', tmp_doc)
    return tmp_doc

def prepare_20news_data():
    """
    Load the 20NewsGroups datasets and split the original train set into train/dev sets
    :return: dicts of lists of documents and labels and number of labels
    """
    text_set = {}
    label_set = {}
    test_set = fetch_20newsgroups(subset='test', random_state=21)
    text_set['test'] = [clean_20news_data(text) for text in test_set.data]
    label_set['test'] = test_set.target

    train_set = fetch_20newsgroups(subset='train', random_state=21)
    train_text = [clean_20news_data(text) for text in train_set.data]
    train_label = train_set.target

    # take 10% of the train set as the dev set
    text_set['train'], text_set['dev'], label_set['train'], label_set['dev'] = train_test_split(train_text,
                                                                                                train_label,
                                                                                                test_size=0.10,
                                                                                                random_state=21)
    enc = preprocessing.LabelEncoder()
    enc.fit(label_set['train'])
    num_labels = len(enc.classes_)

    # vectorize labels as zeros and ones
    vectorized_labels = {}
    for split in ['train', 'dev', 'test']:
        vectorized_labels[split] = enc.transform(label_set[split])

    return text_set, vectorized_labels, num_labels

class TruncatedDataset(torch.utils.data.Dataset):
    def __init__(self, text, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = text
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        # return {
        #     'ids': torch.tensor(ids),
        #     'mask': torch.tensor(mask),
        #     'token_type_ids': torch.tensor(token_type_ids),
        #     'labels': torch.tensor(self.labels[index])
        # }
        return torch.tensor(ids), torch.tensor(self.labels[index])
def get_long_texts_and_labels(text_dict, label_dict, tokenizer, max_length=512):
    """
    Find texts that have more than a given max token length and their labels
    :param text_dict: dict of lists of texts for train/dev/test splits, keys=['train', 'dev', 'test']
    :param label_dict: dict of lists of labels for train/dev/test splits, keys=['train', 'dev', 'test']
    :param tokenizer: tokenizer of choice e.g. LongformerTokenizer, BertTokenizer
    :param max_length: maximum length of sequence e.g. 512
    :return: dicts of lists of texts with more than the max token length and their labels
    """
    long_text_set = {'train': [], 'dev': [], 'test': []}
    long_label_set = {'train': [], 'dev': [], 'test': []}
    for split in ['train', 'dev', 'test']:
        long_text_idx = []
        for idx, text in enumerate(text_dict[split]):
            if len(tokenizer.tokenize(text)) > (max_length - 2):
                long_text_idx.append(idx)
        long_text_set[split] = [text_dict[split][i] for i in long_text_idx]
        long_label_set[split] = [label_dict[split][i] for i in long_text_idx]
    return long_text_set, long_label_set


def create_20newsgroup_dataset(max_length=1024, min_length=0):
    text_set, label_set, num_classes = prepare_20news_data()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    vocab_size = len(tokenizer)
    text_set, label_set = get_long_texts_and_labels(text_set, label_set, tokenizer, max_length=min_length)
    train_dataset = TruncatedDataset(text_set['train'], label_set['train'], tokenizer, max_length)
    val_dataset = TruncatedDataset(text_set['dev'], label_set['dev'], tokenizer, max_length)
    test_dataset = TruncatedDataset(text_set['test'], label_set['test'], tokenizer, max_length)

    return train_dataset, val_dataset, test_dataset, num_classes, vocab_size



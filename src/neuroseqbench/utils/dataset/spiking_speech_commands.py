import os
import h5py
import tqdm
import torch
import numpy as np
from typing import Union
from torch.utils.data import Dataset


class SpikingSpeechCommands(Dataset):
    def __init__(self, root, subset, saved_data_file="preprocessed", preprocess=None, device: Union[str, torch.device]="cpu"):
        os.makedirs(os.path.join(root, saved_data_file), exist_ok=True)
        preprocessed_data_root = os.path.join(root, saved_data_file, subset)
        if os.path.exists(preprocessed_data_root): 
            # Data preloading
            print(f"The `saved_data_file` exists, data preloading of `{self.__class__.__name__}` from path `{preprocessed_data_root}` start.")
            h5file = h5py.File(f"{preprocessed_data_root}/ssc_preprocessed_data.h5", "r")
            input_iter = h5file["inputs"]
            label_iter = h5file["labels"]
            self.inputs = []
            self.labels = []
            for i in tqdm.tqdm(range(len(label_iter))):
                self.inputs.append(torch.tensor(input_iter[i], dtype=torch.float32).to(device))
                self.labels.append(torch.tensor(label_iter[i], dtype=torch.int64).to(device))
        else: 
            # Fetching original data
            h5file = h5py.File(f"{root}/ssc_{subset}.h5", "r")
            times_iter = h5file["spikes"]["times"]
            units_iter = h5file["spikes"]["units"] 
            label_iter = h5file["labels"]

            # Data preprocessing
            self.preprocess = preprocess
            self.inputs = []
            self.labels = []
            os.mkdir(preprocessed_data_root)
            print(f"The preprocessing of `{self.__class__.__name__}` start.")
            for i in tqdm.tqdm(range(len(label_iter))):
                times = times_iter[i]
                units = units_iter[i]
                label = label_iter[i]
                if preprocess is not None:
                    data_input, data_label = self.preprocess(times, units, label)
                else: 
                    raise NotImplementedError(f"The `preprocess` in class `{self.__class__.__name__}` cannot be None.")
                self.inputs.append(data_input.to(device))
                self.labels.append(data_label.to(device))

            # Data saving
            print(f"The saving to path `{preprocessed_data_root}` start.")
            with h5py.File(f"{preprocessed_data_root}/ssc_preprocessed_data.h5", "w") as fp:
                saved_inputs = fp.create_dataset("inputs", (len(self.inputs), *self.inputs[0].shape), dtype=np.float32)
                saved_labels = fp.create_dataset("labels", (len(self.labels), *self.labels[0].shape), dtype=np.int64)

                for i in tqdm.tqdm(range(len(self.labels))):
                    saved_inputs[i] = self.inputs[i].cpu()
                    saved_labels[i] = self.labels[i].cpu()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, n):
        return (self.inputs[n], self.labels[n])

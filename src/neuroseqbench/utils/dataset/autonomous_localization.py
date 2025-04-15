import os
import h5py
import tqdm
import torch
import numpy as np
from typing import Union
from torch.utils.data import Dataset
import torch.nn.functional as F


class AL(Dataset):
    def __init__(self, 
        root, 
        subset                           = "train",
        num_data                         = 50000, # Number of data in dataset
        seq_length                       = 400,   # Length of the action command
        capacity                         = 20,    # Probability of turning left/right
        device: Union[str, torch.device] = "cpu",
    ):
        saved_data_file = f"AL"
        os.makedirs(os.path.join(root, saved_data_file), exist_ok=True)
        preprocessed_data_root = os.path.join(root, saved_data_file, f"{subset}_{seq_length}_{capacity}")
        if os.path.exists(preprocessed_data_root):
            # Data preloading
            print(f"The `saved_data_file` path exists, data preloading of `{self.__class__.__name__}` from path `{preprocessed_data_root}` start.")
            h5file = h5py.File(f"{preprocessed_data_root}/preprocessed_data_num({num_data})_seqlen({seq_length}).h5", "r")
            input_iter = h5file["inputs"]
            label_iter = h5file["labels"]
            self.inputs = []
            self.labels = []
            for i in tqdm.tqdm(range(len(label_iter))):
                self.inputs.append(torch.tensor(input_iter[i], dtype=torch.float32).to(device))
                self.labels.append(torch.tensor(label_iter[i], dtype=torch.int64).to(device))
        else: 
            # Data generating (X -- inputs, Y -- labels)
            probabilities = [0.5-(capacity/seq_length), 0.5-(capacity/seq_length), capacity/seq_length, capacity/seq_length]
            distribution = torch.multinomial(torch.tensor(probabilities), num_data * seq_length, replacement=True)
            action = distribution.view(num_data, seq_length)
            X = F.one_hot(action, num_classes=4)[..., 1:].float()
            Y = self.compute(action)
            Y = Y.view(-1)

            # Data preloading
            self.inputs = []
            self.labels = []
            os.mkdir(preprocessed_data_root)
            print(f"The preprocessing of `{self.__class__.__name__}` start.")
            for i in tqdm.tqdm(range(len(Y))):
                data_input, data_label = X[i], Y[i].long()
                self.inputs.append(data_input.to(device))
                self.labels.append(data_label.to(device))

            # Data saving
            print(f"The saving to path `{preprocessed_data_root}` start.")
            with h5py.File(f"{preprocessed_data_root}/preprocessed_data_num({num_data})_seqlen({seq_length}).h5", "w") as fp:
                saved_inputs = fp.create_dataset("inputs", (len(self.inputs), *self.inputs[0].shape), dtype=np.float32)
                saved_labels = fp.create_dataset("labels", (len(self.labels), *self.labels[0].shape), dtype=np.int64)

                for i in tqdm.tqdm(range(len(self.labels))):
                    saved_inputs[i] = self.inputs[i].cpu()
                    saved_labels[i] = self.labels[i].cpu()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, n):
        return (self.inputs[n], self.labels[n])

    def compute(self, action):
        num_data, seq_length = action.shape
        Y = torch.zeros((num_data, 1))

        for i in range(num_data):
            coordinate = torch.tensor([0, 0, 0])
            for j in range(seq_length):
                act = action[i, j].item()
                if act == 0:
                    # stop
                    pass
                elif act == 1:
                    # go straight
                    if coordinate[2] == 0:
                        coordinate[1] += 1
                    elif coordinate[2] == 1:
                        coordinate[0] += 1
                    elif coordinate[2] == 2:
                        coordinate[1] -= 1
                    elif coordinate[2] == 3:
                        coordinate[0] -= 1
                elif act == 2:
                    # turn left
                    coordinate[2] -= 1
                elif act == 3:
                    # turn right
                    coordinate[2] += 1
                # keep the direction to 0, 90, 180, 270 degree
                coordinate[2] = coordinate[2] % 4
            # decison: left or right
            Y[i, 0] = coordinate[0].clamp(0, 1)
        return Y

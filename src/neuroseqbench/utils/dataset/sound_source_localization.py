"""
According to: Xinyuan Qian \emph{et al.}, SLoClas: A Database for Joint Sound Localization and Classification, 2021.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataloaderClass(Dataset):
    def __init__(self, X_data, Y_data):
        self.x_data = X_data
        self.y_data = Y_data
        self.len = X_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class SLoClas:
    def __new__(cls, train_data, test_data, sequence_length=1024):
        train_x, test_x = np.array(train_data["X"]), np.array(test_data["X"])
        train_x, test_x = torch.from_numpy(train_x), torch.from_numpy(test_x)
        train_y, test_y = torch.from_numpy(np.array(train_data["Y"])), torch.from_numpy(np.array(test_data["Y"]))  # Binary target
    
        train_x = train_x.view(sequence_length, 4, -1).permute(2, 0, 1) # [T, C, B] -> [B, T, C]
        test_x = test_x.view(sequence_length, 4, -1).permute(2, 0, 1)
    
        train_y = (train_y.view(-1) // 5 - 1).long()
        test_y = (test_y.view(-1) // 5 - 1).long()
    
        num_train = train_x.shape[0]
        indices = list(range(num_train))
        np.random.seed(1234)
        np.random.shuffle(indices)
        train_x = train_x[indices, ]
        train_y = train_y[indices, ]
        train_dataset = MyDataloaderClass(train_x, train_y)
        test_dataset = MyDataloaderClass(test_x, test_y)
    
        return train_dataset, test_dataset

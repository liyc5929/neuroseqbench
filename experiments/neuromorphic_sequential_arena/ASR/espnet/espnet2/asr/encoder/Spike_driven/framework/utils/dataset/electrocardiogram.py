import os
import h5py
import tqdm
import torch
import numpy as np
from typing import Union
from torch.utils.data import Dataset
import scipy.io
from torch.utils.data import TensorDataset


class Electrocardiogram(Dataset):
    """
    An ECG dataset
    Laguna P, Mark RG, Goldberger AL, Moody GB. A Database for Evaluation of Algorithms for Measurement of QT and Other Waveform Intervals in the ECG. Computers in Cardiology 24:673-676 (1997).
    """
    def __init__(self, 
        subset: str,  # "train" or "test"
    ):
        # Data preloading
        data_mat = scipy.io.loadmat(f"src/benchmark/framework/utils/datasource/ECG/QTDB_{subset}.mat")
        data_dt, data_x, data_y = self.convert_dataset_wtime(data_mat)
        data_max_i = self.load_max_i(data_mat)
        self.tensors = (torch.from_numpy(data_x * 1.0).float(), torch.from_numpy(data_y))

    def convert_dataset_wtime(self, mat_data):
        X = mat_data["x"]
        Y = mat_data["y"]
        t = mat_data["t"]
        Y = np.argmax(Y[:, :, :], axis=-1)
        d1, d2 = t.shape

        dt = np.zeros((d1,d2))
        for trace in range(d1):
            dt[trace, 0] = 1
            dt[trace, 1:] = t[trace, 1:] - t[trace, :-1]
        return dt, X, Y

    def load_max_i(self, mat_data):
        max_i = mat_data["max_i"]
        return np.array(max_i.squeeze(),dtype=np.float16)

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

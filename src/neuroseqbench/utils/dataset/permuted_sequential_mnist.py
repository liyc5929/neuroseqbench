import os
import h5py
import tqdm
import torch
import numpy as np
from typing import Union
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class PSMNIST(Dataset):
    permutation = None

    def __init__(self,
            root, 
            train                            = True, 
            download                         = True, 
            time_step                        = 1, 
            device: Union[str, torch.device] = "cpu",
        ):
        self.root      = root
        self.time_step = time_step

        class_data_root = os.path.join(self.root, self.__class__.__name__)
        saved_data_root = os.path.join(class_data_root, f"preprocessed_T{time_step}")
        os.makedirs(class_data_root, exist_ok=True)
        os.makedirs(saved_data_root, exist_ok=True)
        preprocessed_data_root = os.path.join(saved_data_root, "train" if train else "test")
        if os.path.exists(preprocessed_data_root): 
            # Data preloading
            print(f"The `saved_data_file` exists, data preloading of `{self.__class__.__name__}` from path `{preprocessed_data_root}` start.")
            h5file = h5py.File(os.path.join(preprocessed_data_root, "psmnist_preprocessed_data.h5"), "r")
            input_iter = h5file["inputs"]
            label_iter = h5file["labels"]
            self.inputs = []
            self.labels = []
            for i in tqdm.tqdm(range(len(label_iter))):
                self.inputs.append(torch.tensor(input_iter[i], dtype=torch.float32).to(device))
                self.labels.append(torch.tensor(label_iter[i], dtype=torch.int64).to(device))
        else: 
            # Data preprocessing
            if PSMNIST.permutation is None:
                PSMNIST.permutation = np.random.permutation(int(self.time_step))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(1, 784).t()),
                transforms.Lambda(lambda x: x[PSMNIST.permutation])
            ])
            source_data = MNIST(
                root      = self.root,
                train     = train,
                download  = download,
                transform = transform
            )
            self.inputs = []
            self.labels = []
            os.mkdir(preprocessed_data_root)
            print(f"The preprocessing of `{self.__class__.__name__}` start.")
            for i in tqdm.tqdm(range(source_data.__len__())):
                data_input, data_label = source_data.__getitem__(i)
                self.inputs.append(data_input.to(device))
                self.labels.append(torch.tensor(data_label, dtype=torch.int64, device=device))

            # Data saving
            print(f"The saving to path `{preprocessed_data_root}` start.")
            with h5py.File(f"{preprocessed_data_root}/psmnist_preprocessed_data.h5", "w") as fp:
                saved_inputs = fp.create_dataset("inputs", (len(self.inputs), *self.inputs[0].shape), dtype=np.float32)
                saved_labels = fp.create_dataset("labels", (len(self.labels), *self.labels[0].shape), dtype=np.int64)

                for i in tqdm.tqdm(range(len(self.labels))):
                    saved_inputs[i] = self.inputs[i].cpu()
                    saved_labels[i] = self.labels[i].cpu()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, n):
        return (self.inputs[n], self.labels[n])

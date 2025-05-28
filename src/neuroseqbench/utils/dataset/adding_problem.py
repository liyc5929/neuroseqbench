import os
import h5py
import tqdm
import torch
import numpy as np
from typing import Union
from torch.utils.data import Dataset


class AddingProblem(Dataset):
    def __init__(self, 
        root, 
        subset                           = "train", 
        is_binary                        = True,
        num_data                         = 50000, # Number of data in dataset
        seq_length                       = 100,   # Length of the adding problem data
        preprocess                       = None, 
        seed                             = 0,
        device: Union[str, torch.device] = "cpu"
    ):
        saved_data_file = f"{'Binary' if is_binary else ''}AddingProblem"
        os.makedirs(os.path.join(root, saved_data_file), exist_ok=True)
        preprocessed_data_root = os.path.join(root, saved_data_file, f"{subset}_{seq_length}")
        if os.path.exists(preprocessed_data_root):
            # Load saved data
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
            torch.manual_seed(seed)
            if is_binary:
                total_samples = num_data * 80
                X_num = ((torch.rand(total_samples, 1, seq_length) < 0.5) * 1.)
                X_mask = torch.zeros([total_samples, 1, seq_length])
                Y = torch.zeros([total_samples, 1])
                positions = np.array([np.random.choice(seq_length, size=9, replace=False) for _ in range(total_samples)])
                # Use advanced indexing to set the mask and compute Y
                print(f"The data generating of `{self.__class__.__name__}` start.")
                for i in tqdm.tqdm(range(9)):
                    X_mask[np.arange(total_samples), 0, positions[:, i]] = 1
                    Y += X_num[np.arange(total_samples), 0, positions[:, i]].unsqueeze(1)
    
                X = torch.cat((X_num, X_mask), dim=1)
                Y = Y.view(-1).type(torch.LongTensor)
                # Balance the samples in each class
                X, Y = self.balance_classes(X, Y, num_data // 10)
            else:
                X_num = torch.rand([num_data, 1, seq_length])
                X_mask = torch.zeros([num_data, 1, seq_length])
                Y = torch.zeros([num_data, 1])
                print(f"The data generating of `{self.__class__.__name__}` start.")
                for i in tqdm.tqdm(range(num_data)):
                    positions = np.random.choice(seq_length, size=2, replace=False)
                    X_mask[i, 0, positions[0]] = 1
                    X_mask[i, 0, positions[1]] = 1
                    Y[i, 0] = X_num[i, 0, positions[0]] + X_num[i, 0, positions[1]]
                X = torch.cat((X_num, X_mask), dim=1)
            X = X.transpose(1, 2)

            # Data preloading and preprocessing
            self.preprocess = preprocess
            self.inputs = []
            self.labels = []
            os.mkdir(preprocessed_data_root)
            print(f"The preprocessing of `{self.__class__.__name__}` start.")
            for i in tqdm.tqdm(range(len(Y))):
                if preprocess is not None:
                    data_input, data_label = self.preprocess(X[i], Y[i])
                else: 
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

    def balance_classes(self, X, Y, samples_per_class):
        # Convert to numpy arrays for easier manipulation
        # X -- inputs, Y -- labels
        X = X.numpy()
        Y = Y.numpy()
    
        # Find the unique classes and their counts
        unique_classes, class_counts = np.unique(Y, return_counts=True)
        # Create lists to hold the balanced samples
        X_balanced = []
        Y_balanced = []
    
        # Resample each class to have the same number of samples
        for cls in unique_classes:
            # Get all samples for the current class
            X_cls = X[Y == cls]
            Y_cls = Y[Y == cls]
    
            # Resample the current class to have min_samples samples
            # If there are more samples than needed, randomly select samples_per_class samples
            if len(X_cls) > samples_per_class:
                indices = np.random.choice(len(X_cls), samples_per_class, replace=False)
            # If there are fewer samples than needed, randomly sample with replacement
            else:
                indices = np.random.choice(len(X_cls), samples_per_class, replace=True)
    
            X_resampled = X_cls[indices]
            Y_resampled = Y_cls[indices]
            # Append the resampled data to the balanced lists
            X_balanced.append(X_resampled)
            Y_balanced.append(Y_resampled)
    
        # Concatenate the balanced lists into numpy arrays
        X_balanced = np.concatenate(X_balanced)
        Y_balanced = np.concatenate(Y_balanced)
    
        # Convert back to torch tensors
        X_balanced = torch.tensor(X_balanced)
        Y_balanced = torch.tensor(Y_balanced)
    
        return X_balanced, Y_balanced

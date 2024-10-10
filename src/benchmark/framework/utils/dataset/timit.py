import os
import tqdm
import h5py
import torch
import numpy as np
import torchaudio
import python_speech_features as sf
from typing import Union


class TIMIT(torch.utils.data.Dataset):
    """
    References:
    [1] John S. Garofolo \emph{et al.}, DARPA} TIMIT acoustic-phonetic continuous speech corpus CD-ROM. NIST speech disc 1-1.1, 1993.
    [2] https://github.com/JoergFranke/phoneme_recognition
    [3] Bojian Yin \emph{et al.}, Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks.
    """

    data_dir_list = ["DR5", "DR6", "DR7", "DR3", "DR2", "DR1", "DR4", "DR8"]
    phonem_assignment = [
        "bcl", "dcl", "gcl", "pcl", "tck", "kcl", "dcl", "tcl", "b", "d", 
        "g", "p", "t", "k", "dx", "q", "jh", "ch", "s", "sh", "z", "zh", 
        "f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", 
        "r", "w", "y", "hh", "hv", "el", "iy", "ih", "eh", "ey", "ae", "aa", 
        "aw", "ay", "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er", "ax", 
        "ix", "axr", "ax-h", "pau", "epi", "h#", "1", "2",
    ]

    # Mel-Frequency Cepstrum Coefficients, default 12
    numcep = 12
    # The number of filters in the filterbank, default 26.
    numfilt = 26
    # The length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    winlen = 0.025
    # The step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    winstep = 0.01
    # Use first or first & second order derivation
    grad = 2

    def __init__(self, 
        root: str,    # Path to raw data
        subset: str,  # "train", "valid" or "test"
        saved_data_file                  = "preprocessed", 
        time_step                        = 100,
        seed                             = 0,
        device: Union[str, torch.device] = "cpu"
    ):
        assert subset in ("train", "valid", "test"), "The `subset` should be string \"train\", \"valid\" or \"test\"."
        self.root       = root
        self.subset     = subset
        self.seed       = seed
        self.timit_dict = self.get_timit_dict()

        os.makedirs(os.path.join(root, saved_data_file), exist_ok=True)
        preprocessed_data_root = os.path.join(root, saved_data_file, subset)

        scratch_data_root = os.path.join(root, saved_data_file, "scratch")

        if os.path.exists(preprocessed_data_root):
            # Load saved data
            print(f"The `saved_data_file` path exists, data preloading of `{self.__class__.__name__}` from path `{preprocessed_data_root}` start.")
            h5file = h5py.File(f"{preprocessed_data_root}/preprocessed_t{time_step}.h5", "r")
            input_iter = h5file["inputs"]
            label_iter = h5file["labels"]
            self.inputs = []
            self.labels = []
            for i in tqdm.tqdm(range(len(label_iter))):
                self.inputs.append(torch.tensor(input_iter[i], dtype=torch.float32).to(device))
                self.labels.append(torch.tensor(label_iter[i], dtype=torch.int64).to(device))
        else:
            # 1. Check or generate scratch data and save to the `scratch_data_root`
            if not os.path.exists(scratch_data_root):
                print(f"The `scratch_data_root` does not exist, `train`, `valid` and `test` scratch data generating of `{self.__class__.__name__}` start.")
                os.makedirs(scratch_data_root)
                for dir in ("train", "valid", "test"):
                    os.makedirs(os.path.join(scratch_data_root, dir))
    
                # 1.1. Generate `train` and `valid` scratch data
                train_speaker, valid_speaker = self.get_speaker_lists(os.path.join(self.root, "TRAIN"))
                i = 0
                for d in self.data_dir_list:
                    for dirName, subdirList, fileList in os.walk(os.path.join(self.root, "TRAIN", d)):
                        path, folder_name = os.path.split(dirName)
                        if folder_name.__len__() >= 1:
                            temp_name = ""
                            for fname in sorted(fileList):
                                name = fname.split(".")[0]
                                if name != temp_name:
                                    if "SI" in name or "SX" in name:
                                        temp_name = name
                                        wav_location = os.path.join(dirName, f"{name}.WAV")
                                        phn_location = os.path.join(dirName, f"{name}.PHN")
                                        feat, frames, samplerate = self.get_features(
                                            wav_location, 
                                            self.numcep, 
                                            self.numfilt, 
                                            self.winlen, 
                                            self.winstep, 
                                            self.grad
                                        )
                                        input_size = feat.shape[0]
                                        target = self.get_target(phn_location, self.timit_dict, input_size)
                                        if folder_name in train_speaker:
                                            np.save(f"{scratch_data_root}/train/{name}-{str(i)}.npy", np.hstack((feat, target)))
                                        elif folder_name in valid_speaker:
                                            np.save(f"{scratch_data_root}/valid/{name}-{str(i)}.npy", np.hstack((feat, target)))
                                        else:
                                            pass # print(f"unknown name: {folder_name}")
                                        i += 1
                                        feat = []
                                        target = []
                # 1.2. Generate `test` scratch data
                i = 0
                for d in self.data_dir_list:
                    for dirName, subdirList, fileList in os.walk(os.path.join(root, "TEST", d)):
                        path, folder_name = os.path.split(dirName)
                        if folder_name.__len__() >= 1:
                            temp_name = ""
                            for fname in sorted(fileList):
                                name = fname.split(".")[0]
                                if name != temp_name:
                                    if "SA" not in name:
                                        temp_name = name
                                        wav_location = os.path.join(dirName, f"{name}.WAV")
                                        phn_location = os.path.join(dirName, f"{name}.PHN")
                                        feat, frames, samplerate = self.get_features(
                                            wav_location, 
                                            self.numcep, 
                                            self.numfilt, 
                                            self.winlen, 
                                            self.winstep, 
                                            self.grad
                                        )
                                        input_size = feat.shape[0]
                                        target = self.get_target(phn_location, self.timit_dict, input_size)
                                        np.save(f"{scratch_data_root}/test/{name}-{str(i)}.npy", np.hstack((feat, target)))
                                        i += 1
                                        feat = []
                                        target = []

            # 2. Data preprocess
            subset_data = self.dataset_cutoff(os.path.join(scratch_data_root, subset), time_step)
            num_channels = 39
            Vmax = np.max(subset_data[:, :, :num_channels], axis=(0, 1))
            Vmin = np.min(subset_data[:, :, :num_channels], axis=(0, 1))
            
            subset_x = (subset_data[:, :, :num_channels] - Vmin) / (Vmax - Vmin)
            subset_y = subset_data[:, :, num_channels:]

            self.inputs = torch.Tensor(subset_x).to(device)
            self.labels = torch.Tensor(np.argmax(subset_y, axis=-1)).to(device)

            # 3. Save preprocessed data
            print(f"The preprocessed data saving to path `{preprocessed_data_root}` start.")
            os.makedirs(os.path.join(preprocessed_data_root))
            with h5py.File(f"{preprocessed_data_root}/preprocessed_t{time_step}.h5", "w") as fp:
                saved_inputs = fp.create_dataset("inputs", (len(self.inputs), *self.inputs[0].shape), dtype=np.float32)
                saved_labels = fp.create_dataset("labels", (len(self.labels), *self.labels[0].shape), dtype=np.int64)

                for i in tqdm.tqdm(range(len(self.labels))):
                    saved_inputs[i] = self.inputs[i].cpu()
                    saved_labels[i] = self.labels[i].cpu()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, n):
        return (self.inputs[n], self.labels[n])

    def get_timit_dict(self):
        # Read all phonemes (silences are all in line 61)
        phonemlist_length = self.phonem_assignment.__len__()
        max_phonem_length = 61 - 1 # -1 for array
        # Create key value dictionary
        dict_timit = {}
        for i in range(phonemlist_length):
            symbol = self.phonem_assignment[i]
            dict_timit[symbol] = min(i, max_phonem_length)
        return dict_timit

    def get_speaker_lists(self, root_dir):
        np.random.seed(self.seed)
        train_speaker = []
        valid_speaker = []
        for data_dir in self.data_dir_list:
            region_speakers = []
    
            for dirName, subdirList, fileList in os.walk(os.path.join(root_dir, data_dir)):
                path, folder_name = os.path.split(dirName)
                if folder_name.__len__() >= 1:
                    region_speakers.append(folder_name)
    
            len = region_speakers.__len__()
            valid_len = int(round(len * 0.1))
            random_valid = np.random.randint(0,len-1,valid_len)
            random_train = np.delete(np.arange(0,len),random_valid)
            region_speakers = np.asarray(region_speakers)
    
            train_speaker = train_speaker + list(region_speakers[random_train])
            valid_speaker = valid_speaker + list(region_speakers[random_valid])
    
        return train_speaker, valid_speaker

    def get_features(self, filename, numcep, numfilt, winlen, winstep, grad):
        waveform, samplerate = torchaudio.load(filename)
        frames = waveform.size(1)  # Number of frames
        data = waveform.numpy()
    
        # Calculate MFCC
        feat_raw, energy = sf.fbank(data, samplerate, winlen, winstep, nfilt=numfilt)
        feat = np.log(feat_raw)
        feat = sf.dct(feat, type=2, axis=1, norm="ortho")[:, :numcep]
        feat = sf.lifter(feat,L=22)
        feat = np.asarray(feat)
    
        # Calculate log energy
        log_energy = np.log(energy) #np.log( np.sum(feat_raw**2, axis=1) )
        log_energy = log_energy.reshape([log_energy.shape[0], 1])
    
        mat = (feat - np.mean(feat, axis=0)) / (0.5 * np.std(feat, axis=0))
        mat = np.concatenate((mat, log_energy), axis=1)
    
        # Calculate first order derivatives
        if grad >= 1:
            gradf = np.gradient(mat)[0]
            mat = np.concatenate((mat, gradf), axis=1)
    
        # Calculate second order derivatives
        if grad == 2:
            grad2f = np.gradient(gradf)[0]
            mat = np.concatenate((mat, grad2f), axis=1)
    
        return mat, frames, samplerate

    def get_target(self, phn_location, timit_dict, input_size):
        phn_file = open(phn_location, "r")
        phn_position = phn_file.readlines()
        phn_file.close()
    
        phn_position_length = phn_position.__len__() - 1
    
        target = np.empty([input_size,61])
    
        # Get first phonem
        phn_count = 0
        low_bound, hight_bound, symbol = phn_position[phn_count].split(" ")
        hight_bound = int(hight_bound)
        hight_bound_ms = hight_bound * 0.0625
        symbol = symbol.rstrip()
    
        # Go step by step through target vector and add phonem vector
        for i in range(input_size):
            threshold =  16 + i * 10
            if hight_bound_ms > threshold:
                tarray = np.zeros(61)
                tarray[timit_dict[symbol]] = 1
                target[i] = tarray
            else:
                # Get next phonem
                phn_count = min(phn_count + 1, phn_position_length)
                low_bound, hight_bound, symbol = phn_position[phn_count].split(" ")
                hight_bound = int(hight_bound)
                hight_bound_ms = hight_bound * 0.0625
                symbol = symbol.rstrip()
    
                tarray = np.zeros(61)
                tarray[timit_dict[symbol]] = 1
                target[i] = tarray
        return target

    def dataset_cutoff(self, dataset_dr, target_length=20):
        new_dataset = []
        data_files = os.listdir(dataset_dr)
        for f in data_files:
            data_ = np.load(os.path.join(dataset_dr, f))
            l = data_.shape[0]
            data_x = range(l)
            if l < target_length:
                data_idx = np.hstack((np.arange(l), np.ones(target_length - l) * (l - 1)))
                data_idx = [int(p) for p in data_idx]
                new_dataset.append(data_[data_idx, :])
            else: 
                N = int(l // target_length) + 1
                for i in range(N): 
                    if i == N - 1: 
                        data_idx = data_x[-target_length:]
                        new_dataset.append(data_[data_idx, :])
                    else: 
                        data_idx = data_x[i * target_length : (i + 1) * target_length]
                        new_dataset.append(data_[data_idx, :])
        return np.array(new_dataset)

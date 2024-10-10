import os
import h5py
import tqdm
import torch
import warnings
import numpy as np
from pathlib import Path
from torch import Tensor
from typing import Optional, Tuple, Union, Callable
from torchaudio.datasets import SPEECHCOMMANDS


# Default settings
FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.02"


class GoogleSpeechCommands(SPEECHCOMMANDS):
    def __init__(
        self,
        root: Union[str, Path],
        url: str                         = URL,
        folder_in_archive: str           = FOLDER_IN_ARCHIVE,
        download: bool                   = False,
        subset: Optional[str]            = None,
        saved_data_file                  = "preprocessed",
        preprocess: Callable             = None,
        device: Union[str, torch.device] = "cpu"
    ):
        super().__init__(root, url, folder_in_archive, download, subset)
        os.makedirs(os.path.join(root, folder_in_archive, saved_data_file), exist_ok=True)
        preprocessed_data_root = os.path.join(root, folder_in_archive, saved_data_file, subset)
        if os.path.exists(preprocessed_data_root): 
            # Data preloading
            print(f"The `saved_data_file` exists, data preloading of `{self.__class__.__name__}` from path `{preprocessed_data_root}` start.")
            h5file = h5py.File(f"{preprocessed_data_root}/gsc_preprocessed_data.h5", "r")
            waveform_iter          = h5file["waveforms"]
            sample_rate_iter       = h5file["sample_rates"]
            label_iter             = h5file["labels"]
            speaker_id_iter        = h5file["speaker_ids"]
            utterance_number_iter  = h5file["utterance_numbers"]
            self.waveforms         = []
            self.sample_rates      = []
            self.labels            = []
            self.speaker_ids       = []
            self.utterance_numbers = []
            for i in tqdm.tqdm(range(len(label_iter))):
                self.waveforms.append(torch.tensor(waveform_iter[i], dtype=torch.float32).to(device))
                self.sample_rates.append(torch.tensor(sample_rate_iter[i], dtype=torch.int64).to(device))
                self.labels.append(torch.tensor(label_iter[i], dtype=torch.int64).to(device))
                self.speaker_ids.append(torch.tensor(speaker_id_iter[i], dtype=torch.int64).to(device)) 
                self.utterance_numbers.append(torch.tensor(utterance_number_iter[i], dtype=torch.int64).to(device))
        else:
            # Data preprocessing
            self.preprocess = preprocess
            self.waveforms         = []
            self.sample_rates      = []
            self.labels            = []
            self.speaker_ids       = []
            self.utterance_numbers = []
            os.mkdir(preprocessed_data_root)
            print(f"The preprocessing of `{self.__class__.__name__}` start.")
            for i in tqdm.tqdm(range(super().__len__())):
                if self.preprocess is not None: 
                    (
                        data_waveform,
                        data_sample_rate,
                        data_label,
                        data_speaker_id,
                        data_utterence_number
                    ) = self.preprocess(*super().__getitem__(i))
                else: 
                    warnings.warn(f"The `preprocess` in class `{self.__class__.__name__}` is not recommended as None.")
                    (
                        data_waveform,
                        data_sample_rate,
                        data_label,
                        data_speaker_id,
                        data_utterence_number
                    ) = super().__getitem__(i)
                self.waveforms.append(data_waveform.to(device))
                self.sample_rates.append(data_sample_rate.to(device))
                self.labels.append(data_label.to(device))
                self.speaker_ids.append(data_speaker_id.to(device))
                self.utterance_numbers.append(data_utterence_number.to(device))

            # Data saving
            print(f"The saving to path `{preprocessed_data_root}` start.")
            with h5py.File(f"{preprocessed_data_root}/gsc_preprocessed_data.h5", "w") as fp:
                saved_waveforms         = fp.create_dataset("waveforms", (len(self.waveforms), *self.waveforms[0].shape), dtype=np.float32)
                saved_sample_rates      = fp.create_dataset("sample_rates", (len(self.sample_rates), *self.sample_rates[0].shape), dtype=np.int64)
                saved_labels            = fp.create_dataset("labels", (len(self.labels), *self.labels[0].shape), dtype=np.int64)
                saved_speaker_ids       = fp.create_dataset("speaker_ids", (len(self.speaker_ids), *self.speaker_ids[0].shape), dtype=np.int64)
                saved_utterence_numbers = fp.create_dataset("utterance_numbers", (len(self.utterance_numbers), *self.utterance_numbers[0].shape), dtype=np.int64)

                for i in tqdm.tqdm(range(len(self.labels))):
                    saved_waveforms[i]         = self.waveforms[i].cpu()
                    saved_sample_rates[i]      = self.sample_rates[i].cpu()
                    saved_labels[i]            = self.labels[i].cpu()
                    saved_speaker_ids[i]       = self.speaker_ids[i].cpu()
                    saved_utterence_numbers[i] = self.utterance_numbers[i].cpu()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int, int, int]:
        return (
            self.waveforms[n], 
            self.sample_rates[n], 
            self.labels[n], 
            self.speaker_ids[n], 
            self.utterance_numbers[n]
        )

"""
According to: Min-Ho Lee~\emph{et al.}, EEG dataset and OpenBMI toolbox for three BCI paradigms: An investigation into BCI illiteracy, 2019.
"""
import numpy as np
import scipy.io as scio
from random import shuffle


class OpenBMI:
    def __new__(cls,path, subject_ID, train_ratio):
        total_sample = 0
        data = []
        label = []
        raw_path = path + "/eeg_data/"
        for idx in subject_ID:
            sdn = "data_" + idx
            sdp = raw_path + sdn + ".mat"
            sln = "label_" + idx
            slp = raw_path + sln + ".mat"
            rate = 10
            # Read data
            _data = scio.loadmat(sdp)[sdn]
            _label = scio.loadmat(slp)[sln]
            # Downsampling
            s0, s1, s3 = np.shape(_data)  # [B, L, C]
            re_s1 = int(s1 / rate)
            downsample_data = _data.reshape(s0, re_s1, rate, s3).mean(2)
    
            for i in range(s3):
                # Normalize the data for each channel
                mue = np.mean(downsample_data[..., i])     # Mean
                sigma = np.std(downsample_data[..., i])    # Standard deviation
                downsample_data[..., i] = (downsample_data[..., i]-mue) / sigma
    
            data.append(downsample_data)
            label.append(_label)
    
            # Compute total sample number
            n_sample = np.size(downsample_data, 0)
            total_sample += n_sample
        datasets_list = list(range(0, total_sample))
        shuffle(datasets_list)
    
        data = np.concatenate(data)
        label = np.concatenate(label)
        print(data.shape)
    
        # Build data and label
        eeg_data = data
        _eeg_label = label
    
        eeg_label = np.array(range(0, len(_eeg_label)))
        for i in range(0, len(_eeg_label)):
            eeg_label[i] = _eeg_label[i]
    
        eeg_label = eeg_label - 1
        eeg_data = np.transpose(np.expand_dims(eeg_data, axis=1), (0, 1, 3, 2))
    
        print(eeg_data.shape,eeg_label.shape)
    
        xtrain = eeg_data[datasets_list][:int(train_ratio*total_sample),]
        ytrain = eeg_label[datasets_list][:int(train_ratio*total_sample),]
    
        xtest = eeg_data[datasets_list][int(train_ratio * total_sample):, ]
        ytest = eeg_label[datasets_list][int(train_ratio * total_sample):, ]
    
        np.save(path + "/x_train.npy", xtrain)
        np.save(path + "/x_test.npy", xtest)
        np.save(path + "/y_train.npy", ytrain)
        np.save(path + "/y_test.npy", ytest)
        print("\n.npy data[xtrain, xtest, ytrain, ytest] is saved to [%s]\n" % (path))
    
        return xtrain,ytrain,xtest,ytest

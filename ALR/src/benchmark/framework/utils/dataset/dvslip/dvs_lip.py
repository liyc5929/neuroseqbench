import glob
import json
import os
import torch
from torch.utils.data import Dataset
from .cvtransforms import *
import torchvision.transforms as trans
import logging
class Zoom:
    """ Randomly zoom-in or zoom-out the given sample (spatial data augmentation)
    Args:
        max_scale (int): max number of pixels to add (or remove) from the image outline
    """
    def __init__(self, max_scale):
        self.max_scale = max_scale

    def __call__(self, item):
        # item : (T, Cin, Y, X)
        p = np.random.random()
        if p < 0.5: ## zoom in:
            scale = np.random.randint(88 - self.max_scale, 88)
            zoom = trans.Compose([trans.CenterCrop(scale), trans.Resize(88)])
        else: ## zoom out:
            scale = np.random.randint(0, self.max_scale)
            zoom = trans.Compose([trans.Pad(scale), trans.Resize(88)])
        item = zoom(item)
        return item


class Cutout(object):
    ## from https://github.com/Intelligent-Computing-Lab-Yale/NDA_SNN/blob/d355bcb813c6e00c162a20ae7fd5c817b7014c02/functions/data_loaders.py
    """Randomly mask out one or more patches from an image (spatial data augmentation)
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, max_length):
        self.n_holes = n_holes
        self.max_length = max_length

    def __call__(self, img):
        h = img.size(2)
        w = img.size(3)
        for i in range(self.n_holes):
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)
            length = np.random.randint(1, self.max_length)
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask
        return img

# https://github.com/uzh-rpg/rpg_e2vid/blob/d0a7c005f460f2422f2a4bf605f70820ea7a1e5f/utils/inference_utils.py#L480
def events_to_voxel_grid_pytorch(events, num_bins, width, height, device):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """

    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)

    with torch.no_grad():

        events_torch = torch.from_numpy(events).float()
        events_torch = events_torch.to(device)

        voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device)
        if events_torch.shape[0] == 0:
            return voxel_grid

        voxel_grid = voxel_grid.flatten()

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events_torch[-1, 0]
        first_stamp = events_torch[0, 0]
        deltaT = float(last_stamp - first_stamp)

        if deltaT == 0:
            deltaT = 1.0

        events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
        ts = events_torch[:, 0]
        xs = events_torch[:, 1].long()
        ys = events_torch[:, 2].long()
        pols = events_torch[:, 3].float()
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = torch.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = pols * (1.0 - dts.float())
        vals_right = pols * dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0
        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices]
                                    * width + tis_long[valid_indices] * width * height,
                              source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices] * width
                                    + (tis_long[valid_indices] + 1) * width * height,
                              source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid


def events_to_voxel_all(events, frame_nums, seq_len, num_bins, width, height, device):
    voxel_len = min(seq_len, frame_nums) * num_bins
    voxel_grid_all = np.zeros((num_bins * seq_len, 1, height, width))
    voxel_grid = events_to_voxel_grid_pytorch(events, voxel_len, width, height, device)
    voxel_grid = voxel_grid.unsqueeze(1).cpu().numpy()
    voxel_grid_all[:voxel_len] = voxel_grid
    return voxel_grid_all




class DVSLipDataset(Dataset):
    def __init__(self, data_root, train=True, augment_spatial=False, augment_temporal=False, T=90, Tnbmask=6,
                 Tmaxmasklength=18):
        self.filenames = []
        self.labels = []
        self.samples = []
        self.augment_spatial = augment_spatial
        self.augment_temporal = augment_temporal
        self.train = train
        self.T = T
        training_words = get_training_words()
        label_dct = {k: i for i, k in enumerate(training_words)}

        if self.train:
            data_root = os.path.join(data_root, 'train')
            self.base_transform = trans.Compose(
                [trans.CenterCrop(96), trans.RandomCrop(88), trans.RandomHorizontalFlip(0.5)])
        else:
            data_root = os.path.join(data_root, 'test')
            self.base_transform = trans.CenterCrop(88)
        if self.augment_spatial:
            self.cutout = Cutout(n_holes=4, max_length=20)
            self.zoom = Zoom(max_scale=26)
        # if self.augment_temporal:
        #     # self.timemask = Masking(nb_mask=6, max_mask_length=self.T // 5)
        #     self.timemask = Masking(nb_mask=Tnbmask, max_mask_length=Tmaxmasklength)
        logging.info(f"data path: {data_root}")
        for root, dirs, files in os.walk(data_root):
            for filename in files:
                word = root.split("/")[-1]
                label = label_dct.get(word)
                if label is None:
                    print("ignored word: %s" % word)
                    break
                partial_path = '/'.join([word, filename])
                full_name = os.path.join(root, filename)
                self.filenames.append(full_name)
                self.labels.append(label)
                # save_path = full_name.replace('extract', 'processed').replace('.npy', '.npz')
                self.samples.append(np.load(full_name))
                # self.samples.append(np.load(save_path)['item'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # item = torch.from_numpy(self.samples[idx]).float()
        # item = self.base_transform(item)
        # if self.augment_spatial:
        #     item = self.cutout(item)
        #     item = self.zoom(item)
        # if self.augment_temporal:
        #     item = self.timemask(item)
        # label = self.labels[idx]
        # return item, label

        filename = self.filenames[idx]
        X = 128
        Y = 128
        # Set processed file path
        # save_path = filename.replace('extract', 'processed').replace('.npy', '.npz')

        # Check if preprocessed file exists
        # if os.path.exists(save_path):
        #     print(f"idx: {idx}")
        # item = torch.from_numpy(np.load(save_path)['item']).float()
        # DATA AUGMENTATION
        # item = self.base_transform(item)
        # if self.augment_spatial:
        #     item = self.cutout(item)
        #     item = self.zoom(item)
        # if self.augment_temporal:
        #     item = self.timemask(item)
        # label = self.labels[idx]
        # return item, label
        # else:
        #     raise NotImplementedError
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)


        # load sample
        # sample = np.array(np.load(filename).tolist())  # list of elements of size : (t, x, y, p)
        sample = np.array(self.samples[idx].tolist())  # list of elements of size : (t, x, y, p)


        # PRE-PROCESSING
        time_step = 4e4 / (self.T / 30)
        ts = (np.round(sample[:, 0] / time_step).astype(np.int))
        # print(f"idx: {idx}, max: {ts.max()}")
        # remove events >= T
        restrict_idx = (ts < self.T)
        ts = ts[restrict_idx]
        xs = sample[restrict_idx, 1]
        ys = sample[restrict_idx, 2]
        polarity = sample[restrict_idx, 3]
        # compute polarity (p=0 corresponds to polarity -1 and 1 corresponds to polarity 1)
        p_idx_neg = (polarity == 0)
        polarity[p_idx_neg] = -1

        # separate positive and negative events in 2 channels
        p_idx_pos = np.logical_not(p_idx_neg)
        ts_pos = ts[p_idx_pos]
        xs_pos = xs[p_idx_pos]
        ys_pos = ys[p_idx_pos]
        polarity_pos = polarity[p_idx_pos]
        ts_neg = ts[p_idx_neg]
        xs_neg = xs[p_idx_neg]
        ys_neg = ys[p_idx_neg]
        polarity_neg = polarity[p_idx_neg]
        coo_pos = [[] for i in range(3)]
        coo_pos[0].extend(ts_pos)
        coo_pos[1].extend(xs_pos)
        coo_pos[2].extend(ys_pos)
        i_pos = torch.LongTensor(coo_pos)
        v_pos = torch.FloatTensor(polarity_pos)
        item_pos = torch.sparse.FloatTensor(i_pos, v_pos, torch.Size([self.T, X, Y])).to_dense()
        coo_neg = [[] for i in range(3)]
        coo_neg[0].extend(ts_neg)
        coo_neg[1].extend(xs_neg)
        coo_neg[2].extend(ys_neg)
        i_neg = torch.LongTensor(coo_neg)
        v_neg = torch.FloatTensor(polarity_neg)
        item_neg = torch.sparse.FloatTensor(i_neg, v_neg, torch.Size([self.T, X, Y])).to_dense()
        item = torch.stack((item_pos, item_neg))
        item = item.transpose(0, 1)  # output: (T, Cin, X, Y)

        # to put in pytorch order height, width (for horizontalflip) - apply before transform
        item = item.permute(0, 1, 3, 2)  # output: (T, Cin, Y, X)

        # Save processed sample
        # np.savez_compressed(save_path, item=item.numpy())


        # DATA AUGMENTATION
        item = self.base_transform(item)
        if self.augment_spatial:
            item = self.cutout(item)
            item = self.zoom(item)
        if self.augment_temporal:
            item = self.timemask(item)

        label = self.labels[idx]
        return item, label


def get_training_words():
    classes = [
        "accused",
        "action",
        "allow",
        "allowed",
        "america",
        "american",
        "another",
        "around",
        "attacks",
        "banks",
        "become",
        "being",
        "benefit",
        "benefits",
        "between",
        "billion",
        "called",
        "capital",
        "challenge",
        "change",
        "chief",
        "couple",
        "court",
        "death",
        "described",
        "difference",
        "different",
        "during",
        "economic",
        "education",
        "election",
        "england",
        "evening",
        "everything",
        "exactly",
        "general",
        "germany",
        "giving",
        "ground",
        "happen",
        "happened",
        "having",
        "heavy",
        "house",
        "hundreds",
        "immigration",
        "judge",
        "labour",
        "leaders",
        "legal",
        "little",
        "london",
        "majority",
        "meeting",
        "military",
        "million",
        "minutes",
        "missing",
        "needs",
        "number",
        "numbers",
        "paying",
        "perhaps",
        "point",
        "potential",
        "press",
        "price",
        "question",
        "really",
        "right",
        "russia",
        "russian",
        "saying",
        "security",
        "several",
        "should",
        "significant",
        "spend",
        "spent",
        "started",
        "still",
        "support",
        "syria",
        "syrian",
        "taken",
        "taking",
        "terms",
        "these",
        "thing",
        "think",
        "times",
        "tomorrow",
        "under",
        "warning",
        "water",
        "welcome",
        "words",
        "worst",
        "years",
        "young",
    ]
    return classes


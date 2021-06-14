import torch
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import pandas as pd
import skvideo.io as vio
import skimage.io as imgio
from skimage.util import crop
from skimage.transform import rescale, resize
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class Load_Paths(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def load(self, phase):
        train_path = []
        for dirs in os.listdir(self.dataset_dir):
            file_dirs = os.path.join(self.dataset_dir, dirs)
            for f in os.listdir(file_dirs):
                full_path = os.path.join(file_dirs, f)
                train_path.append(full_path)

        df_csv = pd.DataFrame(data={"videos":train_path})
        train, validation = train_test_split(df_csv, test_size=0.2)
        if phase == 'train':
            return train
        elif phase == 'valid':
            return validation

class AlgoDataset(Dataset):
    def __init__(self, root_dir, phase, transform=None):

        self.root_dir = root_dir
        loadpaths = Load_Paths(self.root_dir)
        self.model_data = loadpaths.load(phase)
        self.transform = transform
        self.phase = phase

    def normalize(self, feats):
        mean = feats.mean()
        std = feats.std()
        normal_feats = (feats - mean)/std
        return normal_feats

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.model_data.iloc[idx, 0])
        video = vio.vread(video_path)
        normalized_video = self.normalize(video)
        sample = {'features':normalized_video}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        features, lbl = sample['features'], sample['lables']
        features = features.transpose((3, 0, 1, 2))
        features = torch.from_numpy(features)

        return features


# How to call the dataset
train_dataset = AlgoDataset(root_dir=root_path, phase='train', transform=transforms.Compose([ToTensor()]))
valid_dataset = AlgoDataset(root_dir=root_path, phase='valid', transform=transforms.Compose([ToTensor()]))
 
train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader =  torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

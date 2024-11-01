import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models.ts2tcc.augmentations import DataTransform

def fft_transform(train_X):
    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    
    train_X_fft = train_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X_fft.reshape(-1, train_X_fft.shape[-1]))
    train_X_fft = scaler.transform(train_X_fft.reshape(-1, train_X_fft.shape[-1])).reshape(train_X_fft.shape)
    return train_X_fft

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config):
        super(Load_Dataset, self).__init__()
        # self.training_mode = training_mode

        X_train = torch.from_numpy(dataset)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        print(X_train.shape)
        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train =X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
        else:
            self.x_data = X_train

        self.len = X_train.shape[0]
        self.aug1, self.aug2 = DataTransform(self.x_data, config)
        if isinstance(self.aug1, np.ndarray):
            self.aug1 = torch.from_numpy(self.aug1)
        if isinstance(self.aug2, np.ndarray):
            self.aug2 = torch.from_numpy(self.aug2)

    def __getitem__(self, index):
        return self.x_data[index],  self.aug1[index], self.aug2[index]

    def __len__(self):
        return self.len


class TwoViewloader(Dataset):
    """
    Return the dataitem and corresponding index
    The batch of the loader: A list
        - [B, L, 1] (For univariate time series)
        - [B]: The corresponding index in the train_set tensors

    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample_tem = self.data[0][index]
        sample_fre = self.data[1][index]


        return index, sample_tem, sample_fre

    def __len__(self):
        return len(self.data[0])



class ThreeViewloader(Dataset):
    """
    Return the dataitem and corresponding index
    The batch of the loader: A list
        - [B, L, 1] (For univariate time series)
        - [B]: The corresponding index in the train_set tensors

    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample_tem = self.data[0][index]
        sample_fre = self.data[1][index]
        sample_sea = self.data[2][index]


        return index, sample_tem, sample_fre,sample_sea

    def __len__(self):
        return len(self.data[0])







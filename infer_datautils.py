
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import torch


def compute_seasonal(train_X,decomp_period=40,decomp_mode='additive'):

    # 季节项
    train_X_sea = torch.permute(train_X,(0,2,1))
    sean_list = []
    for sample in train_X_sea :
        dims_list=[]
        for dims in sample:
            dims = dims.abs() + 0.00001
            result = seasonal_decompose(dims, model=decomp_mode, period=decomp_period)
            trend = pd.Series(result.seasonal)

            trend = trend.ffill().bfill()
            trend = trend.to_numpy()
            dims_list.append(trend)
        dims_list = np.stack(dims_list)
        sean_list.append(dims_list)

    sean_list = np.stack(sean_list)
    train_X_sea = torch.Tensor(sean_list)
    print(train_X_sea.shape)
    train_X_sea = torch.permute(train_X_sea,(0,2,1))
    scaler = StandardScaler()
    scaler.fit(train_X_sea.reshape(-1, train_X_sea.shape[-1]))
    train_X_sea = scaler.transform(train_X_sea.reshape(-1, train_X_sea.shape[-1])).reshape(train_X_sea.shape)

    return train_X_sea

def format_ts2sea(train_X,decomp_period=40,decomp_mode='additive'):
    
    if len(train_X.shape)==2:
        train_X = train_X.unsqueeze(1)

    train_X = torch.transpose(train_X,2,1)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)

    # 频域

    train_X = torch.from_numpy(train_X)
    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()

    train_X_fft = train_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X_fft.reshape(-1, train_X_fft.shape[-1]))
    train_X_fft = scaler.transform(train_X_fft.reshape(-1, train_X_fft.shape[-1])).reshape(train_X_fft.shape)

    train_X_sea = compute_seasonal(train_X,decomp_period,decomp_mode)

    return [train_X, train_X_fft,train_X_sea]


def format_ts2cot(train_X):

    if len(train_X.shape)==2:
        train_X = train_X.unsqueeze(1)

    train_X = torch.transpose(train_X,2,1)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)

    # 频域

    train_X = torch.from_numpy(train_X)
    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()

    train_X_fft = train_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X_fft.reshape(-1, train_X_fft.shape[-1]))
    train_X_fft = scaler.transform(train_X_fft.reshape(-1, train_X_fft.shape[-1])).reshape(train_X_fft.shape)

    return [train_X, train_X_fft]


def format_ts2vec(train_X):

    if len(train_X.shape)==2:
        train_X = train_X.unsqueeze(1)
    
    train_X = torch.transpose(train_X,2,1)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))

    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)

    return train_X
 


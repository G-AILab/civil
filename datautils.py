
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import torch

def load_base(data_path,mode="train"):
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    #val_ = torch.load(data_path + "val.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    #val = val_['samples']
    #val = torch.transpose(val, 1, 2)
    #val_labels = val_['labels']
    test = test_['samples']
    test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = train #torch.cat([train, val])
    train_y = train_labels #torch.cat([train_labels, val_labels])
        
    test_X = test
    test_y = test_labels

    return train_X,train_y,test_X,test_y


def compute_seasonal(dataset,train_X,test_X):
    decomp_period = 40
    if dataset=="ISRUC":
        decomp_period=decomp_period*6
    decomp_mode = 'multiplicative'
    # 季节项
    print("to compute seasonal of train_X...")
    print(train_X.shape)
    train_X_sea = torch.permute(train_X,(0,2,1))
    sean_list = []
    for sample in train_X_sea :
        dims_list=[]
        for dims in sample:
            dims = dims.abs() + 0.00001
            result = seasonal_decompose(dims, model=decomp_mode, period=decomp_period)
            trend = pd.Series(result.seasonal)
            if dataset=="ISRUC":
                trend=pd.Series(result.resid)
            trend = trend.ffill().bfill()
            trend = trend.to_numpy()
            dims_list.append(trend)
        dims_list = np.stack(dims_list)
        sean_list.append(dims_list)

    sean_list = np.stack(sean_list)
    train_X_sea = torch.Tensor(sean_list)
    print(train_X_sea.shape)
    train_X_sea = torch.permute(train_X_sea,(0,2,1))
    print(train_X_sea.shape)
    print("to compute seasonal of test_X...")
    print(test_X.shape)
    test_X_sea = torch.permute(test_X,(0,2,1))
    sean_list = []
    for sample in test_X_sea :
        dims_list=[]
        for dims in sample:
            dims = dims.abs() + 0.00001
            result = seasonal_decompose(dims, model=decomp_mode, period=decomp_period)
            trend = pd.Series(result.seasonal)
            if dataset=="ISRUC":
                trend=pd.Series(result.resid)
            trend = trend.ffill().bfill()
            trend = trend.to_numpy()
            dims_list.append(trend)
        dims_list = np.stack(dims_list)
        sean_list.append(dims_list)

    sean_list = np.stack(sean_list)
    test_X_sea = torch.Tensor(sean_list)
    print(test_X_sea.shape)
    test_X_sea = torch.permute(test_X_sea,(0,2,1))
    print(test_X_sea.shape)


    scaler = StandardScaler()
    scaler.fit(train_X_sea.reshape(-1, train_X_sea.shape[-1]))
    train_X_sea = scaler.transform(train_X_sea.reshape(-1, train_X_sea.shape[-1])).reshape(train_X_sea.shape)
    test_X_sea = scaler.transform(test_X_sea.reshape(-1, test_X_sea.shape[-1])).reshape(test_X_sea.shape)
    print(train_X.shape)
    print(train_X_sea.shape)

    return train_X_sea,test_X_sea

def load_itri_view(mode="train",dataset="ISRUC",decompose_mode=None):
    # train_X, train_y, test_X, test_y = load_HAR()
    # train_X_fft, _, test_X_fft, _ = load_HAR_fft()
    # train_X_sea, _, test_X_sea, _ = load_HAR_seasonal()
    print(mode)
    data_path = f"/workspace/CA-TCC/data/{dataset}/"
    print(data_path)
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    # val_ = torch.load(data_path + "val.pt")
    
    train_X = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_y = train_['labels']
    print(train_X.shape)
    train_X = torch.transpose(train_X,2,1)
    print(train_X.shape)

    test_ = torch.load(data_path + "test.pt")
    test = test_['samples']
    test_X = torch.transpose(test, 1, 2)
    test_y = test_['labels']

    
    print(test_X.shape)
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    # 频域
    print(type(train_X))
    print(train_X.shape)
    train_X = torch.from_numpy(train_X)
    test_X = torch.from_numpy(test_X)
    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    train_X_fft = train_X_fft.transpose(1, 2)
    test_X_fft = test_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X_fft.reshape(-1, train_X_fft.shape[-1]))
    train_X_fft = scaler.transform(train_X_fft.reshape(-1, train_X_fft.shape[-1])).reshape(train_X_fft.shape)
    test_X_fft = scaler.transform(test_X_fft.reshape(-1, test_X_fft.shape[-1])).reshape(test_X_fft.shape)


    if decompose_mode == "seasonal":
        if mode!="train":
            train_sea_ = torch.load(data_path + f"train_{mode}_sea.pt")
        else:
            train_sea_ = torch.load(data_path + "train_sea.pt")
        test_sea_ = torch.load(data_path + "test_sea.pt")
        train_X_sea = train_sea_['samples']
        test_X_sea = test_sea_['samples']
    elif decompose_mode == "trend":
        if mode!="train":
            train_trend_ = torch.load(data_path + f"train_{mode}_trend.pt")
        else:
            train_trend_ = torch.load(data_path + "train_trend.pt")
        test_trend_ = torch.load(data_path + "test_trend.pt")
        train_X_sea = train_trend_['samples']
        test_X_sea = test_trend_['samples']
    elif decompose_mode == "resid":
        if mode!="train":
            train_resid_ = torch.load(data_path + f"train_{mode}_resid.pt")
        else:
            train_resid_ = torch.load(data_path + "train_resid.pt")
        test_resid_ = torch.load(data_path + "test_resid.pt")
        train_X_sea = train_resid_['samples']
        test_X_sea = test_resid_['samples']
    elif decompose_mode == "generate":
        import os 
        if os.path.exists(f"{data_path}.train_seasonal_{mode}.pt"):            
            train_X_sea = torch.load(f"{data_path}.train_seasonal_{mode}.pt")
            test_X_sea = torch.load(f"{data_path}.test_seasonal_{mode}.pt")
        else:
            
            train_X_sea,test_X_sea = compute_seasonal(dataset,train_X,test_X)
            torch.save(train_X_sea,f"{data_path}.train_seasonal_{mode}.pt")
            torch.save(test_X_sea,f"{data_path}.test_seasonal_{mode}.pt")
    else:
        train_X_sea,test_X_sea = None,None
    print("train_X.shape",train_X.shape)
    print("train_X_fft.shape",train_X_fft.shape)
    print("train_X_sea.shape",train_X_sea.shape)
    return [train_X, train_X_fft,train_X_sea], train_y, [test_X, test_X_fft,test_X_sea], test_y


def load_itwo_view(mode="train",dataset="ISRUC"):
    # train_X, train_y, test_X, test_y = load_HAR()
    # train_X_fft, _, test_X_fft, _ = load_HAR_fft()
    # train_X_sea, _, test_X_sea, _ = load_HAR_seasonal()
    print(mode)
    data_path = f"/workspace/CA-TCC/data/{dataset}/"
    print(data_path)
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    # val_ = torch.load(data_path + "val.pt")
    
    train_X = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_y = train_['labels']
    print(train_X.shape,type(train_X))
    train_X = torch.transpose(train_X,2,1)
    print(train_X.shape)

    test_ = torch.load(data_path + "test.pt")
    test = test_['samples']
    test_X = torch.transpose(test, 1, 2)
    test_y = test_['labels']

    # test_X = test
    # test_y = test_labels


    
    # test_X = torch.permute(test_X,(0,2,1))
    
    print(test_X.shape)
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    # 频域
    print(type(train_X))
    print(train_X.shape)
    train_X = torch.from_numpy(train_X)
    test_X = torch.from_numpy(test_X)
    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    train_X_fft = train_X_fft.transpose(1, 2)
    test_X_fft = test_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X_fft.reshape(-1, train_X_fft.shape[-1]))
    train_X_fft = scaler.transform(train_X_fft.reshape(-1, train_X_fft.shape[-1])).reshape(train_X_fft.shape)
    test_X_fft = scaler.transform(test_X_fft.reshape(-1, test_X_fft.shape[-1])).reshape(test_X_fft.shape)



    print("train_X.shape",train_X.shape)
    print("train_X_fft.shape",train_X_fft.shape)
    return [train_X, train_X_fft], train_y, [test_X, test_X_fft], test_y



def load_roadbank_two_view(mode="train",dataset="RoadBank"):
    # train_X, train_y, test_X, test_y = load_HAR()
    # train_X_fft, _, test_X_fft, _ = load_HAR_fft()
    # train_X_sea, _, test_X_sea, _ = load_HAR_seasonal()
    data_path = f"/workspace/CA-TCC/data/{dataset}/"
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    # val_ = torch.load(data_path + "val.pt")
    
    train_X = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_y = train_['labels']

    test_ = torch.load(data_path + "test.pt")
    test = test_['samples']
    test_X = torch.transpose(test, 1, 2)
    test_y = test_['labels']

    # test_X = test
    # test_y = test_labels


    train_X = torch.transpose(train_X,2,1)
    # test_X = torch.permute(test_X,(0,2,1))
    print(train_X.shape)
    print(test_X.shape)
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    # 频域
    print(type(train_X))
    print(train_X.shape)
    train_X = torch.from_numpy(train_X)
    test_X = torch.from_numpy(test_X)
    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    train_X_fft = train_X_fft.transpose(1, 2)
    test_X_fft = test_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X_fft.reshape(-1, train_X_fft.shape[-1]))
    train_X_fft = scaler.transform(train_X_fft.reshape(-1, train_X_fft.shape[-1])).reshape(train_X_fft.shape)
    test_X_fft = scaler.transform(test_X_fft.reshape(-1, test_X_fft.shape[-1])).reshape(test_X_fft.shape)

    # 季节项
    print("to compute seasonal of train_X...")
    print(train_X.shape)
    train_X_sea = torch.permute(train_X,(0,2,1))
    sean_list = []
    for sample in train_X_sea :
        dims_list=[]
        for dims in sample:
            result = seasonal_decompose(dims, model='additive', period=30)
            dims_list.append(result.seasonal)
        dims_list = np.stack(dims_list)
        sean_list.append(dims_list)

    sean_list = np.stack(sean_list)
    train_X_sea = torch.Tensor(sean_list)
    print(train_X_sea.shape)
    train_X_sea = torch.permute(train_X_sea,(0,2,1))
    print(train_X_sea.shape)
    print("to compute seasonal of test_X...")
    print(test_X.shape)
    test_X_sea = torch.permute(test_X,(0,2,1))
    sean_list = []
    for sample in test_X_sea :
        dims_list=[]
        for dims in sample:
            result = seasonal_decompose(dims, model='additive', period=30)
            dims_list.append(result.seasonal)
        dims_list = np.stack(dims_list)
        sean_list.append(dims_list)

    sean_list = np.stack(sean_list)
    test_X_sea = torch.Tensor(sean_list)
    print(test_X_sea.shape)
    test_X_sea = torch.permute(test_X_sea,(0,2,1))
    print(test_X_sea.shape)


    scaler = StandardScaler()
    scaler.fit(train_X_sea.reshape(-1, train_X_sea.shape[-1]))
    train_X_sea = scaler.transform(train_X_sea.reshape(-1, train_X_sea.shape[-1])).reshape(train_X_sea.shape)
    test_X_sea = scaler.transform(test_X_sea.reshape(-1, test_X_sea.shape[-1])).reshape(test_X_sea.shape)
    print(train_X.shape)
    print(train_X_fft.shape)
    print(train_X_sea.shape)
    return [train_X, train_X_fft,train_X_sea], train_y, [test_X, test_X_fft,test_X_sea], test_y



def load_uea_two_view(mode="train",_type="UEA",dataset="SelfRegulationSCP1"):
    # train_X, train_y, test_X, test_y = load_HAR()
    # train_X_fft, _, test_X_fft, _ = load_HAR_fft()
    # train_X_sea, _, test_X_sea, _ = load_HAR_seasonal()
    data_path = f"/workspace/CA-TCC/data/{_type}/{dataset}/"
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    # val_ = torch.load(data_path + "val.pt")
    
    train_X = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_y = train_['labels']

    test_ = torch.load(data_path + "test.pt")
    test = test_['samples']
    test_X = torch.transpose(test, 1, 2)
    test_y = test_['labels']

    # test_X = test
    # test_y = test_labels


    train_X = torch.transpose(train_X,2,1)
    # test_X = torch.permute(test_X,(0,2,1))
    print(train_X.shape)
    print(test_X.shape)
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    # 频域
    print(type(train_X))
    print(train_X.shape)
    train_X = torch.from_numpy(train_X)
    test_X = torch.from_numpy(test_X)
    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    train_X_fft = train_X_fft.transpose(1, 2)
    test_X_fft = test_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X_fft.reshape(-1, train_X_fft.shape[-1]))
    train_X_fft = scaler.transform(train_X_fft.reshape(-1, train_X_fft.shape[-1])).reshape(train_X_fft.shape)
    test_X_fft = scaler.transform(test_X_fft.reshape(-1, test_X_fft.shape[-1])).reshape(test_X_fft.shape)

    # 季节项
    print("to compute seasonal of train_X...")
    print(train_X.shape)
    train_X_sea = torch.permute(train_X,(0,2,1))
    sean_list = []
    for sample in train_X_sea :
        dims_list=[]
        for dims in sample:
            result = seasonal_decompose(dims, model='additive', period=30)
            dims_list.append(result.seasonal)
        dims_list = np.stack(dims_list)
        sean_list.append(dims_list)

    sean_list = np.stack(sean_list)
    train_X_sea = torch.Tensor(sean_list)
    print(train_X_sea.shape)
    train_X_sea = torch.permute(train_X_sea,(0,2,1))
    print(train_X_sea.shape)
    print("to compute seasonal of test_X...")
    print(test_X.shape)
    test_X_sea = torch.permute(test_X,(0,2,1))
    sean_list = []
    for sample in test_X_sea :
        dims_list=[]
        for dims in sample:
            result = seasonal_decompose(dims, model='additive', period=30)
            dims_list.append(result.seasonal)
        dims_list = np.stack(dims_list)
        sean_list.append(dims_list)

    sean_list = np.stack(sean_list)
    test_X_sea = torch.Tensor(sean_list)
    print(test_X_sea.shape)
    test_X_sea = torch.permute(test_X_sea,(0,2,1))
    print(test_X_sea.shape)


    scaler = StandardScaler()
    scaler.fit(train_X_sea.reshape(-1, train_X_sea.shape[-1]))
    train_X_sea = scaler.transform(train_X_sea.reshape(-1, train_X_sea.shape[-1])).reshape(train_X_sea.shape)
    test_X_sea = scaler.transform(test_X_sea.reshape(-1, test_X_sea.shape[-1])).reshape(test_X_sea.shape)
    print(train_X.shape)
    print(train_X_fft.shape)
    print(train_X_sea.shape)
    return [train_X, train_X_fft,train_X_sea], train_y, [test_X, test_X_fft,test_X_sea], test_y



def get_data_loader(args):
    if args.dataloader in ["general_cls",'FordA','Bridge','ECG','SleepEDF','Gesture','EMG','SleepEEG','FD-A','FD-B']:
        train_data, train_labels, test_data, test_labels = load_itri_view(args.data_perc,args.dataset,args.decomp_mode)

    if args.dataloader == 'RoadBank' or args.dataloader == "Bridge":
        train_data, train_labels, test_data, test_labels = load_roadbank_two_view(args.data_perc,args.dataset)
    
    if args.dataloader == 'UEA':
        train_data, train_labels, test_data, test_labels = load_uea_two_view(args.data_perc,"UEA",args.dataset)
    
    if args.dataloader == 'UCR':
        train_data, train_labels, test_data, test_labels = load_uea_two_view(args.data_perc,"UCR",args.dataset)
    
    return train_data, train_labels, test_data, test_labels


def get_idata_loader(args):
    # if args.dataloader in ["HAR","Epilepsy","ISRUC",'FordA','Bridge','ECG','SleepEDF','Gesture']:
    train_data, train_labels, test_data, test_labels = load_itwo_view(args.data_perc,args.dataset)
    
    return train_data, train_labels, test_data, test_labels



def get_ts2vec_loader(args):
    dataset_root = "/workspace/CA-TCC/data"

    dataset = args.dataset
    mode = args.data_perc
    
    print(mode)
    data_path = f"{dataset_root}/{dataset}/"
    print(data_path)
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    # val_ = torch.load(data_path + "val.pt")
    
    train_X = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_y = train_['labels']
    if len(train_X.shape)==2:
        train_X = train_X.unsqueeze(1)
    train_X = torch.transpose(train_X,2,1)
    #train_X = train_X[:, ::3, :]
    print(train_X.shape)

    test_ = torch.load(data_path + "test.pt")
    test = test_['samples']
    if len(test.shape)==2:
        test = test.unsqueeze(1)
    test_X = torch.transpose(test, 1, 2)
    #test_X = test_X[:, ::3, :]
    test_y = test_['labels']

    
    print(test_X.shape)
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)

    print("train_X.shape",train_X.shape)
    print("test_X.shape",test_X.shape)

    return train_X, train_y, test_X, test_y
 


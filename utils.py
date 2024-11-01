import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime

import faiss
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys



def save_pkl(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def init_cuda(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]



def run_kmeans(x, args, last_clusters = None):
    results = {'im2cluster': [], 'centroids': [], 'density': [], 'distance': [], 'distance_2_center': []}

    if not type(x)==np.ndarray:
        x = x.reshape(x.shape[0], -1).numpy()
    if len(x.shape) == 3:
        x = x.reshape(x.shape[0], -1)
    x = x.astype(np.float32)

    cluster_id = 0

    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)

        clus = faiss.Clustering(d, k)

        # clus.verbose = True
        clus.niter = 20
        clus.nredo = 1
        # clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        if last_clusters is not None:
            cen = (last_clusters['centroids'][cluster_id].cpu().numpy()).astype(np.float32)
            cen2 = faiss.FloatVector()
            faiss.copy_array_to_vector(cen.reshape(-1), cen2)
            clus.centroids = cen2
        #res = faiss.StandardGpuResources()
        #cfg = faiss.GpuIndexFlatConfig()
        #cfg.useFloat16 = False
        #cfg.device = args.gpu
        #cfg.verbose = True
        #print("index")
        index = faiss.IndexFlatL2(d)
        #print("index2")
        clus.train(x, index)
        D, I = index.search(x, k)
        im2cluster = [int(n[0]) for n in I]
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)
        #print("centroids")
        Dcluster = [[] for c in range(k)]
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10),
                               np.percentile(density, 90))  # clamp extreme values for stability
        density = args.temperature * density / density.mean()  # scale the mean to temperature


        centroids = torch.Tensor(centroids).cuda()
        xx_norm = torch.nn.functional.normalize(torch.tensor(x).cuda(), p=2, dim=1)
        dist = (xx_norm.unsqueeze(-1).repeat((1,1,k))- centroids.t().unsqueeze(0).repeat((x.shape[0],1,1)))**2
        dist = torch.sum(dist, 1)
        dist = torch.nn.functional.softmax(-dist, 1)

        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        im2cluster = torch.LongTensor(im2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)
        results['distance'].append(dist)
        results['distance_2_center'].append(D)

        cluster_id += 1

    return results


def prototype_loss_cotrain(out, index, cluster_result=None, args=None, crop_offset=None, crop_eleft=None, crop_right=None, crop_l=None):
    criterion = nn.CrossEntropyLoss().cuda()
    if len(out.shape) == 2:
        out = out.unsqueeze(-1)
    out = out.permute(0, 2, 1)
    if cluster_result is not None:
        proto_labels = []
        proto_logits = []
        for n, (im2cluster, prototypes, density, pro) in enumerate(
                zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'], cluster_result['ma_centroids'])):

            prototypes = torch.unsqueeze(prototypes, 0)
            prototypes = prototypes.repeat(out.shape[0], 1, 1)
            prototypes = prototypes.permute(0, 2, 1)
            prototypes /= density


            try:
                pos_proto_id = im2cluster[index]
                retain_index = torch.where(pos_proto_id >= 0)
                pos_proto_id = pos_proto_id[retain_index]
                out2 = out[retain_index ]
                prototypes2 = prototypes[retain_index]
            except:
                import pdb

            logits_proto_instance = torch.matmul(out2, prototypes2).squeeze(1)
            proto_loss_instance = criterion(logits_proto_instance, pos_proto_id)

            loss_proto = proto_loss_instance
            for cl in range(pro.shape[0]):
                if (pos_proto_id == cl).sum() > 0:
                    pro[cl, :] = args.ma_gamma * pro[cl, :] + (1-args.ma_gamma) * out2.detach()[(pos_proto_id == cl), ...].mean(0).squeeze(0)
                else:
                    pro[cl, :] = pro[cl, :]
            cluster_result['ma_centroids'][n] = pro


        return loss_proto, cluster_result['ma_centroids']
    else:
        return  None, None


def get_logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger



#####################################
### ts2vec
#####################################


def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr
    
def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]

def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B*T,
        size=int(B*T*p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res

def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '/' + now.strftime("%Y%m%d_%H%M%S")

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]


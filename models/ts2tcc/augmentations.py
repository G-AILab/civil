import numpy as np
import torch


def DataTransform(sample, config):
    weak_aug = scaling(sample, sigma=config.aug_jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.aug_max_seg), config.aug_jitter_ratio)
    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    x = x.cpu().numpy()
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutationbk(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    x = x.cpu().numpy()
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
                print(split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            _tmp_ = np.random.permutation(splits)
            warp = np.concatenate(_tmp_).ravel()
            print(warp.shape)
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
        print(ret[i].shape)
    return torch.from_numpy(ret)

def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    x = x.cpu().numpy()
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                
                split_points.sort()
                indices = split_points!=0
                split_points = split_points[indices]
                splits = np.split(orig_steps, split_points)
                segs_len =len(splits)
                segs_arr = np.arange(segs_len)
                segs_arr = np.random.permutation(segs_arr)
                idx_arr = []
                for seg in segs_arr:
                    idx_arr.extend(splits[seg])
                warp =np.array(idx_arr)
            else:
                warp = orig_steps

            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)
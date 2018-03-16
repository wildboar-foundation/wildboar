import numpy as np
from numpy.lib.stride_tricks import as_strided


def z_norm(x, axis=1):
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    return (x - mean) / std


def euclidean(a, b):
    return np.sqrt(np.sum(np.power(a - b, 2), axis=1))


def sliding_distance(s, t, distance=euclidean):
    no_shapelets = t.shape[0] - s.shape[0] + 1
    t = as_strided(t, shape=(no_shapelets, s.shape[0]), strides=(8, 8))
    t = z_norm(t)
    d = distance(s, t)
    arg = np.argmin(d)
    return arg, d[arg] 

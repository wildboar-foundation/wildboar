import numpy as np

def z_norm(x, axis=1):
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    return (x - mean) / std




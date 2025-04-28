import numpy as np

def rolling_average(arr, window=8):
    # 3D rolling average
    shape = (arr.shape[0] - window + 1, window) + arr.shape[1:]
    strides = (arr.strides[0],) + arr.strides
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    return windows.mean(axis=1)
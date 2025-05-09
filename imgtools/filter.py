import numpy as np
from scipy import ndimage

def rolling_average(arr, window=8):
    # 3D rolling average
    shape = (arr.shape[0] - window + 1, window) + arr.shape[1:]
    strides = (arr.strides[0],) + arr.strides
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    return windows.mean(axis=1)


def medfilt(image, win=3):
    if image.ndim != 3:
        raise ValueError("Input must be a 3D array.")
    if win % 2 == 0:
        raise ValueError("Window size must be an odd integer.")
    
    pad_width = win // 2
    padded = np.pad(image, pad_width, mode='edge')
    output = np.empty_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                region = padded[i:i+win, j:j+win, k:k+win]
                output[i, j, k] = np.median(region)

    return output


def gaussian_kernel_3d(size, sigma):
    ax = np.arange(-size//2 + 1, size//2 + 1)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing='ij')
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


def gaussfilt(image, win=3, sigma=1):
    if image.ndim != 3:
        raise ValueError("Input must be a 3D array.")
    if win % 2 == 0:
        raise ValueError("Kernel size must be an odd integer.")

    kernel = gaussian_kernel_3d(win, sigma)
    pad_width = win // 2
    padded = np.pad(image, pad_width, mode='reflect')
    output = np.empty_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                region = padded[i:i+win, j:j+win, k:k+win]
                output[i, j, k] = np.sum(region * kernel)

    return output


def high_pass_filter_3d(image, sigma):
    if image.ndim != 3:
        raise ValueError("Input must be a 3D array.")
    blurred_data = ndimage.gaussian_filter(image, sigma=sigma)
    filtered_data = image - blurred_data
    return filtered_data


def high_pass_filter_2d(image, sigma, axis=0):
    if image.ndim != 3:
        raise ValueError("Input must be a 3D array.")
    filtered_data = np.zeros_like(image)
    for f in range(np.size(image,axis)):
        blurred_data = ndimage.gaussian_filter(image[f,:,:], sigma=sigma)
        filtered_data[f] = image[f,:,:] - blurred_data
    return filtered_data
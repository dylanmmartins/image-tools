# -*- coding: utf-8 -*-
"""
Filtering functions for multi-dimensional arrays.

Functions
---------
rolling_average(arr, window=8)
    Compute the rolling average of a 3D array.
rolling_average_1d(data, window_size)
    Compute the rolling average of a 1D array.
medfilt(image, win=3)
    Apply a median filter to a 3D image.
gaussian_kernel_3d(size, sigma)
    Create a 3D Gaussian kernel.
gaussfilt(image, win=3, sigma=1)
    Apply a Gaussian filter to a 3D image.
high_pass_filter_3d(image, sigma)
    Apply a high-pass filter to a 3D image.
high_pass_filter_2d_along_axis(image, sigma, axis=0)
    Apply a high-pass filter to a 2D signal.
high_pass_filter_2d_single_frame(image, sigma)
    Apply a high-pass filter to a single 2D image.
sub2ind(array_shape, rows, cols)
    Convert subscripts to linear indices.
nanmedfilt(A, sz=5)
    Median filtering of 1D or 2D array while ignoring NaNs.
boxcar_smooth(data, window_size)
    Apply a boxcar smoothing filter to a 1D array.

Author: DMM, 2025
"""


import numpy as np
from scipy import ndimage


def rolling_average(arr, window=8, ensure_shape_match=False):
    """ Compute the rolling average of a 3D array.
    
    Parameters
    ----------
    arr : np.ndarray
        Input 3D array.
    window : int
        Size of the rolling window. Default is 8.

    Returns
    -------
    rolled_array : np.ndarray
        3D array with the rolling average applied.
    """

    shape = (arr.shape[0] - window + 1, window) + arr.shape[1:]
    strides = (arr.strides[0],) + arr.strides
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    rolled_array = windows.mean(axis=1)

    if ensure_shape_match:
        band = np.zeros([
            window,
            np.size(rolled_array,1),
            np.size(rolled_array,2)
        ])
        return np.concatenate([band, rolled_array, band], axis=0)

    return rolled_array


def rolling_average_1d(data, window_size):
    """ Compute the rolling average of a 1D array.
    
    Parameters
    ----------
    data : np.ndarray
        Input 1D array.
    window_size : int
        Size of the rolling window. Must be at least 1.

    Returns
    -------
    result : np.ndarray
        1D array with the rolling average applied.
    """


    if window_size < 1:
        raise ValueError("Window size must be at least 1")

    data = np.asarray(data)
    result = np.empty_like(data, dtype=float)

    half_w = window_size // 2

    for i in range(len(data)):
        start = max(0, i - half_w)
        end = min(len(data), i + half_w + 1)
        result[i] = np.mean(data[start:end])

    return result


def medfilt(image, win=3):
    """ Apply a median filter to a 3D image.

    Parameters
    ----------
    image : np.ndarray
        Input 3D image.
    win : int
        Size of the median filter window. Must be an odd integer.
    
    Returns
    -------
    output : np.ndarray
        3D image with the median filter applied.
    """

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
    """ Create a 3D Gaussian kernel.

    Parameters
    ----------
    size : int
        Size of the kernel. Must be an odd integer.
    sigma : float
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    kernel : np.ndarray
        3D Gaussian kernel.
    """

    ax = np.arange(-size//2 + 1, size//2 + 1)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing='ij')
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)

    return kernel


def gaussfilt(image, win=3, sigma=1):
    """ Apply a Gaussian filter to a 3D image.

    Parameters
    ----------
    image : np.ndarray
        Input 3D image.
    win : int
        Size of the Gaussian kernel. Must be an odd integer.
    sigma : float
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    output : np.ndarray
        3D image with the Gaussian filter applied.
    """

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
    """ Apply a high-pass filter to a 3D image.
    
    Parameters
    ----------
    image : np.ndarray
        Input 3D image.
    sigma : float
        Standard deviation of the Gaussian distribution used for blurring.

    Returns
    -------
    filtered_data : np.ndarray
        3D image with the high-pass filter applied.
    """

    if image.ndim != 3:
        raise ValueError("Input must be a 3D array.")
    
    blurred_data = ndimage.gaussian_filter(image, sigma=sigma)
    filtered_data = image - blurred_data

    return filtered_data


def high_pass_filter_2d_along_axis(image, sigma, axis=0):
    """ Apply a high-pass filter to a 2D signal.

    Parameters
    ----------
    image : np.ndarray
        Input 3D image.
    sigma : float
        Standard deviation of the Gaussian distribution used for blurring.
    axis : int
        Axis along which to apply the filter. Default is 0.

    Returns
    -------
    filtered_data : np.ndarray
        3D image with the high-pass filter applied along the specified axis.
    """

    if image.ndim != 3:
        raise ValueError("Input must be a 3D array.")
    
    filtered_data = np.zeros_like(image)
    
    for f in range(np.size(image,axis)):
        blurred_data = ndimage.gaussian_filter(image[f,:,:], sigma=sigma)
        filtered_data[f] = image[f,:,:] - blurred_data

    return filtered_data


def high_pass_filter_2d_single_frame(image, sigma):
    """ Apply a high-pass filter to a single 2D image.

    Parameters
    ----------
    image : np.ndarray
        Input 2D image.
    sigma : float
        Standard deviation of the Gaussian distribution used for blurring.
    
    Returns
    -------
    filtered_data : np.ndarray
        2D image with the high-pass filter applied.
    """

    blurred_data = ndimage.gaussian_filter(image, sigma=sigma)
    filtered_data = image - blurred_data

    return filtered_data


def sub2ind(array_shape, rows, cols):
    """ Convert subscripts to linear indices.

    Equivalent to Matlab's sub2ind function
    https://www.mathworks.com/help/matlab/ref/sub2ind.html

    Parameters
    ----------
    array_shape : tuple
        Shape of the array.
    rows : np.array
        Row subscripts.
    cols : np.array
        Column subscripts.
    
    Returns
    -------
    ind : np.array
        Multidimensional subscripts.
    """

    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1

    return ind


def nanmedfilt(A, sz=5):
    """ Median filtering of 1D or 2D array while ignoring NaNs.

    Parameters
    ----------
    A : np.array
        1D or 2D array.
    sz : int
        Kernel size for median filter. Must be an odd integer.

    Returns
    -------
    M : np.array
        Array matching shape of input, A, with median filter applied.

    Notes
    -----
    Adapted from the Matlab function
    https://www.mathworks.com/matlabcentral/fileexchange/41457-nanmedfilt2
    """

    if type(sz) == int:
        sz = np.array([sz, sz])

    if any(sz % 2 == 0):
        print('kernel size must be odd')

    margin = np.array((sz-1) // 2)

    if len(np.shape(A)) == 1:
        A = np.expand_dims(A, axis=0)

    AA = np.zeros(np.squeeze(np.array(np.shape(A)) + 2*np.expand_dims(margin, 0)))
    AA[:] = np.nan
    AA[margin[0]:-margin[0], margin[1]:-margin[1]] = A

    iB, jB = np.mgrid[0:sz[0], 0:sz[1]]
    isB = sub2ind(np.shape(AA.T), jB, iB) + 1

    iA, jA = np.mgrid[0:np.size(A,0), 0:np.size(A,1)]
    iA += 1
    isA = sub2ind(np.shape(AA.T), jA, iA)

    idx = isA + np.expand_dims(isB.flatten('F')-1, 1)
    
    B = np.sort(AA.T.flatten()[idx-1], 0)
    j = np.any(np.isnan(B), 0)

    last = np.zeros([1, np.size(B,1)]) + np.size(B,0)
    last[:, j] = np.argmax(np.isnan(B[:, j]),0)
    
    M = np.zeros([1, np.size(B,1)])
    M[:] = np.nan

    valid = np.where(last > 0)[1]
    mid = (1 + last) / 2

    i1 = np.floor(mid[:, valid])
    i2 = np.ceil(mid[:, valid])
    i1 = sub2ind(np.shape(B.T), valid, i1)
    i2 = sub2ind(np.shape(B.T), valid, i2)

    M[:,valid] = 0.5*(B.flatten('F')[i1.astype(int)-1] + B.flatten('F')[i2.astype(int)-1])

    M = np.reshape(M, np.shape(A))

    return M


def boxcar_smooth(data, window_size):
    """ Apply a boxcar smoothing filter to a 1D array.

    Parameters
    ----------
    data : np.ndarray
        Input 1D array.
    window_size : int
        Size of the smoothing window. Must be at least 1 and less than
        or equal to the length of the data.

    Returns
    -------
    smoothed : np.ndarray
        Smoothed 1D array.
    """

    if window_size < 1:
        raise ValueError("Window size must be at least 1")
    if window_size > len(data):
        raise ValueError("Window size cannot be greater than data length")
    
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data, kernel, mode='same')

    return smoothed



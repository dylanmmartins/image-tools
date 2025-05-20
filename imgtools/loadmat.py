"""
Load .mat files and convert them to Python dictionaries.

This is preferable to using scipy.io.loadmat directly.

Based on:
https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries

Author: DMM, 2024
"""


import scipy.io
import numpy as np


def _check_keys(d):
    """ Check if dict entries in dictionary are matobjects.

    Parameters
    ----------
    d : dict
        The dictionary to be checked for matobjects.
    
    Returns
    -------
    d : dict
        The dictionary with matobjects converted to dictionaries.
    """

    for key in d:
        if isinstance(d[key], scipy.io.matlab.mio5_params.mat_struct):
            d[key] = _todict(d[key])

    return d


def _todict(matobj):
    """ Recursive function to construct nested dictionaries from matobjects.
    
    Parameters
    ----------
    matobj : scipy.io.matlab.mio5_params.mat_struct
        The matobject to be converted to a dictionary.
    
    Returns
    -------
    d : dict
        A dictionary representation of the matobject.
    """

    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            d[strg] = _todict(elem)
        elif isinstance(elem, np.ndarray):
            d[strg] = _tolist(elem)
        else:
            d[strg] = elem

    return d


def _tolist(ndarray):
    """Construct lists from cellarrays (loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.

    Parameters
    ----------
    ndarray : np.ndarray
        The numpy ndarray to be converted to a list.

    Returns
    -------
    elem_list : list
        A list representation of the ndarray.
    """

    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, scipy.io.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem, np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)

    return elem_list


def loadmat(filename):
    """ Load a .mat file and convert it to a Python dictionary.

    Parameters
    ----------
    filename : str
        The path to the .mat file to be loaded.
        
    Returns
    -------
    d : dict
        A dictionary representation of the .mat file.
    """

    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)

    return _check_keys(data)


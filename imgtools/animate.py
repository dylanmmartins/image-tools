# -*- coding: utf-8 -*-
"""
Write animations from a series of matplotlib figures.

Functions
---------
fmt_figure(fig)
    Format a matplotlib figure for saving as an image.
make_frames(plot_fn, *args)
    Make a series of frames from a plotting function.
write_animation(img_stack, savepath, fps)
    Write a series of images to an mp4 file.

Author: DMM, April 2024
"""


import cv2
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import multiprocessing
import matplotlib.pyplot as plt


def fmt_figure(fig):
    """ Format a matplotlib figure for saving as an image.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to format.

    Returns
    -------
    img : np.ndarray
        The formatted image as a numpy array.
    """

    width, height = fig.get_size_inches() * fig.get_dpi()

    fig.canvas.draw()

    img = np.frombuffer(
        fig.canvas.tostring_rgb(),
        dtype=np.uint8
    ).reshape(int(height), int(width), 3)
    plt.close()

    return img


def make_frames(plot_fn, *args):
    """ Make a series of frames from a plotting function.

    Parameters
    ----------
    plot_fn : function
        The function to call to create each frame.
    *args : list
        The arguments to pass to the plotting function. Each argument must
        have the same size in axis=0.
    """

    mpl.use('agg')

    n_proc = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_proc)

    print('Parallel pool started with {} processes.'.format(n_proc))

    if type(args[0]) == list:
        numF = len(args[0])
    elif type(args[0]) == np.ndarray:
        numF = np.size(args[0], axis=0)

    print('Prepping parallel instructions.')
    mp_params = []
    for f in range(numF):
        mp_params.append(pool.apply_async(plot_fn, args=[arg[f] for arg in args]))

    # Execute the mp pool
    print('Plotting all frames.    (slow)')
    mp_out = [result.get() for result in tqdm(mp_params)]

    fr_image_stack = np.stack([
        mp_out[i] for i in range(len(mp_out))
    ])

    pool.close()

    return fr_image_stack


def write_animation(img_stack, savepath, fps):
    """ Write a series of images to an mp4 file.

    Parameters
    ----------
    img_stack : np.ndarray
        The image stack to write to the video file.
    savepath : str
        The path to save the video file.
    fps : int
        The frames per second for the video.
    """

    out = cv2.VideoWriter(
        savepath,
        cv2.VideoWriter_fourcc(*'mp4v'),
        int(fps),
        (img_stack.shape[-2], img_stack.shape[-3])
    )

    print('Writing MP4 file.')
    for fr in tqdm(range(np.size(img_stack, axis=0))):
        out.write(cv2.cvtColor(
            img_stack[fr],
            cv2.COLOR_BGR2RGB
        ))
    
    out.release()

    print('Video written to {}'.format(savepath))


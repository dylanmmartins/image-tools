# -*- coding: utf-8 -*-
"""
Write animated video for 2P data.

Functions
---------
animate_stack_from_2P(imgpath=None, savename=None)
    Animate a full tiff stack of two-photon data.

Author: DMM, 2024
"""


import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import imgtools


def animate_stack_from_2P(imgpath=None, savename=None, num_frames=3600):
    """ Animate a full tiff stack of two-photon data.

    Parameters
    ----------
    imgpath : str, optional
        Path to the tiff stack. If None, a default path is used.
    savename : str, optional
        Name of the output video file. If None, a default name is used.
    """

    if imgpath is None:
        imgpath = r'C:\Data\dylan\LPaxons\recordings_for_demo\250426_DMM_DMM046_LPaxons_f05.tif'
    if savename is None:
        savename = 'animation_01.mp4'

    savedir_ = os.path.split(os.path.split(imgpath)[0])[0]
    savepath = os.path.join(savedir_, savename)

    # Animate a full tiff stack of two-photon data.
    img_rollavg = imgtools.rolling_average(
        imgtools.read_tif_until(
            imgpath, num_frames
        ),
    window=8)
    # img_rollavg = img_rollavg[:num_frames,:,:]

    plot_stack = []
    for f in tqdm(range(np.size(img_rollavg,0))):

        fig = plt.figure(dpi=300, figsize=(4,4))
        # vmax can be 2000 for benchtop, 300 for mini2p
        plt.imshow(img_rollavg[f,:,:], cmap='gray', vmin=0, vmax=850)
        plt.axis('off')
        plt.tight_layout()
        plot_stack.append(imgtools.fmt_figure(fig))
    
    plot_stack = np.array(plot_stack)

    imgtools.write_animation(
        plot_stack,
        savepath,
        30 # 60 == 8x real-time for 7.5 Hz data
    )


if __name__ == '__main__':

    animate_stack_from_2P(
        r'K:\Mini2P_V1PPC\251029_DMM_DMM061_pos03\sn2_demo\file_00003.tif'
    )

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import imgtools

def animate_stack_from_2P(imgpath=None, savename=None):

    if imgpath is None:
        imgpath = r'T:\axonal_imaging_LP\250426_DMM_DMM046_LPaxons\fm5\file_00005.tif'
    if savename is None:
        savename = 'f05_demo_vid_60Hz_win5_v01.mp4'

    savedir_ = os.path.split(os.path.split(imgpath)[0])[0]
    savepath = os.path.join(savedir_, savename)

    # Animate a full tiff stack of two-photon data.
    img_rollavg = imgtools.rolling_average(imgtools.load_tif_stack(
        imgpath, doReg=True, doNorm=False
    ))
    plot_stack = []
    for f in tqdm(range(np.size(img_rollavg,0))):
        fig = plt.figure(dpi=300, figsize=(4,4))
        plt.imshow(img_rollavg[f,:,:], cmap='gray', vmin=0, vmax=300)
        plt.axis('off')
        plt.tight_layout()
        plot_stack.append(imgtools.fmt_figure(fig))
    plot_stack = np.array(plot_stack)
    imgtools.write_animation(
        plot_stack,
        savepath,
        60
    )
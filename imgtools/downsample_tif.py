
import os
import numpy as np
import tifffile as tiff
import skimage.measure
from tqdm import tqdm

import imgtools

def downsample_tif(tif_path=None, ds=0.5):

    if tif_path is None:
        print('CHoose tif file.')
        tif_path = imgtools.select_file(
            title='Choose tif file.',
            filetypes=[('TIF', '*.tif'),('TIF','*.tiff'),]
        )

    print(tif_path)

    arr = tiff.imread(tif_path)

    nF = int(np.size(arr,0))
    sz1 = int(np.size(arr,1))
    sz2 = int(np.size(arr,2))

    blocksize = (
        int(sz1/(sz1*ds)),
        int(sz2/(sz2*ds))
    )

    newarr = np.zeros([nF, int(sz1*ds), int(sz2*ds)])

    for fr in tqdm(range(nF)):
        newarr[fr,:,:] = skimage.measure.block_reduce(arr[fr,:,:], blocksize, np.nanmean)

    savepath = os.path.splitext(tif_path)[0] + '_downsample.tif'
    tiff.imwrite(savepath, newarr)


if __name__ == '__main__':

    downsample_tif()
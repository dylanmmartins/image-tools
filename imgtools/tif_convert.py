"""
imgtools/tif_convert.py
Convert a directory of .tif files to a multi-page
.tif file

Author        : Dylan Martins
Written       : April 12 2024
"""


import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import PySimpleGUI as sg
import tifffile
from scipy.io import savemat

sg.theme('Default1')


def tif_convert(firstfile=None, savepath=None,
                delete_singles=False, ret=False, saveas='tif',
                multicycle=False):

    if firstfile is None:
        firstfile = sg.popup_get_file(
            'Choose first file',
            title='Choose first file',
            multiple_files=False,
            no_window=True,
            file_types=(('TIF', '*.tif'),
                        ('TIF','*.tiff')),
            keep_on_top=True
        )

    firstfile = os.path.normpath(firstfile)

    head, tail = os.path.split(firstfile)
    
    if savepath is None:
        savepath = head
    else:
        savepath = os.path.normpath(savepath)

    base_filename = '_'.join(tail.split('_')[:-1])
    add_ext = '.'.join(firstfile.split('.')[1:])

    file_list = []

    if multicycle is True:
        file_list = glob(os.path.join(head, '*.ome.tif'))
        file_list = [f for f in file_list if 'Reference' not in f]
        file_list = sorted(file_list)

    elif multicycle is False:

        nF = 0
        while True:
            nF += 1
            test_name = '{}_{:06d}.{}'.format(
                base_filename,
                nF,
                add_ext
            )
            testpath = os.path.join(head, test_name)

            if os.path.isfile(testpath):
                file_list.append(testpath)

            elif not os.path.isfile(testpath):
                break

    num_files = len(file_list)

    im = np.array(Image.open(firstfile))
    xPixels = np.size(im, 0)
    yPixels = np.size(im, 1)

    imgstack = np.zeros(
        [num_files, xPixels, yPixels],
        dtype=np.uint16
    )

    print('Reading single-page tif files.')
    for i in tqdm(range(len(file_list))):
        f = file_list[i]
        imgstack[i,:,:] = np.array(Image.open(f), dtype=np.uint16)

    if saveas=='tif':
        print('Writing to multi-page tif file.')
        savefilename = os.path.join(head, base_filename+'.tif')
        with tifffile.TiffWriter(savefilename, bigtiff=True) as savestack:
            savestack.write(
                data=imgstack,
                dtype=np.uint16,
                shape=(num_files, xPixels, yPixels),
                photometric='MINISBLACK'
            )

    #TODO: test npy and mat save methods... haven't had to used these yet

    elif saveas=='npy':
        print('Writing to numpy array.')
        savefilename = os.path.join(head, base_filename+'.npy')
        np.save(savefilename, imgstack)

    elif saveas=='mat':
        print('Writing to Matlab file.')
        savefilename = os.path.join(head, base_filename+'.mat')
        savemat(
            savefilename,
            mdict={'TifStack': imgstack}
        )

    print('New file written to {}'.format(savefilename))

    if delete_singles is True:
        print('Deleting single-page tif files.')
        for f in file_list:
            os.remove(f)

    if ret is True:
        return savefilename, imgstack
    
    elif ret is False:
        return savefilename


def from_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str, default=None)
    parser.add_argument('-s', '--savepath', type=str, default=None)
    parser.add_argument('-as', '--saveas', default='tif')
    parser.add_argument('-del', '--delete', default=True)
    args = parser.parse_args()

    # ret flag will always be false when called from terminal, which
    # is the only access point to this func

    tif_convert(args.filepath, args.savepath,
                delete_singles=args.delete,
                ret=False,
                saveas=args.saveas)


if __name__ == '__main__':

    from_args()
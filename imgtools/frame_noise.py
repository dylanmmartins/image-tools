import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import tifffile
import os
from matplotlib.backends.backend_pdf import PdfPages

import imgtools


def resscan_denoise(tif_path=None, ret=False):
    # Remove noise added into mini2p image stack by resonance scanner. noise
    # appears as hazy vertical banding which sweeps slowly along the x axis
    # (they are not in static positions, and there are ~10 overlapping bands
    # in the image for any given frame. they move both leftwards and rightwards.
    # If ret is true, the function will return the image stack with a short (3 frame)
    # rolling average applied.

    if tif_path is None:
        tif_path = imgtools.select_file(
            'Select tif stack.',
            filetypes=[('TIF', '*.tif'),('TIF','*.tiff'),]
        )

    rawimg = imgtools.load_tif_stack(tif_path)

    base_path = os.path.split(tif_path)[0]
    pdf = PdfPages(os.path.join(base_path, 'denoising_figs.pdf'))

    mean_of_banded_block = np.mean(rawimg[:,412:,:],1)

    fig = plt.figure(figsize=(7,4), dpi=300)
    plt.imshow(mean_of_banded_block, aspect=0.08, cmap='gray', vmin=0, vmax=100)
    plt.colorbar()
    plt.xlabel('y pixels')
    plt.ylabel('time (frames)')
    plt.tight_layout()
    pdf.savefig(fig)


    f_size = np.shape(rawimg[0,:,:])
    noise_pattern = np.zeros_like(rawimg)
    for f in tqdm(range(np.size(noise_pattern,0))):
        frsn = imgtools.boxcar_smooth(mean_of_banded_block[f,:],5)
        noise_pattern[f,:,:] = np.broadcast_to(frsn, f_size).copy()

    newimg = np.subtract(rawimg, noise_pattern)

    f = 500
    fig, [ax1,ax2,ax3] = plt.subplots(1,3, figsize=(5.5,3), dpi=300)
    ax1.imshow(rawimg[f,:,:], cmap='gray', vmin=0, vmax=200)
    ax2.imshow(noise_pattern[f,:,:], cmap='gray')
    ax3.imshow(newimg[f,:,:], cmap='gray', vmin=0, vmax=200)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    fig.tight_layout()
    pdf.savefig(fig)

    # mean across frames for output image data (is there drift over time or large jumps? it should not.)
    meanF = np.mean(newimg,axis=(1,2))
    # mean across frames for the oinse pattern (does it have large jumps or drift over time? it should)
    meanP = np.mean(noise_pattern,axis=(1,2))

    fig, [ax1,ax2] = plt.subplots(1,2, dpi=300, figsize=(8,2.5))
    ax1.plot(meanF, color='k', lw=1)
    ax2.plot(meanP, color='k', lw=1)
    ax1.set_xlabel('frames')
    ax2.set_xlabel('frames')
    ax1.set_ylabel('frame mean pixel value')
    ax2.set_ylabel('frame mean pixel value')
    ax1.set_ylim([np.percentile(meanF, 0.1), np.percentile(meanF, 99.9)])
    ax2.set_ylim([np.percentile(meanP, 0.1), np.percentile(meanP, 99.9)])
    ax1.set_title('noise-corrected stack')
    ax2.set_title('putative noise pattern')
    fig.tight_layout()
    pdf.savefig(fig)

    # Save two versions of the output video: one raw video, one with a small
    # rolling average, and one with a large rolling average.
    # For the small rolling average, apply a 400 msec smoothing window
    sra_newimg = imgtools.rolling_average(newimg, 3)
    # for the large rolling average, apply a 1600 msec smoothing window (this
    # is probably only useful for visualization)
    lra_newimg = imgtools.rolling_average(newimg, 12)

    full_numF = np.size(newimg,0)
    sra_len = np.size(sra_newimg,0)
    lra_len = np.size(lra_newimg,0)

    frame_note = (
        'The full tif stack had {} frames. The denoised tif stack with a short running average '
        'has {} frames, and the one with a long running average has {} frames. When aligning '
        'the denoised stacks to other data streams, subtract diff/2 from the start and end. '
        'Adjust SRA by {} and LRA by {}.'
    )
    sra_adjust = int((np.size(noise_pattern,0)-np.size(sra_len,0))/2)
    lra_adjust = int((np.size(noise_pattern,0)-np.size(lra_len,0))/2)
    frame_note = frame_note.format(full_numF, sra_len, lra_len, sra_adjust, lra_adjust)
    with open('note_on_denoised_tif_dims.txt', 'w') as file:
        file.write(frame_note)
    print(frame_note)

    base_tif_path = os.path.splitext(tif_path)[0]
    s_savefilename = base_tif_path+'_denoised_SRA.tif'
    with tifffile.TiffWriter(s_savefilename, bigtiff=True) as savestack:
        savestack.write(
            data=sra_newimg,
            dtype=np.uint16,
            shape=sra_newimg.shape,
            photometric='MINISBLACK'
        )

    l_savefilename = base_tif_path+'_denoised_SRA.tif'
    with tifffile.TiffWriter(l_savefilename, bigtiff=True) as savestack:
        savestack.write(
            data=lra_newimg,
            dtype=np.uint16,
            shape=lra_newimg.shape,
            photometric='MINISBLACK'
        )

    pdf.close()

    if ret:
        return sra_newimg

def make_denoise_diagnostic_video(ra_img, noise_pattern, ra_newimg, vid_save_path, startF, endF):
    # make animation

    # start/end crop value to align noise pattern with smoothed image stacks
    # important to do the smoothing after noise is subtracted instead of before!
    startEndFCrop = int((np.size(noise_pattern,0)-np.size(ra_img,0))/2)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(vid_save_path, fourcc, (7.5*8), (1650, 900))

    for f in tqdm(np.arange(startF, endF)):

        fig, [ax1,ax2,ax3] = plt.subplots(1,3, figsize=(5.5,3), dpi=300)
        ax1.imshow(ra_img[f,:,:], cmap='gray', vmin=0, vmax=200)
        ax2.imshow(noise_pattern[f+startEndFCrop,:,:], cmap='gray', vmin=-10, vmax=120)
        ax3.imshow(ra_newimg[f,:,:], cmap='gray', vmin=0, vmax=200)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        fig.suptitle('frame {}'.format(f))
        fig.tight_layout()

        fig.canvas.draw()
        frame_as_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        frame_as_array = frame_as_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        img = cv2.cvtColor(frame_as_array, cv2.COLOR_RGB2BGR)
        out_vid.write(img.astype('uint8'))

    out_vid.release()


if __name__ == '__main__':


    # tif_path = r'T:\axonal_imaging_LP\250430_DMM_DMM046_LPaxons\file_00004.tif'
    resscan_denoise()


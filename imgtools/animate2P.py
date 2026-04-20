# -*- coding: utf-8 -*-
"""
Write animated video for 2P data.

Functions
---------
animate_stack_from_2P(imgpath, ...)
    Animate a full tiff stack of two-photon data.

Author: DMM, 2024
"""

import os
import sys
import argparse
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

import imgtools


_SHARED_FRAMES = None


def _render_chunk(args):

    start, end, vmin, vmax, cmap_name, figsize, dpi = args
    chunk = _SHARED_FRAMES[start:end]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(chunk[0], cmap=cmap_name, vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.tight_layout()
    fig.canvas.draw()

    fig_w_in, fig_h_in = fig.get_size_inches()
    w_px = int(fig_w_in * fig.get_dpi())
    h_px = int(fig_h_in * fig.get_dpi())

    bgr_frames = []
    for frame in chunk:
        im.set_data(frame)
        fig.canvas.draw()

        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4)
        bgr_frames.append(cv2.cvtColor(buf[:, :, :3].copy(), cv2.COLOR_RGB2BGR))

    plt.close(fig)
    return bgr_frames


def animate_stack_from_2P(
    imgpath,
    savename=None,
    num_frames=3600,
    window=20,
    vmin=0,
    vmax=400,
    fps=30,
    dpi=300,
    figsize=(4, 4),
    cmap='gray',
    n_workers=None,
    chunk_size=50,
):

    global _SHARED_FRAMES

    if savename is None:
        savename = 'animation_01.mp4'

    savedir_ = os.path.split(os.path.split(imgpath)[0])[0]
    savepath = os.path.join(savedir_, savename)

    print('Loading and preprocessing frames...')
    img_rollavg = imgtools.rolling_average(
        imgtools.read_tif_until(imgpath, num_frames),
        window=window,
    )
    n_frames = img_rollavg.shape[0]

    _SHARED_FRAMES = img_rollavg

    if n_workers is None:
        n_workers = os.cpu_count()

    fig_test, ax_test = plt.subplots(figsize=figsize, dpi=dpi)
    ax_test.imshow(img_rollavg[0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax_test.axis('off')
    plt.tight_layout()
    fig_test.canvas.draw()
    fig_w_in, fig_h_in = fig_test.get_size_inches()
    w_px = int(fig_w_in * fig_test.get_dpi())
    h_px = int(fig_h_in * fig_test.get_dpi())
    plt.close(fig_test)

    out = cv2.VideoWriter(
        savepath,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w_px, h_px),
    )

    starts = list(range(0, n_frames, chunk_size))
    chunk_args = [
        (s, min(s + chunk_size, n_frames), vmin, vmax, cmap, figsize, dpi)
        for s in starts
    ]

    print(
        f'Rendering {n_frames} frames with {n_workers} workers '
        f'({len(chunk_args)} chunks of up to {chunk_size} frames)...'
    )

    ctx = mp.get_context('fork')
    with ctx.Pool(n_workers) as pool:
        for bgr_chunk in tqdm(
            pool.imap(_render_chunk, chunk_args),
            total=len(chunk_args),
            unit='chunk',
        ):
            for bgr_frame in bgr_chunk:
                out.write(bgr_frame)

    out.release()
    print(f'Video written to {savepath}')



def _build_parser():
    parser = argparse.ArgumentParser(
        prog='animate2P',
        description='Animate a two-photon TIFF stack as an MP4 video.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        'imgpath',
        help='Path to the input TIFF stack.',
    )
    parser.add_argument(
        '-o', '--savename',
        default=None,
        metavar='FILE',
        help='Output filename (saved one directory above the TIFF parent). '
             'Defaults to animation_01.mp4.',
    )
    parser.add_argument(
        '-n', '--num-frames',
        type=int,
        default=3600,
        metavar='N',
        help='Maximum number of frames to read from the TIFF.',
    )
    parser.add_argument(
        '--window',
        type=int,
        default=20,
        metavar='W',
        help='Rolling-average temporal window (frames).',
    )
    parser.add_argument(
        '--vmin',
        type=float,
        default=0.0,
        help='Lower bound of the display intensity range.',
    )
    parser.add_argument(
        '--vmax',
        type=float,
        default=400.0,
        help='Upper bound of the display intensity range.',
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Output video frame rate.',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Figure resolution in dots per inch.',
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[4.0, 4.0],
        metavar=('W', 'H'),
        help='Figure width and height in inches.',
    )
    parser.add_argument(
        '--cmap',
        default='gray',
        help='Matplotlib colormap name.',
    )
    parser.add_argument(
        '-j', '--workers',
        type=int,
        default=None,
        dest='n_workers',
        metavar='N',
        help='Number of worker processes. Defaults to CPU count.',
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=50,
        metavar='N',
        help='Frames rendered per worker dispatch.',
    )

    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    animate_stack_from_2P(
        imgpath=args.imgpath,
        savename=args.savename,
        num_frames=args.num_frames,
        window=args.window,
        vmin=args.vmin,
        vmax=args.vmax,
        fps=args.fps,
        dpi=args.dpi,
        figsize=tuple(args.figsize),
        cmap=args.cmap,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
    )

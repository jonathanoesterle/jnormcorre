import numpy as np
from matplotlib import pyplot as plt


def create_test_data(nx, ny, nt, sx, sy, offxs, offys, maxx, maxy, noise_std=0.5, seed=42):
    np.random.seed(seed)
    dat = np.random.uniform(0, noise_std, (nx, ny, nt))

    shifts_x = np.round(np.linspace(0, maxx, nt)).astype(int)
    shifts_y = np.round(np.linspace(0, maxy, nt)).astype(int)

    for i, (shift_x, shift_y) in enumerate(zip(shifts_x, shifts_y)):
        for offx, offy in zip(offxs, offys):
            dat[offx + shift_x:offx + shift_x + sx, offy + shift_y:offy + shift_y + sy, i] = (
                np.random.normal(1, noise_std, (sx, sy)))

    return dat, shifts_x, shifts_y


def plot_test_data(dat, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 4, sharex='all', sharey='all', figsize=(12, 4))
    for i, frame_i in enumerate(np.linspace(0, dat.shape[2] - 1, len(axs), endpoint=True).astype(int)):
        axs[i].set_title(frame_i)
        axs[i].imshow(dat[:, :, frame_i].T)
    return axs


def plot_test_data_and_corrected_test_data(dat, dat_corr, axs=None):
    if axs is None:
        fig, axs = plt.subplots(2, 4, sharex='all', sharey='all', figsize=(12, 8))

    for i, frame_i in enumerate(np.linspace(0, dat.shape[2] - 1, len(axs), endpoint=True).astype(int)):
        axs[0, i].set_title(frame_i)
        axs[0, i].imshow(dat[:, :, frame_i].T)
        axs[1, i].imshow(dat_corr[:, :, frame_i].T)
    return axs

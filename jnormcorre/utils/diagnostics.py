import numpy as np
import tifffile


def motion_correction_diagnostic(original_file, registered_file, frame_list=None):
    if frame_list is None:
        original_movie = tifffile.imread(original_file).transpose(1, 2, 0)
        registered_movie = tifffile.imread(registered_file).transpose(1, 2, 0)
    else:
        original_movie = tifffile.imread(original_file, key=frame_list).transpose(1, 2, 0)
        registered_movie = tifffile.imread(registered_file, key=frame_list).transpose(1, 2, 0)
    d1, d2, T = original_movie.shape
    display_movie = np.zeros((d1, d2 * 2, T), dtype=np.float32)
    display_movie[:, :d2, :] = original_movie
    display_movie[:, d2:, :] = registered_movie

    return display_movie

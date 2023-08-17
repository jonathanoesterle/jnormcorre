import numpy as np
from jnormcorre import motion_correction


def compute_shifts(
        stack,
        pixel_size_um,
        max_shift_um=(12., 12.),
        max_deviation_rigid=3,
        patch_motion_um=(100., 100.),
        overlaps=(24, 24),
        splits=200,
        gSig_filt=None,
):
    """
    Runs motion correction from caiman on the input dataset with the
    option to process the same dataset in multiple passes.
    Parameters
    ----------
    stack : np.ndarray (x, y, t)
        Full path + name for destination of output config file.
    pixel_size_um: float or (float, float)
        Spatial resolution in x and y in (um per pixel)
    max_shift_um: float or (float, float)
        Maximum shift in um
    max_deviation_rigid: int
        Maximum deviation allowed for patch with respect to rigid shifts
    patch_motion_um:
        Patch size for non-rigid correction in um
    overlaps:
        Overlap between patches
    splits: int
        We divide the registration into chunks (temporally). Splits = number of frames in each chunk.
        So splits = 200 means we break the data into chunks, each containing ~200 frames.
    """
    nx, xy, nt = stack.shape
    splits = int(np.ceil(nt / splits))

    if not isinstance(pixel_size_um, tuple):
        pixel_size_um = (pixel_size_um, pixel_size_um)
    if not isinstance(max_shift_um, tuple):
        max_shift_um = (max_shift_um, max_shift_um)

    max_shifts = [int(a / b) for a, b in zip(max_shift_um, pixel_size_um)]
    strides = tuple([int(a / b) for a, b in zip(patch_motion_um, pixel_size_um)])

    mc_dict = {
        'min_mov': -5,  # minimum value of movie
        'niter_rig': 4,  # number of iterations rigid motion correction
        'niter_els': 1,  # number of iterations of piecewise rigid motion correction
        'nonneg_movie': True,  # flag for producing a non-negative movie
        'num_splits_to_process_els': None,
        'num_splits_to_process_rig': None,
        'splits_els': splits,  # number of splits across time for pw-rigid registration
        'splits_rig': splits,  # number of splits across time for rigid registration
        'upsample_factor_grid': 4,  # motion field upsampling factor during FFT shifts
        'indices': (slice(None), slice(None)),  # part of FOV to be corrected
        'max_shifts': max_shifts,  # maximum shifts per dimension (in pixels)
        'strides': strides,
        'gSig_filt': gSig_filt,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
    }

    corrector = motion_correction.MotionCorrect(stack.T, pw_rigid=False, **mc_dict)
    corrector.motion_correct(save_movie=False)

    shifts_y, shifts_x = - np.vstack(corrector.shifts_rig).T

    return shifts_x, shifts_y

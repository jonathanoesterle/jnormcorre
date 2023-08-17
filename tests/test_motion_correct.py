import numpy as np

from jnormcorre import correct_stack
from jnormcorre.utils import simulate


def test_motion_correct_simple_rigid():
    dat, shifts_x, shifts_y = simulate.create_test_data(30, 20, 500, 5, 5, (1, 10), (1, 4), 10, 0)
    rec_shifts_x, rec_shifts_y = correct_stack.compute_shifts(dat, pixel_size_um=1., max_shift_um=6., splits=4)
    corr = np.corrcoef(shifts_x, rec_shifts_x)[0, 1]
    assert corr > 0.95

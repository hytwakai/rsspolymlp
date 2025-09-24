import numpy as np
import spglib


def convert_niggli(axis, positions):
    niggli_axis = spglib.niggli_reduce(axis)
    proj = np.linalg.solve(niggli_axis.T, axis.T).T
    converted_pos = positions @ proj

    return niggli_axis, converted_pos

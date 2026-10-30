from typing import Optional

import numpy as np
import spglib

from pypolymlp.core.data_format import PolymlpStructure
from rsspolymlp.analysis.struct_matcher.chiral_spg import is_chiral
from rsspolymlp.common.interface_vasp import Poscar


def calc_metric_tensor(axis: np.ndarray):
    a, b, c = np.array(axis)
    aa = np.sum(a**2)
    bb = np.sum(b**2)
    cc = np.sum(c**2)
    ab = a @ b
    ac = a @ c
    bc = b @ c
    return np.array([aa, bb, cc, ab, ac, bc])


def get_reduced_lattice(
    axis: np.ndarray,
    spg_number: int,
    symprec: list[float] = [1e-2, 1e-2, 1e-2],
):
    proper_matrices, improper_matrices = signed_permutation_matrices()
    if not is_chiral(spg_number):
        all_signed_permutation_matrices = np.concatenate(
            [proper_matrices, improper_matrices], axis=0
        )
    else:
        all_signed_permutation_matrices = proper_matrices

    all_axis = []
    for P in all_signed_permutation_matrices:
        trans_axis = P @ axis
        all_axis.append(trans_axis)
    all_axis = np.array(all_axis)

    all_metric_tensor = []
    for one_axis in all_axis:
        all_metric_tensor.append(calc_metric_tensor(one_axis))
    all_metric_tensor = np.array(all_metric_tensor)

    axis_cands = all_axis
    metric_tensor_cands = all_metric_tensor
    signed_permutation_cands = all_signed_permutation_matrices
    for idx in range(6):
        # Get the reduced axis representation
        ref_metric = np.min(metric_tensor_cands[:, idx])
        is_near_max = np.where(
            np.abs(metric_tensor_cands[:, idx] - ref_metric) <= symprec[idx % 3]
        )[0]
        # Update the axis candidates
        axis_cands = axis_cands[is_near_max, :]
        metric_tensor_cands = metric_tensor_cands[is_near_max, :]
        signed_permutation_cands = signed_permutation_cands[is_near_max, :]
        if metric_tensor_cands.shape[0] == 1:
            break

    reduced_metric_tensor = metric_tensor_cands[0]

    aa, bb, cc, ab, ac, bc = reduced_metric_tensor
    G = np.array([[aa, ab, ac], [ab, bb, bc], [ac, bc, cc]], dtype=float)
    w, U = np.linalg.eigh(G)
    w = np.clip(w, 0, None)
    G_sqrt = (U * np.sqrt(w)) @ U.T
    metric_tensor_sqrt = np.array(
        [
            G_sqrt[0, 0],
            G_sqrt[1, 1],
            G_sqrt[2, 2],
            G_sqrt[0, 1],
            G_sqrt[0, 2],
            G_sqrt[1, 2],
        ]
    )
    return metric_tensor_sqrt, axis_cands, signed_permutation_cands


def signed_permutation_matrices():
    proper_matrices = np.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
            [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
            [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
            [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
            [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
            [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
            [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
            [[0, 0, -1], [0, -1, 0], [-1, 0, 0]],
        ]
    )

    improper_matrices = np.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[1, 0, 0], [0, 0, -1], [0, -1, 0]],
            [[-1, 0, 0], [0, 0, 1], [0, -1, 0]],
            [[-1, 0, 0], [0, 0, -1], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            [[0, 1, 0], [-1, 0, 0], [0, 0, -1]],
            [[0, -1, 0], [1, 0, 0], [0, 0, -1]],
            [[0, -1, 0], [-1, 0, 0], [0, 0, 1]],
            [[0, 1, 0], [0, 0, 1], [-1, 0, 0]],
            [[0, 1, 0], [0, 0, -1], [1, 0, 0]],
            [[0, -1, 0], [0, 0, 1], [1, 0, 0]],
            [[0, -1, 0], [0, 0, -1], [-1, 0, 0]],
            [[0, 0, 1], [1, 0, 0], [0, -1, 0]],
            [[0, 0, 1], [-1, 0, 0], [0, 1, 0]],
            [[0, 0, -1], [1, 0, 0], [0, 1, 0]],
            [[0, 0, -1], [-1, 0, 0], [0, -1, 0]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            [[0, 0, 1], [0, -1, 0], [-1, 0, 0]],
            [[0, 0, -1], [0, 1, 0], [-1, 0, 0]],
            [[0, 0, -1], [0, -1, 0], [1, 0, 0]],
        ]
    )

    return proper_matrices, improper_matrices


def niggli_reduce(
    poscar_name: str = None,
    st: Optional[PolymlpStructure] = None,
    axis: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    symprec: float = 1e-5,
):
    """Niggli reduction."""
    if poscar_name is not None:
        st = Poscar(poscar_name).structure

    if st is None:
        axis = np.array(axis)
        positions = np.array(positions)
    else:
        axis = st.axis.T
        positions = st.positions.T

    axis_niggli = spglib.niggli_reduce(
        axis,
        eps=symprec,
    )
    M_float = axis_niggli @ np.linalg.inv(axis)
    M = np.rint(M_float).astype(np.int64)
    pos = positions @ np.linalg.inv(M)
    pos %= 1.0

    assert np.allclose(M @ axis, axis_niggli, atol=1e-5)
    assert abs(round(np.linalg.det(M))) == 1
    return axis_niggli, pos, M

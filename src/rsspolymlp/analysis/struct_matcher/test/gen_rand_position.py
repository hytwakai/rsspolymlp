from typing import Literal, Optional

import numpy as np

from rsspolymlp.analysis.struct_matcher.redrep_position import (
    enumerate_metric_symmetries,
)
from rsspolymlp.common.interface_spglib import SymCell


def apply_rand_disp(
    axis,
    positions,
    elements,
    spg_number: Optional[int] = None,
    max_disp: float = 1e-2,
    disp_mode: Literal["uniform", "fixed"] = "uniform",
    tol=1e-2,
):
    pos_rep = np.asarray(positions, dtype=float).copy()
    _elements = np.asarray(elements, dtype=str).copy()

    if spg_number is None:
        symutil = SymCell(
            axis,
            pos_rep,
            _elements,
            symprec=tol,
        )
        spg_number = symutil.get_spg_number()

    if disp_mode == "uniform":
        random_disp = 2 * np.random.rand(len(elements), 3) - 1
        random_disp *= max_disp
    elif disp_mode == "fixed":
        random_disp = np.random.randn(len(elements), 3)
        random_disp /= np.linalg.norm(random_disp, axis=1)[:, None]
        random_disp *= max_disp
    else:
        raise ValueError(f"Unknown disp_mode: {disp_mode}")
    pos_rep = (pos_rep + random_disp @ np.linalg.inv(axis)) % 1.0

    random_translation = np.random.rand(1, 3)
    pos_rep = (pos_rep + random_translation) % 1.0

    random_permutation = np.random.permutation(len(elements))
    pos_rep = pos_rep[random_permutation]
    _elements = _elements[random_permutation]

    sort_idx = np.argsort(_elements)
    pos_rep = pos_rep[sort_idx]
    _elements = _elements[sort_idx].tolist()

    transformation_matrices = enumerate_metric_symmetries(
        axis,
        axis,
        spg_number,
        tol=tol,
    )
    selected_i = np.random.randint(len(transformation_matrices))
    random_transformation = transformation_matrices[selected_i]
    pos_rep = pos_rep @ np.linalg.inv(random_transformation)
    pos_rep %= 1.0

    return pos_rep, _elements


def gen_random_elements(n_atom, n_elements):
    element_cands = np.array(list("ABCDEFGHIJKLMN"))[:n_elements]
    random_elements = np.sort(np.random.choice(element_cands, size=n_atom))
    return random_elements

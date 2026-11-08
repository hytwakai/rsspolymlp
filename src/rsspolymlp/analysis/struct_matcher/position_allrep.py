import itertools
from collections import Counter
from typing import Optional

import numpy as np

from rsspolymlp.analysis.struct_matcher.chiral_spg import is_chiral


def get_all_position_representations(
    axis: np.ndarray,
    positions: np.ndarray,
    elements: np.ndarray,
    tol_sm: float = 1e-2,
    spg_number: int = 1,
    transformation_matrices: Optional[np.ndarray] = None,
    normalize_axis: bool = False,
):
    positions = np.asarray(positions, dtype=float)
    elements = np.asarray(elements, dtype=float)

    unique_elements = np.sort(np.unique(elements))
    types = np.array([np.where(unique_elements == el)[0][0] for el in elements])

    counts = Counter(elements)
    min_count = min(counts.values())
    least_elements = [el for el, cnt in counts.items() if cnt == min_count]
    target_element = sorted(least_elements)[0]
    target_type = np.where(unique_elements == target_element)[0][0]
    types = (types - target_type) % (np.max(types) + 1)

    sort_idx = np.argsort(types)
    sorted_positions = positions[sort_idx]
    sorted_elements = elements[sort_idx]

    if normalize_axis:
        volume = abs(np.linalg.det(axis))
        axis = axis / ((volume / len(positions)) ** (1 / 3))
    a, b, c = np.array(axis)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    norm_c = np.linalg.norm(c)
    axis_abc = np.array([norm_a, norm_b, norm_c])

    if transformation_matrices is None:
        transformation_matrices = matching_metric_tensor(
            axis, axis, spg_number, tol=tol_sm
        )

    all_posrep = []
    target_indices = np.where(types == 0)[0]
    for trans_mat in transformation_matrices:
        _positions = sorted_positions.copy()
        _positions = _positions @ np.linalg.inv(trans_mat)
        _positions %= 1.0

        pos_cands = _positions[None, :, :] - _positions[target_indices, None, :]
        pos_cands %= 1.0
        pos_cands = pos_cands * axis_abc
        for pos in pos_cands:
            all_posrep.append(pos)
    all_posrep = np.array(all_posrep)

    return all_posrep, sorted_elements


def matching_metric_tensor(
    axis_1: np.ndarray,
    axis_2: np.ndarray,
    spg_number: int,
    tol: float = 1e-3,
) -> list[np.ndarray]:
    """Enumerate unimodular integer matrices satisfying M @ G_1 @ M.T = G_2."""
    G_1 = axis_1 @ axis_1.T
    G_2 = axis_2 @ axis_2.T
    if not (np.allclose(G_1, G_1.T, atol=tol) and np.allclose(G_2, G_2.T, atol=tol)):
        raise ValueError("G must be symmetric.")

    norm_a = G_1[0, 0] ** 0.5
    norm_b = G_1[1, 1] ** 0.5
    norm_c = G_1[2, 2] ** 0.5
    e_aa = tol * 2 * norm_a
    e_bb = tol * 2 * norm_b
    e_cc = tol * 2 * norm_c
    e_bc = tol * (norm_b + norm_c)
    e_ac = tol * (norm_a + norm_c)
    e_ab = tol * (norm_a + norm_b)

    # Enumerate candidate column vectors.
    candidates = [
        enumerate_integer_vectors(G_1, G_2[i, i], [e_aa, e_bb, e_cc][i])
        for i in range(3)
    ]
    transformation_matrices = []

    # Enumerate combinations of the three columns.
    for m1, m2, m3 in itertools.product(*candidates):
        if abs(m1 @ G_1 @ m2 - G_2[0, 1]) > e_ab:
            continue
        if abs(m1 @ G_1 @ m3 - G_2[0, 2]) > e_ac:
            continue
        if abs(m2 @ G_1 @ m3 - G_2[1, 2]) > e_bc:
            continue
        M = np.column_stack((m1, m2, m3))

        det = round(np.linalg.det(M))
        if abs(det) != 1:
            continue
        if is_chiral(spg_number) and det == -1:
            continue
        transformation_matrices.append(M)

    return transformation_matrices


def enumerate_integer_vectors(
    G: np.ndarray,
    q: float,
    tol: float = 1e-5,
) -> np.ndarray:
    """Enumerate integer vectors satisfying v.T @ G @ v ~= q."""
    G = np.asarray(G, dtype=float)

    # Upper bound from the minimum eigenvalue.
    lambda_min = np.linalg.eigvalsh(G)[0]
    if lambda_min <= 0:
        raise ValueError("G must be positive definite.")

    vectors = []
    bound = int(np.ceil(np.sqrt((q + tol) / lambda_min)))
    for v in itertools.product(range(-bound, bound + 1), repeat=3):
        v = np.asarray(v, dtype=int)
        if abs(v @ G @ v - q) <= tol:
            vectors.append(v)

    return np.asarray(vectors, dtype=int).reshape(-1, 3)

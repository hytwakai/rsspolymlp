import itertools

import numpy as np

from rsspolymlp.analysis.struct_matcher.chiral_spg import is_chiral
from rsspolymlp.analysis.struct_matcher.niggli_reduce import get_niggli_reduced_cell


def get_reduced_lattice(
    axis: np.ndarray,
    positions: np.ndarray,
    spg_number: int,
    tol: float = 1e-3,
):
    axis_niggli, positions_niggli = niggli_reduce(
        axis=axis, positions=positions, tol=tol
    )
    axis = axis_niggli
    positions = positions_niggli

    G = np.dot(axis, axis.T)
    transformation_matrices = enumerate_metric_symmetries(
        G, is_chiral(spg_number), tol=tol
    )

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
    return axis, positions, metric_tensor_sqrt, transformation_matrices


def niggli_reduce(
    axis: np.ndarray,
    positions: np.ndarray,
    tol: float = 1e-3,
):
    """Niggli reduction."""
    axis = np.array(axis)
    positions = np.array(positions)

    axis_niggli, P = get_niggli_reduced_cell(
        axis,
        tol=tol,
    )
    pos = positions @ np.linalg.inv(P)
    pos %= 1.0

    assert abs(round(np.linalg.det(P))) == 1
    return axis_niggli, pos


def enumerate_metric_symmetries(
    G: np.ndarray,
    is_chiral: bool = False,
    tol: float = 1e-3,
) -> list[np.ndarray]:
    """Enumerate unimodular integer matrices satisfying M @ G @ M.T = G."""
    G = np.asarray(G, dtype=float)

    if G.shape != (3, 3):
        raise ValueError("G must be a 3x3 matrix.")

    if not np.allclose(G, G.T, atol=tol):
        raise ValueError("G must be symmetric.")

    if np.linalg.eigvalsh(G)[0] <= 0:
        raise ValueError("G must be positive definite.")

    # Enumerate candidate column vectors.
    candidates = [enumerate_integer_vectors(G, G[i, i], tol) for i in range(3)]
    matrices = []

    # Enumerate combinations of the three columns.
    for m1, m2, m3 in itertools.product(*candidates):
        # Inner-product conditions.
        if abs(m1 @ G @ m2 - G[0, 1]) > tol:
            continue
        if abs(m1 @ G @ m3 - G[0, 2]) > tol:
            continue
        if abs(m2 @ G @ m3 - G[1, 2]) > tol:
            continue
        M = np.column_stack((m1, m2, m3))

        # Unimodularity condition.
        det = round(np.linalg.det(M))
        if abs(det) != 1:
            continue
        if is_chiral and det == -1:
            continue
        matrices.append(M)

    return matrices


def enumerate_integer_vectors(
    G: np.ndarray,
    q: float,
    tol: float = 1e-8,
) -> np.ndarray:
    """Enumerate integer vectors satisfying v.T @ G @ v ~= q."""
    G = np.asarray(G, dtype=float)

    # Upper bound from the minimum eigenvalue.
    lambda_min = np.linalg.eigvalsh(G)[0]
    if lambda_min <= 0:
        raise ValueError("G must be positive definite.")

    bound = int(np.ceil(np.sqrt((q + tol) / lambda_min)))

    vectors = []
    for v in itertools.product(range(-bound, bound + 1), repeat=3):
        v = np.asarray(v, dtype=int)

        if abs(v @ G @ v - q) <= tol:
            vectors.append(v)

    return np.asarray(vectors, dtype=int).reshape(-1, 3)

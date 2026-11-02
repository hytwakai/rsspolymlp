import itertools

import numpy as np

from rsspolymlp.analysis.struct_matcher.chiral_spg import is_chiral
from rsspolymlp.analysis.struct_matcher.niggli_reduce import get_niggli_reduced_cell


def get_reduced_lattice(
    axis: np.ndarray,
    positions: np.ndarray,
    spg_number: int,
    symprec: float = 1e-2,
):
    axis_niggli, positions_niggli, _ = niggli_reduce(
        axis=axis, positions=positions, symprec=symprec
    )
    axis = axis_niggli
    positions = positions_niggli

    metric_tensor = calc_metric_tensor(axis)
    aa, bb, cc, ab, ac, bc = metric_tensor
    G = np.array([[aa, ab, ac], [ab, bb, bc], [ac, bc, cc]], dtype=float)
    transformation_matrices = enumerate_metric_symmetries(
        G, is_chiral(spg_number), tol=symprec
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


def calc_metric_tensor(axis: np.ndarray):
    a, b, c = np.array(axis)
    aa = np.sum(a**2)
    bb = np.sum(b**2)
    cc = np.sum(c**2)
    ab = a @ b
    ac = a @ c
    bc = b @ c
    return np.array([aa, bb, cc, ab, ac, bc])


def niggli_reduce(
    axis: np.ndarray,
    positions: np.ndarray,
    symprec: float = 1e-5,
):
    """Niggli reduction."""
    axis = np.array(axis)
    positions = np.array(positions)

    axis_niggli, P = get_niggli_reduced_cell(
        axis,
        tol=symprec,
    )
    pos = positions @ np.linalg.inv(P)
    pos %= 1.0

    assert abs(round(np.linalg.det(P))) == 1
    return axis_niggli, pos, P


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


def enumerate_metric_symmetries(
    G: np.ndarray,
    is_chiral: bool = False,
    tol: float = 1e-5,
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

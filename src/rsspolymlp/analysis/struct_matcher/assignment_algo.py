import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import maximum_bipartite_matching


def sort_match_screen(pos_reps1, pos_reps2, pos_tol):
    # pos1: (N1, n_atoms, 3)
    # pos2: (N2, n_atoms, 3)
    pos1_sorted = np.sort(pos_reps1, axis=1)
    pos2_sorted = np.sort(pos_reps2, axis=1)

    diff = pos1_sorted[:, None, :, :] - pos2_sorted[None, :, :, :]
    diff -= np.round(diff)
    return np.all(np.abs(diff) <= pos_tol, axis=(2, 3))


def greedy_match(pos1, pos2, pos_tol):
    n = len(pos1)
    if n != len(pos2):
        return False

    used = np.zeros(n, dtype=bool)
    for i in range(n):
        diff = np.max(np.abs(pos2 - pos1[i]), axis=1)
        diff -= np.round(diff)
        diff[used] = np.inf
        j = np.argmin(np.abs(diff))
        if diff[j] >= pos_tol:
            return False
        used[j] = True
    return True


def bottleneck_assignment(pos_d):
    n = pos_d.shape[0]

    # Sort distinct candidate distances
    values = np.unique(pos_d)
    left, right = 0, len(values) - 1

    while left < right:
        mid = (left + right) // 2
        rows, cols = np.where(pos_d <= values[mid])
        graph = csr_array(
            (
                np.ones(len(rows), dtype=np.int8),
                (rows, cols),
            ),
            shape=pos_d.shape,
        )

        matching = maximum_bipartite_matching(graph, perm_type="column")
        if np.all(matching >= 0):
            right = mid
        else:
            left = mid + 1

    threshold = values[left]
    rows, cols = np.where(pos_d <= threshold)
    graph = csr_array(
        (
            np.ones(len(rows), dtype=np.int8),
            (rows, cols),
        ),
        shape=pos_d.shape,
    )

    matching = maximum_bipartite_matching(graph, perm_type="column")
    row = np.arange(n)
    col = matching
    return row, col

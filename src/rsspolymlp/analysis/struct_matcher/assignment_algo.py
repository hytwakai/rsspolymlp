import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import maximum_bipartite_matching


def sort_match_screen(
    pos_reps1: np.ndarray,
    pos_reps2: np.ndarray,
    elements1: np.ndarray,
    elements2: np.ndarray,
    axis_abc: np.ndarray,
    pos_tol: float,
):
    # pos1: (N1, n_atoms, 3)
    # pos2: (N2, n_atoms, 3)
    if len(elements1) != len(elements2):
        return np.zeros((len(pos_reps1), len(pos_reps2)), dtype=bool)
    unique_elements = np.unique(elements1)

    def sort_positions(pos_reps, elements):
        sorted_reps = []
        for pos in pos_reps:
            pos_sorted = []
            for el in unique_elements:
                pos_el = pos[elements == el]
                pos_el = np.sort(pos_el, axis=0)
                pos_sorted.append(pos_el)
            sorted_reps.append(np.concatenate(pos_sorted, axis=0))
        return np.asarray(sorted_reps)

    pos1_sorted = sort_positions(pos_reps1, elements1)
    pos2_sorted = sort_positions(pos_reps2, elements2)

    diff = pos1_sorted[:, None, :, :] - pos2_sorted[None, :, :, :]
    inv_axis_abc = np.asarray(1 / axis_abc).reshape(1, 1, 1, 3)
    diff -= axis_abc * np.round(diff * inv_axis_abc)
    return np.all(np.abs(diff) <= pos_tol, axis=(2, 3))


def greedy_match(
    pos1: np.ndarray,
    pos2: np.ndarray,
    elements1: np.ndarray,
    elements2: np.ndarray,
    axis_abc: np.ndarray,
    pos_tol: float,
):
    inv_axis_abc = 1 / axis_abc
    for el in np.unique(elements1):
        targets1 = np.where(elements1 == el)[0]
        targets2 = np.where(elements2 == el)[0]
        used = np.zeros(len(targets1), dtype=bool)
        for i in targets1:
            diff = np.abs(pos2[targets2] - pos1[i])
            diff -= axis_abc * np.round(inv_axis_abc * diff)
            diff_max = np.max(np.abs(diff), axis=1)
            diff[used] = np.inf
            j = np.argmin(diff_max)
            if diff_max[j] >= pos_tol:
                return False
            used[j] = True
    return True


def bottleneck_assignment(
    pos1: np.ndarray,
    pos2: np.ndarray,
    elements1: np.ndarray,
    elements2: np.ndarray,
    axis_abc: np.ndarray,
    pos_tol: float,
):
    inv_axis_abc = 1 / axis_abc
    pos_max_diffs = []
    for el in np.unique(elements1):
        targets1 = np.where(elements1 == el)[0]
        targets2 = np.where(elements2 == el)[0]
        pos1_sub = pos1[targets1]
        pos2_sub = pos2[targets2]

        diff = pos1_sub[:, None, :] - pos2_sub[None, :, :]
        diff -= axis_abc * np.round(diff * inv_axis_abc)
        pos_d = np.max(np.abs(diff), axis=2)
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
        pos_max_diffs.extend(pos_d[np.arange(n), matching].tolist())

    if max(pos_max_diffs) < pos_tol:
        return True
    else:
        return False

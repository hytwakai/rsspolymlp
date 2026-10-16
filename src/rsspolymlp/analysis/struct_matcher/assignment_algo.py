import time

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
    verbose: bool = False
):
    # pos1: (N1, n_atoms, 3)
    # pos2: (N2, n_atoms, 3)
    if len(elements1) != len(elements2):
        return np.zeros((len(pos_reps1), len(pos_reps2)), dtype=bool)
    unique_elements = np.unique(elements1)

    def sort_positions_axis(pos_reps, elements, axis):
        pos_reps = np.asarray(pos_reps)
        sorted_reps = np.empty_like(pos_reps[:, :, axis])
        start = 0
        for el in unique_elements:
            idx = np.flatnonzero(elements == el)
            n = len(idx)
            sorted_reps[:, start : start + n] = np.sort(
                pos_reps[:, idx, axis],
                axis=1,
            )
            start += n
        return sorted_reps

    matched = np.ones(
        (len(pos_reps1), len(pos_reps2)),
        dtype=bool,
    )
    for axis in range(3):
        rows, cols = np.where(matched)
        if len(rows) == 0:
            break

        t_start_s = time.time()
        pos1_axis = sort_positions_axis(pos_reps1, elements1, axis)
        pos2_axis = sort_positions_axis(pos_reps2, elements2, axis)
        if verbose:
            print(f"[sort_match_screen (axis {axis})]")
            print(f"  Sorting: {round((time.time() - t_start_s) * 1000, 2)} ms")

        t_start_m = time.time()
        diff = pos1_axis[rows] - pos2_axis[cols]
        diff -= axis_abc[axis] * np.round(diff / axis_abc[axis])
        matched[rows, cols] = np.all(np.abs(diff) <= pos_tol, axis=1)
        if verbose:
            print(f"  Matching: {round((time.time() - t_start_m) * 1000, 2)} ms")
            print(f"  Candidate matches: {np.count_nonzero(matched)}")
    return matched


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
    best_matching: bool = False,
    verbose: bool = False,
):
    inv_axis_abc = 1 / axis_abc
    for el in np.unique(elements1):
        targets1 = np.where(elements1 == el)[0]
        targets2 = np.where(elements2 == el)[0]
        pos1_sub = pos1[targets1]
        pos2_sub = pos2[targets2]

        t_start_d = time.time()
        diff = pos1_sub[:, None, :] - pos2_sub[None, :, :]
        diff -= axis_abc * np.round(diff * inv_axis_abc)
        pos_d = np.max(np.abs(diff), axis=2)
        n = pos_d.shape[0]
        if verbose:
            print("[bottleneck_assignment]")
            print(
                f"  Difference Calc.: {round((time.time() - t_start_d) * 1000, 2)} ms"
            )

        t_start_m = time.time()
        if best_matching:
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
            pos_max_diffs = pos_d[np.arange(n), matching].tolist()
            if verbose:
                print(f"  Matching: {round((time.time() - t_start_m) * 1000, 2)} ms")
            if max(pos_max_diffs) > pos_tol:
                return False
        else:
            t_start_m = time.time()
            rows, cols = np.where(pos_d <= pos_tol)
            graph = csr_array(
                (
                    np.ones(len(rows), dtype=np.int8),
                    (rows, cols),
                ),
                shape=pos_d.shape,
            )
            matching = maximum_bipartite_matching(graph, perm_type="column")
            if verbose:
                print(f"  Matching: {round((time.time() - t_start_m) * 1000, 2)} ms")
            if not np.all(matching >= 0):
                return False
    return True

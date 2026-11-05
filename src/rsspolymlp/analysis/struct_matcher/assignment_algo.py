import time

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_array
from scipy.sparse.csgraph import maximum_bipartite_matching


def sort_match_screen(
    pos_reps1: np.ndarray,
    pos_reps2: np.ndarray,
    elements1: np.ndarray,
    elements2: np.ndarray,
    axis_abc: np.ndarray,
    pos_tol: float,
    verbose: bool = False,
):
    # pos1: (N1, n_atoms, 3)
    # pos2: (N2, n_atoms, 3)
    if len(elements1) != len(elements2):
        return np.zeros((len(pos_reps1), len(pos_reps2)), dtype=bool)
    unique_elements = np.unique(elements1)

    def sort_positions_axis(pos_reps, elements, axis):
        pos_reps = np.asarray(pos_reps)
        reps_diff = np.empty_like(pos_reps[:, :, axis])
        start = 0
        for el in unique_elements:
            idx = np.flatnonzero(elements == el)
            sorted_reps = np.sort(pos_reps[:, idx, axis], axis=1)
            diff = np.roll(sorted_reps, -1, axis=1) - sorted_reps
            diff[:, -1] += axis_abc[axis]
            reps_diff[:, start : start + len(idx)] = np.sort(diff, axis=1)
            start += len(idx)
        return reps_diff

    matched = np.ones(
        (len(pos_reps1), len(pos_reps2)),
        dtype=bool,
    )
    if verbose:
        print("code: sort_match_screen (all candidates)")
        print(f"  Candidate matches: {np.count_nonzero(matched)}")
    for axis in range(3):
        rows, cols = np.where(matched)
        if len(rows) == 0:
            break

        t_start_s = time.time()
        pos1_axis = sort_positions_axis(pos_reps1, elements1, axis)
        pos2_axis = sort_positions_axis(pos_reps2, elements2, axis)
        if verbose:
            print(f"code: sort_match_screen (axis {axis})")
            print(f"  Sorting: {round((time.time() - t_start_s) * 1000, 2)} ms")

        t_start_m = time.time()
        diff = pos1_axis[rows] - pos2_axis[cols]
        diff -= axis_abc[axis] * np.round(diff / axis_abc[axis])
        matched[rows, cols] = np.all(np.abs(diff) <= pos_tol, axis=1)
        if verbose:
            print(f"  Matching: {round((time.time() - t_start_m) * 1000, 2)} ms")
            print(f"  Candidate matches: {np.count_nonzero(matched)}")
    return matched


def greedy_matching(pos_d, pos_tol):
    pos_d_res = pos_d.copy()
    pos_d_max = np.max(np.abs(pos_d), axis=2)

    used1 = np.zeros(pos_d_max.shape[0], dtype=bool)
    matching = np.full(len(pos_d_max), -1, dtype=int)
    while not np.all(used1):
        i, j = np.unravel_index(
            np.argmin(pos_d_max),
            pos_d_max.shape,
        )
        if pos_d_max[i, j] >= pos_tol:
            return None, None
        matching[i] = j
        pos_d_max[i, :] = np.inf
        pos_d_max[:, j] = np.inf
        used1[i] = True
    pos_d_res = pos_d_res[np.arange(len(matching)), matching]
    return pos_d_res, matching


def greedy_match(
    pos1: np.ndarray,
    pos2: np.ndarray,
    elements1: np.ndarray,
    elements2: np.ndarray,
    axis_abc: np.ndarray,
    pos_tol: float,
    verbose: bool = False,
):
    max_diff = 0
    t_start_g = time.time()
    for el in np.unique(elements1):
        targets1 = np.where(elements1 == el)[0]
        targets2 = np.where(elements2 == el)[0]
        pos1_sub = pos1[targets1]
        pos2_sub = pos2[targets2]

        col_ind = []
        inv_axis_abc = 1 / axis_abc
        pos_d = pos1_sub[:, None, :] - pos2_sub[None, :, :]
        pos_d -= axis_abc * np.round(pos_d * inv_axis_abc)
        for _ in range(10):
            pos_d_res, matching = greedy_matching(pos_d, pos_tol=pos_tol * 2)
            if matching is None:
                return False, None

            if np.max(np.abs(pos_d_res)) < pos_tol:
                if max_diff < np.max(np.abs(pos_d_res)):
                    max_diff = np.max(np.abs(pos_d_res))
                break
            if np.array_equal(matching, col_ind):
                return False, None

            col_ind = matching
            translation_correction, max_displacement = optimize_translation(pos_d_res)
            pos_d = pos_d - translation_correction

    if verbose:
        print("code: greedy_match")
        print(f"  Matching: {round((time.time() - t_start_g) * 1000, 2)} ms")

    return True, max_diff


def bottleneck_assignment(
    pos1: np.ndarray,
    pos2: np.ndarray,
    elements1: np.ndarray,
    elements2: np.ndarray,
    axis_abc: np.ndarray,
    pos_tol: float,
    verbose: bool = False,
):
    def get_matching(pos_tol, pos_d):
        rows, cols = np.where(pos_d <= pos_tol)
        graph = csr_array(
            (
                np.ones(len(rows), dtype=np.int8),
                (rows, cols),
            ),
            shape=pos_d.shape,
        )
        matching = maximum_bipartite_matching(graph, perm_type="column")
        return matching

    max_diff = 0
    inv_axis_abc = 1 / axis_abc
    for el in np.unique(elements1):
        targets1 = np.where(elements1 == el)[0]
        targets2 = np.where(elements2 == el)[0]
        pos1_sub = pos1[targets1]
        pos2_sub = pos2[targets2]

        t_start_d = time.time()
        diff = pos1_sub[:, None, :] - pos2_sub[None, :, :]
        diff -= axis_abc * np.round(diff * inv_axis_abc)

        if verbose:
            print("code: bottleneck_assignment")
            print(
                f"  Difference Calc.: {round((time.time() - t_start_d) * 1000, 2)} ms"
            )

        col_ind = []
        t_start_m = time.time()
        for n_iter in range(10):
            pos_d = np.max(np.abs(diff), axis=2)

            sorted_pos_d = np.sort(pos_d, axis=0)
            pos_d_minmax = np.max(sorted_pos_d[0])
            if pos_d_minmax > 2 * pos_tol:
                return False, None

            _pos_tol = pos_tol
            matching = get_matching(_pos_tol, pos_d)

            if np.all(matching >= 0):
                _max_diff = np.max(pos_d[np.arange(len(matching)), matching])
                if _max_diff > max_diff:
                    max_diff = _max_diff
                break
            else:
                _pos_tol *= 2
                matching = get_matching(_pos_tol, pos_d)

            if not np.all(matching >= 0):
                return False, None
            if np.array_equal(matching, col_ind):
                return False, None

            row_ind = np.arange(len(matching))
            col_ind = matching
            matched_diff = diff[row_ind, col_ind]
            translation_correction, max_displacement = optimize_translation(
                matched_diff
            )
            diff = diff - translation_correction
            diff -= axis_abc * np.round(diff * inv_axis_abc)

        if verbose:
            print(
                f"  Matching: {round((time.time() - t_start_m) * 1000 / (n_iter + 1), 2)} ms"
            )

    return True, max_diff


def optimize_translation(diffs):
    """Find a translation minimizing the maximum displacement."""

    def objective(translation):
        corrected = diffs - translation
        return np.max(np.linalg.norm(corrected, axis=1))

    diffs = np.asarray(diffs, dtype=float)
    initial_translation = np.mean(diffs, axis=0)
    result = minimize(
        objective,
        initial_translation,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "maxiter": 10000},
    )
    translation = result.x
    max_displacement = objective(translation)

    return translation, max_displacement

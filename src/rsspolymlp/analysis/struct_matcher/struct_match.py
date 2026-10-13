import math
import time
from contextlib import redirect_stdout

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import maximum_bipartite_matching

from rsspolymlp.analysis.struct_matcher.gen_allrep import (
    AllStructReps,
)
from rsspolymlp.analysis.struct_matcher.gen_redrep import (
    ReducedStructReps,
)


def struct_match(
    redreps_1: ReducedStructReps,
    redreps_2: ReducedStructReps,
    axis_tol: float = 0.01,
    pos_tol: float = 0.01,
    spg_match: bool = True,
    verbose: bool = False,
    output_file: str = "struct_matcher.yaml",
) -> bool:
    """
    Determine whether two sets of ReducedStructRep objects are structurally
    equivalent.

    This function compares all pairs of irreducible representations from the
    two input sets and checks if any pair matches within the specified lattice
    and position tolerances.
    Structures are compared only if they share the same space group number
    and identical element counts.

    Parameters
    ----------
    redreps_1 : ReducedStructReps
        Set of reduced representations from structure A.
    redreps_2 : ReducedStructReps
        Set of reduced representations from structure B.
    axis_tol : float, default=0.01
        Tolerance for lattice vector differences, computed using the squared
        L2 norm along each axis.
    pos_tol : float, default=0.01
        Tolerance for atomic position differences. Computed as the minimum of
        the maximum absolute deviation among all pairwise differences.

    Returns
    -------
    bool
        True if a matching pair of structures is found under the given
        tolerances, False otherwise.
    """
    struct_match = False
    axis_d_min = None
    pos_d_min = None
    min_axis_l2_norm = 1e8

    s_time = time.time()
    for st_1 in redreps_1.redrep_list:
        for st_2 in redreps_2.redrep_list:
            if struct_match or st_1.element_count != st_2.element_count:
                continue
            if spg_match and st_1.spg_number != st_2.spg_number:
                continue

            axis_d = (
                st_1.axis[:, None, :] - st_2.axis[None, :, :]
            )  # (N_symp1, N_symp2, 6)
            axis_d_flat = axis_d.reshape(-1, axis_d.shape[2])  # (N_symp1*N_symp2, 6)
            l2_norm = np.linalg.norm(axis_d_flat, axis=1)
            min_axis_l2_norm = min(min_axis_l2_norm, np.min(l2_norm))

            match_axis = l2_norm < axis_tol
            if not np.any(match_axis):
                continue

            pos_d = st_1.positions[:, None, :] - st_2.positions[None, :, :]
            pos_d_flat = pos_d.reshape(-1, pos_d.shape[2])
            max_abs = np.max(np.abs(pos_d_flat), axis=1)
            min_idx = np.argmin(max_abs[match_axis])

            pos_max_abs = max_abs[match_axis][min_idx]
            if pos_max_abs < pos_tol:
                struct_match = True

            if verbose and (pos_d_min is None or pos_d_min[0] > pos_max_abs):
                axis_l2_norm = l2_norm[match_axis][min_idx]
                i, j = divmod(np.where(match_axis)[0][min_idx], st_2.positions.shape[0])
                axis_d_min = [
                    axis_l2_norm,
                    [st_1.axis[i], st_2.axis[j], axis_d[i, j]],
                ]
                pos_d_min = [
                    pos_max_abs,
                    [
                        st_1.symprec_set[i],
                        st_1.positions[i],
                        st_2.symprec_set[j],
                        st_2.positions[j],
                        pos_d[i, j],
                    ],
                ]

    if verbose:
        print("Elapsed time:", np.round((time.time() - s_time) * 1000, 2), "ms")

        def log_axis_positions(symprec, axis, positions, round_axis=4, round_pos=3):
            print("    - symprec:", np.round(symprec, 5).tolist())
            print("      metric_tensor:", np.round(axis, round_axis).tolist())
            print("      positions:")
            for p in positions.reshape(3, -1).tolist():
                formatted = ",".join(f"{val:{round_pos + 3}.{round_pos}f}" for val in p)
                print(f"      - [{formatted}]")

        with open(output_file, "w") as f, redirect_stdout(f):
            print("tolerance:")
            print("  axis_tol:", axis_tol)
            print("  pos_tol:", pos_tol)
            print("structures:")
            for i, st_set in enumerate([redreps_1.redrep_list, redreps_2.redrep_list]):
                print(f"  - struct_No: {i + 1}")
                for st in st_set:
                    print("    spg_number:", st.spg_number)
                    print("    representations:")
                    for h, pos in enumerate(st.positions):
                        log_axis_positions(st.symprec_set[h], st.axis[h], pos)
            print("min_axis_l2_norm:", np.round(min_axis_l2_norm, 3))
            if axis_d_min is not None:
                x_axis = abs(float(axis_d_min[0]))
                if x_axis <= 1e-12:
                    round_axis = 6
                else:
                    round_axis = min(-math.floor(math.log10(x_axis)) + 1, 6)
                x_pos = abs(float(pos_d_min[0]))
                if x_pos <= 1e-12:
                    round_pos = 6
                else:
                    round_pos = min(-math.floor(math.log10(x_pos)) + 1, 6)

                print("difference_log:")
                print("  - axis_l2_norm:", np.round(axis_d_min[0], round_axis))
                print("    pos_max_abs:", np.round(pos_d_min[0], round_pos))
                print("    structure_1:")
                log_axis_positions(
                    pos_d_min[1][0],
                    axis_d_min[1][0],
                    pos_d_min[1][1],
                    round_axis,
                    round_pos,
                )
                print("    structure_2:")
                log_axis_positions(
                    pos_d_min[1][2],
                    axis_d_min[1][1],
                    pos_d_min[1][3],
                    round_axis,
                    round_pos,
                )
                print("    diffs:")
                log_axis_positions(
                    [], axis_d_min[1][2], pos_d_min[1][4], round_axis, round_pos
                )
            print("Match:", struct_match)
        print(f"{output_file} is generated.")

    return struct_match


def struct_match_allrep(
    allreps_1: AllStructReps,
    allreps_2: AllStructReps,
    axis_tol: float = 0.01,
    pos_tol: float = 0.01,
    spg_match: bool = True,
    verbose: bool = False,
) -> bool:

    judge_st_match = False
    min_axis_l2_norm = 1e8
    s_time = time.time()
    for st_1 in allreps_1.allrep_list:
        for st_2 in allreps_2.allrep_list:
            if judge_st_match or st_1.element_count != st_2.element_count:
                continue
            if spg_match and st_1.spg_number != st_2.spg_number:
                continue

            axis_d = st_1.axis - st_2.axis
            l2_norm = np.linalg.norm(axis_d)
            min_axis_l2_norm = min(min_axis_l2_norm, np.min(l2_norm))
            if l2_norm > axis_tol:
                continue

            judge_greedy = False
            for pos1 in st_1.positions[0]["pos_reps"]:
                for pos2_dicts in st_2.positions:
                    for pos2 in pos2_dicts["pos_reps"]:
                        if judge_greedy:
                            break
                        judge_greedy = greedy_match(pos1, pos2, pos_tol * 2)

            judge_Hopcroft_Karp = False
            if judge_greedy:
                for pos1 in st_1.positions[0]["pos_reps"]:
                    for pos2_dicts in st_2.positions:
                        for pos2 in pos2_dicts["pos_reps"]:
                            if judge_Hopcroft_Karp:
                                break
                            diff = pos1[:, None, :] - pos2[None, :, :]
                            diff -= np.round(diff)
                            pos_d = np.max(np.abs(diff), axis=2)
                            row, col = bottleneck_assignment(pos_d)
                            pos_max_abs = pos_d[row, col]
                            if np.max(pos_max_abs) < pos_tol:
                                print(np.max(pos_max_abs))
                                judge_Hopcroft_Karp = True
            if judge_Hopcroft_Karp:
                judge_st_match = True
    if verbose:
        print("Elapsed time:", np.round((time.time() - s_time) * 1000, 2), "ms")
    return judge_st_match


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

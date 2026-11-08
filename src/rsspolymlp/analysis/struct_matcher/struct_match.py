import time
from contextlib import redirect_stdout
from typing import Literal

import numpy as np

from rsspolymlp.analysis.struct_matcher.assignment_algo import (
    position_matching,
    sort_match_screen,
)
from rsspolymlp.analysis.struct_matcher.gen_struct_rep import (
    StructRep,
)


def struct_match(
    st_1: StructRep,
    st_2: StructRep,
    axis_tol: float = 0.01,
    pos_tol: float = 0.01,
    verbose: bool = False,
    output_file: str = "struct_matcher.yaml",
) -> bool:
    """
    Determine whether two sets of StructRep objects are structurally equivalent.

    This function compares all pairs of irreducible representations from the
    two input sets and checks if any pair matches within the specified lattice
    and position tolerances.
    Structures are compared only if they share the same space group number
    and identical element counts.

    Parameters
    ----------
    struct_rep_1 : StructRep
        StructRep object from structure A.
    struct_rep_2 : StructRep
        StructRep object from structure B.
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
    s_time = time.time()
    if st_1.element_count != st_2.element_count:
        return False

    redreps1 = st_1.reduced_posreps
    redreps2 = st_2.reduced_posreps
    axis1 = normalize_metric_tensor(st_1.metric_tensor)
    axis2 = normalize_metric_tensor(st_2.metric_tensor)
    axis_abc1 = np.sqrt(st_1.metric_tensor[:3])
    positions1 = np.stack(redreps1["positions"], axis=0)
    positions2 = np.stack(redreps2["positions"], axis=0)
    elements1 = st_1.elements[1:]
    elements2 = st_2.elements[1:]

    axis_d = axis1 - axis2
    axis_diff_max = np.max(np.abs(axis_d))
    axis_results = [axis_diff_max, [axis1, axis2, axis_d]]
    if not axis_diff_max < axis_tol:
        return False

    if len(st_1.elements) == 1:
        return True

    pos_results = None
    pos_max_abs = 1e6
    struct_match = False
    for pos1, tol1 in zip(positions1, redreps1["tol"]):
        if struct_match:
            break
        for pos2, tol2 in zip(positions2, redreps2["tol"]):
            pos_d = pos1 - pos2
            if np.all(np.abs(pos_d) < pos_tol):
                pos_max_abs = np.max(np.abs(pos_d))
                struct_match = True
            match_idxs = np.max(np.abs(pos_d.reshape(3, -1).T), axis=1) < pos_tol
            if np.any(match_idxs):
                target_idxs = np.where(
                    np.max(np.abs(pos_d.reshape(3, -1).T), axis=1) >= pos_tol
                )[0]
                _pos1 = pos1.reshape(3, -1).T[target_idxs]
                _pos2 = pos2.reshape(3, -1).T[target_idxs]
                _e1 = elements1[target_idxs]
                _e2 = elements2[target_idxs]
                judge, pos_max_abs = position_matching(
                    _pos1,
                    _pos2,
                    _e1,
                    _e2,
                    axis_abc1,
                    pos_tol,
                    assign_method="greedy",
                )
                if judge:
                    struct_match = True
                    break

    pos_results = [pos_max_abs, [tol1, pos1, tol2, pos2, pos_d]]
    if verbose:
        print("Elapsed time:", np.round((time.time() - s_time) * 1000, 2), "ms")
        print(f"{output_file} is generated.")

    return struct_match


def struct_match_allrep(
    st_1: StructRep,
    st_2: StructRep,
    axis_tol: float = 0.01,
    pos_tol: float = 0.01,
    spg_match: bool = True,
    assign_method: Literal["greedy", "HK"] = "greedy",
    verbose: bool = False,
) -> bool:
    axis1 = normalize_metric_tensor(st_1.metric_tensor)
    axis2 = normalize_metric_tensor(st_2.metric_tensor)
    axis_abc1 = np.sqrt(st_1.metric_tensor[:3])
    positions1 = st_1.all_posreps
    positions2 = st_2.all_posreps
    elements1 = st_1.elements[1:]
    elements2 = st_2.elements[1:]

    if st_1.element_count != st_2.element_count:
        return False

    axis_d = axis1 - axis2
    axis_diff_max = np.max(np.abs(axis_d))
    if not axis_diff_max < axis_tol:
        return False

    if len(st_1.elements) == 1:
        return True

    if verbose:
        print("--- Matching start ------")
    iter_match = 0
    t_match = 0

    pos1_array = np.array([positions1[0]])
    pos2_array = np.array(positions2)
    pre_judge = sort_match_screen(
        pos1_array,
        pos2_array,
        elements1,
        elements2,
        axis_abc1,
        pos_tol * 2,
        verbose=verbose,
    )
    row, col = np.where(pre_judge)

    judge_st_match = False
    for i2, j2 in zip(row, col):
        if judge_st_match:
            break
        t_start_match = time.time()
        judge_st_match, max_diff = position_matching(
            pos1_array[i2],
            pos2_array[j2],
            elements1,
            elements2,
            axis_abc1,
            pos_tol,
            assign_method=assign_method,
        )
        t_match += time.time() - t_start_match
        iter_match += 1
    if verbose:
        print("Iterations of matching:", iter_match)
        print("Max difference (Ang.):", max_diff)
        if iter_match > 0:
            print(
                f"Time average of matching: {round((t_match / iter_match) * 1000, 2)} ms"
            )
        print("--- Matching finished ---")
    return judge_st_match


def normalize_metric_tensor(metric_tensor):
    metric_tensor = np.asarray(metric_tensor)
    lengths = np.sqrt(metric_tensor[:3])

    metric_tensor[:3] /= 2 * lengths
    metric_tensor[3] /= lengths[1] + lengths[2]
    metric_tensor[4] /= lengths[0] + lengths[2]
    metric_tensor[5] /= lengths[0] + lengths[1]

    return metric_tensor


def output_log(
    axis_tol,
    pos_tol,
    strep_list1,
    strep_list2,
    axis_d_min,
    pos_d_min,
    output_file: str = "struct_matcher.yaml",
):

    def log_axis_positions(tol, axis, positions, round_axis=4, round_pos=3):
        print("    - tol:", np.round(tol, 5).tolist())
        print("      metric_tensor:", np.round(axis, round_axis).tolist())
        if len(positions) > 0:
            print("      positions:")
            for p in positions.reshape(3, -1).tolist():
                formatted = ",".join(f"{val:{round_pos + 3}.{round_pos}f}" for val in p)
                print(f"      - [{formatted}]")

    with open(output_file, "w") as f, redirect_stdout(f):
        print("tolerance:")
        print("  axis_tol:", axis_tol)
        print("  pos_tol:", pos_tol)
        print("structures:")
        for i, st_list in enumerate([strep_list1, strep_list2]):
            print(f"  - struct_No: {i + 1}")
            for st in st_list:
                print("    spg_number:", st.spg_number)
                print("    representations:")
                for h, pos in enumerate(st.reduced_reps["positions"]):
                    log_axis_positions(
                        st.reduced_reps["tol"][h],
                        st.reduced_reps["metric_tensors"][h],
                        pos,
                    )
        if axis_d_min is not None:
            x_pos = abs(float(pos_d_min[0])) if pos_d_min[0] is not None else 1e6
            print("difference_log:")
            print("  - axis_diff_max:", np.round(axis_d_min[0], 5))
            print("    pos_max_abs:", np.round(x_pos, 5))
            print("    structure_1:")
            log_axis_positions(
                pos_d_min[1][0],
                axis_d_min[1][0],
                pos_d_min[1][1],
            )
            print("    structure_2:")
            log_axis_positions(
                pos_d_min[1][2],
                axis_d_min[1][1],
                pos_d_min[1][3],
            )
            print("    diffs:")
            log_axis_positions([], axis_d_min[1][2], pos_d_min[1][4])
        print("Match:", struct_match)

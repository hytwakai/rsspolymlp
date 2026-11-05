import math
import time
from contextlib import redirect_stdout
from typing import Literal

import numpy as np

from rsspolymlp.analysis.struct_matcher.assignment_algo import (
    bottleneck_assignment,
    greedy_match,
    sort_match_screen,
)
from rsspolymlp.analysis.struct_matcher.gen_struct_rep import (
    StructReps,
)


def struct_match(
    struct_reps_1: StructReps,
    struct_reps_2: StructReps,
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
    struct_reps_1 : StructReps
        StructRep object from structure A.
    struct_reps_2 : StructReps
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
    struct_match = False
    axis_d_min = None
    pos_d_min = None
    min_axis_diff_max = 1e8
    strep_list1 = struct_reps_1.struct_rep_list
    strep_list2 = struct_reps_2.struct_rep_list

    s_time = time.time()
    for st_1 in strep_list1:
        for st_2 in strep_list2:
            if struct_match or st_1.element_count != st_2.element_count:
                continue

            redreps1 = st_1.reduced_reps
            redreps2 = st_2.reduced_reps
            axis1 = normalize_metric_tensors(redreps1["metric_tensors"])
            axis2 = normalize_metric_tensors(redreps2["metric_tensors"])
            positions1 = np.stack(redreps1["positions"], axis=0)
            positions2 = np.stack(redreps2["positions"], axis=0)
            elements1 = st_1.elements
            elements2 = st_2.elements

            struct_match = False
            for a1, pos1, tol1, axis_abc in zip(
                axis1, positions1, redreps1["tol"], redreps1["axis_abc"]
            ):
                if struct_match:
                    break
                for a2, pos2, tol2 in zip(axis2, positions2, redreps2["tol"]):
                    axis_d = a1 - a2
                    axis_diff_max = np.max(np.abs(axis_d))
                    if not axis_diff_max < axis_tol:
                        continue

                    if len(st_1.elements) == 1:
                        struct_match = True
                        break

                    pos_d = pos1 - pos2
                    if np.all(np.abs(pos_d) < pos_tol):
                        pos_max_abs = np.max(np.abs(pos_d))
                        struct_match = True
                    match_idxs = (
                        np.max(np.abs(pos_d.reshape(3, -1).T), axis=1) < pos_tol
                    )
                    if np.any(match_idxs):
                        target_idxs = np.where(
                            np.max(np.abs(pos_d.reshape(3, -1).T), axis=1) >= pos_tol
                        )[0]
                        _pos1 = pos1.reshape(3, -1).T[target_idxs]
                        _pos2 = pos2.reshape(3, -1).T[target_idxs]
                        _e1 = elements1[target_idxs + 1]
                        _e2 = elements2[target_idxs + 1]
                        judge, pos_max_abs = greedy_match(
                            _pos1,
                            _pos2,
                            _e1,
                            _e2,
                            axis_abc,
                            pos_tol,
                        )
                        if judge:
                            struct_match = True
                            break

            if verbose and (pos_d_min is None or pos_d_min[0] > pos_max_abs):
                axis_d_min = [axis_diff_max, [a1, a2, axis_d]]
                pos_d_min = [pos_max_abs, [tol1, pos1, tol2, pos2, pos_d]]

    if verbose:
        print("Elapsed time:", np.round((time.time() - s_time) * 1000, 2), "ms")

        def log_axis_positions(tol, axis, positions, round_axis=4, round_pos=3):
            print("    - tol:", np.round(tol, 5).tolist())
            print("      metric_tensor:", np.round(axis, round_axis).tolist())
            if len(positions) > 0:
                print("      positions:")
                for p in positions.reshape(3, -1).tolist():
                    formatted = ",".join(
                        f"{val:{round_pos + 3}.{round_pos}f}" for val in p
                    )
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
            print("min_axis_diff_max:", np.round(min_axis_diff_max, 3))
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
                log_axis_positions(
                    [], axis_d_min[1][2], pos_d_min[1][4]
                )
            print("Match:", struct_match)
        print(f"{output_file} is generated.")

    return struct_match


def struct_match_allrep(
    struct_reps_1: StructReps,
    struct_reps_2: StructReps,
    axis_tol: float = 0.01,
    pos_tol: float = 0.01,
    spg_match: bool = True,
    assign_method: Literal["greedy", "HK"] = "greedy",
    verbose: bool = False,
) -> bool:
    strep_list1 = struct_reps_1.struct_rep_list
    strep_list2 = struct_reps_2.struct_rep_list
    judge_st_match = False
    min_axis_diff_max = 1e8

    for st_1 in strep_list1:
        for st_2 in strep_list2:
            if judge_st_match or st_1.element_count != st_2.element_count:
                continue
            if spg_match and st_1.spg_number != st_2.spg_number:
                continue

            allreps1 = st_1.all_reps
            allreps2 = st_2.all_reps
            axis1 = normalize_metric_tensors(allreps1["metric_tensors"])
            axis2 = normalize_metric_tensors(allreps2["metric_tensors"])
            axis_abc_list1 = allreps1["axis_abc"]
            allpos_list1 = allreps1["positions"]
            allpos_list2 = allreps2["positions"]
            elements1 = np.array(st_1.elements[1:])
            elements2 = np.array(st_2.elements[1:])

            axis_d = axis1[:, None, :] - axis2[None, :, :]  # (N_symp1, N_symp2, 6)
            axis_d_flat = axis_d.reshape(-1, axis_d.shape[2])  # (N_symp1*N_symp2, 6)
            l2_norm = np.linalg.norm(axis_d_flat, axis=1)
            min_axis_diff_max = min(min_axis_diff_max, np.min(l2_norm))

            match_axis = l2_norm < axis_tol
            if not np.any(match_axis):
                continue

            if verbose:
                print("--- Matching start ------")
            iter_match = 0
            t_match = 0
            if len(st_1.elements) == 1:
                judge_st_match = True
            else:
                match_idx1, match_idx2 = np.where(
                    l2_norm.reshape(axis_d.shape[:2]) < axis_tol
                )
                for i1, j1 in zip(match_idx1, match_idx2):
                    axis_abc = axis_abc_list1[i1]
                    pos1 = allpos_list1[i1]
                    pos2 = allpos_list2[j1]
                    pos_reps1 = [pos1[0]["pos_reps"][0]]
                    pos_reps2 = [
                        pos for pos_dict in pos2 for pos in pos_dict["pos_reps"]
                    ]

                    pos1_array = np.array(pos_reps1)
                    pos2_array = np.array(pos_reps2)
                    pre_judge = sort_match_screen(
                        pos1_array,
                        pos2_array,
                        elements1,
                        elements2,
                        axis_abc,
                        pos_tol * 2,
                        verbose=verbose,
                    )
                    row, col = np.where(pre_judge)

                    judge_st_match = False
                    for i2, j2 in zip(row, col):
                        if judge_st_match:
                            break
                        t_start_match = time.time()
                        if assign_method == "greedy":
                            judge_st_match, max_diff = greedy_match(
                                pos_reps1[i2],
                                pos_reps2[j2],
                                elements1,
                                elements2,
                                axis_abc,
                                pos_tol,
                            )
                        elif assign_method == "HK":
                            judge_st_match, max_diff = bottleneck_assignment(
                                pos_reps1[i2],
                                pos_reps2[j2],
                                elements1,
                                elements2,
                                axis_abc,
                                pos_tol,
                            )
                        else:
                            raise ValueError(
                                f"Unknown assignment method: {assign_method}"
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


def normalize_metric_tensors(metric_tensors):
    metric_tensors = np.asarray(metric_tensors)
    lengths = np.sqrt(metric_tensors[:, :3])

    metric_tensors[:, :3] /= 2 * lengths
    metric_tensors[:, 3] /= lengths[:, 1] + lengths[:, 2]
    metric_tensors[:, 4] /= lengths[:, 0] + lengths[:, 2]
    metric_tensors[:, 5] /= lengths[:, 0] + lengths[:, 1]

    return metric_tensors

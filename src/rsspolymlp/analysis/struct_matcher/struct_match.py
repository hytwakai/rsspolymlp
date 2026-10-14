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
    redreps_1 : StructReps
        Set of reduced representations from structure A.
    redreps_2 : StructReps
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
    strep_list1 = struct_reps_1.struct_rep_list
    strep_list2 = struct_reps_2.struct_rep_list

    s_time = time.time()
    for st_1 in strep_list1:
        for st_2 in strep_list2:
            if struct_match or st_1.element_count != st_2.element_count:
                continue
            if spg_match and st_1.spg_number != st_2.spg_number:
                continue

            redreps1 = st_1.reduced_reps
            redreps2 = st_2.reduced_reps
            axis1 = np.stack(redreps1["metric_tensors"], axis=0)
            axis2 = np.stack(redreps2["metric_tensors"], axis=0)
            positions1 = np.stack(redreps1["positions"], axis=0)
            positions2 = np.stack(redreps2["positions"], axis=0)

            axis_d = axis1[:, None, :] - axis2[None, :, :]  # (N_symp1, N_symp2, 6)
            axis_d_flat = axis_d.reshape(-1, axis_d.shape[2])  # (N_symp1*N_symp2, 6)
            l2_norm = np.linalg.norm(axis_d_flat, axis=1)
            min_axis_l2_norm = min(min_axis_l2_norm, np.min(l2_norm))

            match_axis = l2_norm < axis_tol
            if not np.any(match_axis):
                continue

            if len(st_1.elements) == 1:
                struct_match = True
            else:
                pos_d = positions1[:, None, :] - positions2[None, :, :]
                pos_d_flat = pos_d.reshape(-1, pos_d.shape[2])
                max_abs = np.max(np.abs(pos_d_flat), axis=1)
                min_idx = np.argmin(max_abs[match_axis])

                pos_max_abs = max_abs[match_axis][min_idx]
                if pos_max_abs < pos_tol:
                    struct_match = True

                if verbose and (pos_d_min is None or pos_d_min[0] > pos_max_abs):
                    axis_l2_norm = l2_norm[match_axis][min_idx]
                    i, j = divmod(np.where(match_axis)[0][min_idx], positions2.shape[0])
                    axis_d_min = [
                        axis_l2_norm,
                        [axis1[i], axis2[j], axis_d[i, j]],
                    ]
                    pos_d_min = [
                        pos_max_abs,
                        [
                            redreps1["symprec"][i],
                            positions1[i],
                            redreps2["symprec"][j],
                            positions2[j],
                            pos_d[i, j],
                        ],
                    ]

    if verbose:
        print("Elapsed time:", np.round((time.time() - s_time) * 1000, 2), "ms")

        def log_axis_positions(symprec, axis, positions, round_axis=4, round_pos=3):
            print("    - symprec:", np.round(symprec, 5).tolist())
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
                            st.reduced_reps["symprec"][h],
                            st.reduced_reps["metric_tensors"][h],
                            pos,
                        )
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
    min_axis_l2_norm = 1e8

    s_time = time.time()
    for st_1 in strep_list1:
        for st_2 in strep_list2:
            if judge_st_match or st_1.element_count != st_2.element_count:
                continue
            if spg_match and st_1.spg_number != st_2.spg_number:
                continue

            allreps1 = st_1.all_reps
            allreps2 = st_2.all_reps
            axis1 = np.stack(allreps1["metric_tensors"], axis=0)
            axis2 = np.stack(allreps2["metric_tensors"], axis=0)
            allpos_list1 = allreps1["positions"]
            allpos_list2 = allreps2["positions"]

            axis_d = axis1[:, None, :] - axis2[None, :, :]  # (N_symp1, N_symp2, 6)
            axis_d_flat = axis_d.reshape(-1, axis_d.shape[2])  # (N_symp1*N_symp2, 6)
            l2_norm = np.linalg.norm(axis_d_flat, axis=1)
            min_axis_l2_norm = min(min_axis_l2_norm, np.min(l2_norm))

            match_axis = l2_norm < axis_tol
            if not np.any(match_axis):
                continue

            match_idx1, match_idx2 = np.where(
                l2_norm.reshape(axis_d.shape[:2]) < axis_tol
            )
            for i, j in zip(match_idx1, match_idx2):
                pos1 = allpos_list1[i]
                pos2 = allpos_list2[j]
                pos_reps1 = pos1[0]["pos_reps"]
                pos_reps2 = [pos for pos_dict in pos2 for pos in pos_dict["pos_reps"]]
                pos1_array = np.array(pos_reps1)
                pos2_array = np.array(pos_reps2)
                pre_judge = sort_match_screen(pos1_array, pos2_array, pos_tol)
                row, col = np.where(pre_judge)

                judge_st_match = False
                if assign_method == "greedy":
                    for i, j in zip(row, col):
                        if judge_st_match:
                            break
                        judge_st_match = greedy_match(
                            pos_reps1[i], pos_reps2[j], pos_tol
                        )
                elif assign_method == "HK":
                    for i, j in zip(row, col):
                        if judge_st_match:
                            break
                        diff = pos_reps1[i][:, None, :] - pos_reps2[j][None, :, :]
                        diff -= np.round(diff)
                        pos_d = np.max(np.abs(diff), axis=2)
                        row, col = bottleneck_assignment(pos_d)
                        pos_max_abs = pos_d[row, col]
                        if np.max(pos_max_abs) < pos_tol:
                            print(np.max(pos_max_abs))
                            judge_st_match = True
                else:
                    raise ValueError(f"Unknown assignment method: {assign_method}")
    if verbose:
        print("Elapsed time:", np.round((time.time() - s_time) * 1000, 2), "ms")
    return judge_st_match

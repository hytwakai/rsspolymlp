import itertools
import time
from collections import Counter
from contextlib import redirect_stdout
from typing import Literal

import numpy as np

from rsspolymlp.analysis.struct_matcher.assignment_algo import (
    position_matching,
    sort_match_screen,
)
from rsspolymlp.analysis.struct_matcher.chiral_spg import is_chiral
from rsspolymlp.analysis.struct_matcher.dataclass import StructRep


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
    if Counter(st_1.elements) != Counter(st_2.elements):
        return False

    redreps1 = st_1.reduced_posreps
    redreps2 = st_2.reduced_posreps
    norm_metric_tensor1 = normalize_metric_tensor(st_1.metric_tensor)
    norm_metric_tensor2 = normalize_metric_tensor(st_2.metric_tensor)
    axis_abc1 = np.sqrt(st_1.metric_tensor[:3])
    positions1 = np.stack(redreps1["positions"], axis=0)
    positions2 = np.stack(redreps2["positions"], axis=0)
    elements1 = st_1.elements[1:]
    elements2 = st_2.elements[1:]

    judge = False
    axis_d = norm_metric_tensor1 - norm_metric_tensor2
    axis_diff_max = np.max(np.abs(axis_d))
    if axis_diff_max < axis_tol:
        judge = True

    if not judge:
        if verbose:
            output_log(
                axis_tol,
                pos_tol,
                st_1,
                st_2,
                judge,
                axis_diff_max,
                output_file=output_file,
            )
        return False

    if len(st_1.elements) == 1:
        judge = True
        if verbose:
            output_log(
                axis_tol,
                pos_tol,
                st_1,
                st_2,
                judge,
                axis_diff_max,
                output_file=output_file,
            )
        return True

    pos_max_abs = 1e6
    judge = False
    for pos1, tol1 in zip(positions1, redreps1["tol"]):
        if judge:
            break
        for pos2, tol2 in zip(positions2, redreps2["tol"]):
            pos_d = pos1 - pos2
            if np.all(np.abs(pos_d) < pos_tol):
                pos_max_abs = np.max(np.abs(pos_d))
                judge = True
            match_idxs = np.max(np.abs(pos_d.reshape(3, -1).T), axis=1) < pos_tol
            if np.any(match_idxs):
                target_idxs = np.where(
                    np.max(np.abs(pos_d.reshape(3, -1).T), axis=1) >= pos_tol
                )[0]
                _pos1 = pos1.reshape(3, -1).T[target_idxs]
                _pos2 = pos2.reshape(3, -1).T[target_idxs]
                _e1 = elements1[target_idxs]
                _e2 = elements2[target_idxs]
                _judge, pos_max_abs = position_matching(
                    _pos1,
                    _pos2,
                    _e1,
                    _e2,
                    axis_abc1,
                    pos_tol,
                    assign_method="greedy",
                )
                if _judge:
                    judge = True
                    break

    pos_results = [pos_max_abs, [tol1, pos1, tol2, pos2, pos_d]]
    if verbose:
        print("Elapsed time:", np.round((time.time() - s_time) * 1000, 2), "ms")
        output_log(
            axis_tol,
            pos_tol,
            st_1,
            st_2,
            judge,
            axis_diff_max,
            pos_diff=pos_results,
            output_file=output_file,
        )
        print(f"{output_file} is generated.")

    return judge


def struct_match_allrep(
    st_1: StructRep,
    st_2: StructRep,
    axis_tol: float = 0.01,
    pos_tol: float = 0.01,
    assign_method: Literal["greedy", "HK"] = "greedy",
    match_metric_tensor: bool = True,
    verbose: bool = False,
) -> bool:
    norm_metric_tensor1 = normalize_metric_tensor(st_1.metric_tensor)
    norm_metric_tensor2 = normalize_metric_tensor(st_2.metric_tensor)
    axis_abc1 = np.sqrt(st_1.metric_tensor[:3])
    positions1 = np.array(st_1.all_posreps)
    positions2 = np.array(st_2.all_posreps)
    elements1 = st_1.elements[1:]
    elements2 = st_2.elements[1:]

    if Counter(st_1.elements) != Counter(st_2.elements):
        return False

    axis_d = norm_metric_tensor1 - norm_metric_tensor2
    axis_diff_max = np.max(np.abs(axis_d))
    if not axis_diff_max < axis_tol:
        if match_metric_tensor:
            m1 = st_1.metric_tensor
            m2 = st_2.metric_tensor
            G_1 = [[m1[0], m1[5], m1[4]], [m1[5], m1[1], m1[3]], [m1[4], m1[3], m1[2]]]
            G_2 = [[m2[0], m2[5], m2[4]], [m2[5], m2[1], m2[3]], [m2[4], m2[3], m2[2]]]
            transformation_matrices = matching_metric_tensor(
                G_1,
                G_2,
                spg_number=st_1.spg_number,
                tol=axis_tol,
            )
            if len(transformation_matrices) == 0:
                return False
            else:
                all_positions = []
                for trans_mat in transformation_matrices:
                    pos1 = positions1 @ np.linalg.inv(trans_mat)
                    all_positions.append(pos1)
                all_positions = np.concatenate(all_positions, axis=0)
                positions1 = all_positions
        else:
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


def matching_metric_tensor(
    G_1: np.ndarray,
    G_2: np.ndarray,
    spg_number: int,
    tol: float = 1e-3,
) -> list[np.ndarray]:
    """Enumerate unimodular integer matrices satisfying M @ G_1 @ M.T = G_2."""

    def _enumerate_integer_vectors(
        G: np.ndarray,
        q: float,
        tol: float = 1e-5,
    ) -> np.ndarray:
        """Enumerate integer vectors satisfying v.T @ G @ v ~= q."""
        G = np.asarray(G, dtype=float)

        # Upper bound from the minimum eigenvalue.
        lambda_min = np.linalg.eigvalsh(G)[0]
        if lambda_min <= 0:
            raise ValueError("G must be positive definite.")

        vectors = []
        bound = int(np.ceil(np.sqrt((q + tol) / lambda_min)))
        for v in itertools.product(range(-bound, bound + 1), repeat=3):
            v = np.asarray(v, dtype=int)
            if abs(v @ G @ v - q) <= tol:
                vectors.append(v)

        return np.asarray(vectors, dtype=int).reshape(-1, 3)

    G_1 = np.asarray(G_1, dtype=float)
    G_2 = np.asarray(G_2, dtype=float)
    if not (np.allclose(G_1, G_1.T, atol=tol) and np.allclose(G_2, G_2.T, atol=tol)):
        raise ValueError("G must be symmetric.")

    norm_a = G_1[0, 0] ** 0.5
    norm_b = G_1[1, 1] ** 0.5
    norm_c = G_1[2, 2] ** 0.5
    e_aa = tol * 2 * norm_a
    e_bb = tol * 2 * norm_b
    e_cc = tol * 2 * norm_c
    e_bc = tol * (norm_b + norm_c)
    e_ac = tol * (norm_a + norm_c)
    e_ab = tol * (norm_a + norm_b)

    # Enumerate candidate column vectors.
    candidates = [
        _enumerate_integer_vectors(G_1, G_2[i, i], [e_aa, e_bb, e_cc][i])
        for i in range(3)
    ]
    transformation_matrices = []

    # Enumerate combinations of the three columns.
    for m1, m2, m3 in itertools.product(*candidates):
        if abs(m1 @ G_1 @ m2 - G_2[0, 1]) > e_ab:
            continue
        if abs(m1 @ G_1 @ m3 - G_2[0, 2]) > e_ac:
            continue
        if abs(m2 @ G_1 @ m3 - G_2[1, 2]) > e_bc:
            continue
        M = np.array([m1, m2, m3])

        det = round(np.linalg.det(M))
        if abs(det) != 1:
            continue
        if is_chiral(spg_number) and det == -1:
            continue
        transformation_matrices.append(M)

    return transformation_matrices


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
    strep_1,
    strep_2,
    judge,
    axis_diff,
    pos_diff=None,
    output_file: str = "struct_matcher.yaml",
):

    def log_positions(tol, positions, round_pos=3):
        print("    - tol:", np.round(tol, 5).tolist())
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
        for i, st in enumerate([strep_1, strep_2]):
            print(f"  - struct_No: {i + 1}")
            print("    metric_tensor:", np.round(st.metric_tensor, 4).tolist())
            print("    spg_number:", st.spg_number)
            print("    representations:")
            for h, pos in enumerate(st.reduced_posreps["positions"]):
                log_positions(
                    st.reduced_posreps["tol"][h],
                    pos,
                )
        print("Match:", judge)
        print("difference_log:")
        print("  - axis_diff_max:", np.round(axis_diff, 5))
        if pos_diff is not None:
            x_pos = abs(float(pos_diff[0])) if pos_diff[0] is not None else 1e6
            print("    pos_max_abs:", np.round(x_pos, 5))
            print("    structure_1:")
            log_positions(
                pos_diff[1][0],
                pos_diff[1][1],
            )
            print("    structure_2:")
            log_positions(
                pos_diff[1][2],
                pos_diff[1][3],
            )
            print("    diffs:")
            log_positions([], pos_diff[1][4])

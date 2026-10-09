import math
import re
from contextlib import redirect_stdout
from typing import Optional

import numpy as np

from rsspolymlp.analysis.struct_matcher.gen_redrep import (
    ReducedStructReps,
    generate_redreps_parallel,
)


class UniqueStructIdentifier:

    def __init__(self):
        self.unique_str: list[ReducedStructReps] = []  # Store unique structures
        self.unique_str_prop: list[dict] = []  # Store unique structure properties
        self.unique_str_keep: list[list[ReducedStructReps]] = []
        self.unique_str_prop_keep: list[list[dict]] = []

    def identify_duplicate_struct(
        self,
        reduced_reps: ReducedStructReps,
        other_properties: Optional[dict] = None,
        axis_tol: float = 0.01,
        pos_tol: float = 0.01,
        keep_unique: bool = False,
    ):
        """
        Identify and manage duplicate structures.
        A structure is considered a duplicate if it matches an existing structure based on
        equivalence of the reduced crystal structure representation.

        Parameters
        ----------
        reduced_reps : ReducedStructReps
            The structure to be compared and registered if unique.
        other_properties : dict, optional
            Additional metadata associated with the structure.
        energy_diff : float
            Energy tolerance used in energy-based duplicate detection.

        Returns
        -------
        is_unique : bool
            True if the structure is unique.
        is_change_struct : bool
            True if the existing structure was replaced due to higher symmetry.
        """

        is_unique = True
        is_change_struct = False
        if other_properties is None:
            other_properties = {}

        for idx, _uniq_str in enumerate(self.unique_str):
            uniq_str_list = self.unique_str_keep[idx] if keep_unique else [_uniq_str]
            for uniq_str in uniq_str_list:
                if struct_match(
                    uniq_str,
                    reduced_reps,
                    axis_tol=axis_tol,
                    pos_tol=pos_tol,
                ):
                    is_unique = False
                    if self._spg_count(reduced_reps.spg_list) > self._spg_count(
                        uniq_str.spg_list
                    ) or (
                        self._spg_count(reduced_reps.spg_list)
                        == self._spg_count(uniq_str.spg_list)
                    ):
                        is_change_struct = True
                    break

            if not is_unique:
                break

        if not is_unique:
            if reduced_reps.struct_path not in self.unique_str[idx].dupstr_paths:
                self.unique_str[idx].dupstr_paths.add(reduced_reps.struct_path)
            if is_change_struct:
                # Update duplicate count and replace with better data if necessary
                reduced_reps.dupstr_paths = self.unique_str[idx].dupstr_paths
                reduced_reps.struct_tag = self.unique_str[idx].struct_tag
                self.unique_str[idx] = reduced_reps
                self.unique_str_prop[idx] = other_properties
            if keep_unique:
                self.unique_str_keep[idx].append(reduced_reps)
                self.unique_str_prop_keep[idx].append(other_properties)
        else:
            self.unique_str.append(reduced_reps)
            self.unique_str_prop.append(other_properties)
            if keep_unique:
                self.unique_str_keep.append([reduced_reps])
                self.unique_str_prop_keep.append([other_properties])

        if is_unique and len(self.unique_str) % 500 == 0:
            print(f"Reached {len(self.unique_str)} unique structures.")

        return is_unique, is_change_struct

    def _spg_count(self, spg_list):
        """Extract and sum space group counts from a list of space group strings."""
        return sum(
            int(re.search(r"\((\d+)\)", s).group(1))
            for s in spg_list
            if re.search(r"\((\d+)\)", s)
        )

    def _initialize_unique_structs(
        self,
        unique_structs: list[ReducedStructReps],
        unique_str_prop: Optional[list[dict]] = None,
    ):
        """Initialize unique structures and their associated properties."""
        self.unique_str = unique_structs
        if unique_str_prop is None:
            self.unique_str_prop = [{} for _ in unique_structs]
        else:
            self.unique_str_prop = unique_str_prop


def identify_unique_structure(
    struct_lists: list[dict],
    axis_tol: float = 0.01,
    pos_tol: float = 0.01,
    keep_unique: bool = False,
    num_process: int = -1,
    backend: str = "loky",
    primitive_symprecs: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
    redrep_symprecs: list[float] = [1e-4, 1e-2, 1e-1],
    standardize_axis: bool = False,
    cartesian_coords: bool = True,
    refine_cell: bool = False,
    pre_analyzer: Optional[UniqueStructIdentifier] = None,
    verbose: bool = False,
):
    """
    Parameters
    ----------
    struct_lists : list of dict
        A list of dictionaries, where each dictionary contains a single structure information.
        Each dictionary must include "struct_path" keys:
            - "struct_path": path of POSCAR format file
        Optional keys:
            - "structure": PolymlpStructure object
            - "struct_tag" (optional): structure identifier (e.g., structure number)
    """
    if pre_analyzer is None:
        analyzer = UniqueStructIdentifier()
    else:
        analyzer = pre_analyzer

    if verbose:
        print("   - Converting reduced crystal structure representation...")
    redreps_list = generate_redreps_parallel(
        struct_lists,
        num_process=num_process,
        backend=backend,
        primitive_symprecs=primitive_symprecs,
        redrep_symprecs=redrep_symprecs,
        standardize_axis=standardize_axis,
        cartesian_coords=cartesian_coords,
        refine_cell=refine_cell,
    )

    if verbose:
        print("   - Eliminating duplicate structures...")
    for idx, redreps in enumerate(redreps_list):
        analyzer.identify_duplicate_struct(
            redreps,
            other_properties=struct_lists[idx],
            axis_tol=axis_tol,
            pos_tol=pos_tol,
            keep_unique=keep_unique,
        )
    return analyzer


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
        First set of symmetry-reduced structures (e.g., from structure A).
    redreps_1 : ReducedStructReps
        Second set of symmetry-reduced structures (e.g., from structure B).
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
    for st_1 in redreps_1.reduced_struct_set:
        for st_2 in redreps_2.reduced_struct_set:
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
            for i, st_set in enumerate(
                [redreps_1.reduced_struct_set, redreps_2.reduced_struct_set]
            ):
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

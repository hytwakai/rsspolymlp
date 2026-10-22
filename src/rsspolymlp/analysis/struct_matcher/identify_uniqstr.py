import re
import time
from typing import Literal, Optional

from rsspolymlp.analysis.struct_matcher.gen_struct_rep import (
    StructRepGenerator,
    StructReps,
)
from rsspolymlp.analysis.struct_matcher.struct_match import (
    struct_match,
    struct_match_allrep,
)


class UniqueStructIdentifier:

    def __init__(
        self,
        axis_tol: float = 0.01,
        pos_tol: float = 0.01,
        assign_method: Literal["redrep", "greedy", "HK", "pymatgen"] = "redrep",
        keep_unique: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize UniqueStructIdentifier class.

        Parameters
        ----------
        axis_tol : float
            Tolerance for comparing lattice axes.
        pos_tol : float
            Tolerance for comparing atomic positions.
        assign_method : {"redrep", "greedy", "HK"}
            Method used to match atomic positions.
        keep_unique : bool
            Whether to retain all structures.
        verbose : bool
            Whether to print detailed information.
        """
        self.axis_tol = axis_tol
        self.pos_tol = pos_tol
        self.assign_method = assign_method
        self.keep_unique = keep_unique
        self.verbose = verbose
        if self.assign_method == "pymatgen":
            from rsspolymlp.utils.pymatgen_utils import PymatUtil

            self.pymat = PymatUtil()

        self.unique_str: list[StructReps] = []  # Store unique structures
        self.unique_str_prop: list[dict] = []  # Store unique structure properties
        self.unique_str_keep: list[list[StructReps]] = []
        self.unique_str_prop_keep: list[list[dict]] = []
        self.time_match = 0
        self.iter_match = 0

    def identify_duplicate_struct(
        self,
        struct_reps: StructReps,
        other_properties: Optional[dict] = None,
    ):
        """
        Identify and manage duplicate structures.
        A structure is considered a duplicate if it matches an existing structure
        based on equivalence of the crystal structure representation.

        Parameters
        ----------
        struct_reps : StructReps
            The structure to be compared and registered if unique.
        other_properties : Optional[dict]
            Additional metadata associated with the structure.

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
            uniq_str_list = (
                self.unique_str_keep[idx] if self.keep_unique else [_uniq_str]
            )
            for uniq_str in uniq_str_list:
                t_start = time.time()
                if self.assign_method == "redrep":
                    judge_duplicate = struct_match(
                        uniq_str,
                        struct_reps,
                        axis_tol=self.axis_tol,
                        pos_tol=self.pos_tol,
                    )
                elif self.assign_method == "pymatgen":
                    st1 = self.pymat.polymlp_to_pymat_st(uniq_str.original_polymlp_st)
                    st2 = self.pymat.polymlp_to_pymat_st(
                        struct_reps.original_polymlp_st
                    )
                    judge_duplicate = self.pymat.match_str(st1, st2)
                else:
                    judge_duplicate = struct_match_allrep(
                        uniq_str,
                        struct_reps,
                        axis_tol=self.axis_tol,
                        pos_tol=self.pos_tol,
                        assign_method=self.assign_method,
                        verbose=self.verbose,
                    )
                self.time_match += time.time() - t_start
                self.iter_match += 1

                if judge_duplicate:
                    is_unique = False
                    if self._spg_count(struct_reps.spg_list) > self._spg_count(
                        uniq_str.spg_list
                    ) or (
                        self._spg_count(struct_reps.spg_list)
                        == self._spg_count(uniq_str.spg_list)
                    ):
                        is_change_struct = True
                    break

            if not is_unique:
                break

        if not is_unique:
            if struct_reps.struct_path not in self.unique_str[idx].dupstr_paths:
                self.unique_str[idx].dupstr_paths.add(struct_reps.struct_path)
            if is_change_struct:
                # Update duplicate count and replace with better data if necessary
                struct_reps.dupstr_paths = self.unique_str[idx].dupstr_paths
                struct_reps.struct_tag = self.unique_str[idx].struct_tag
                self.unique_str[idx] = struct_reps
                self.unique_str_prop[idx] = other_properties
            if self.keep_unique:
                self.unique_str_keep[idx].append(struct_reps)
                self.unique_str_prop_keep[idx].append(other_properties)
        else:
            self.unique_str.append(struct_reps)
            self.unique_str_prop.append(other_properties)
            if self.keep_unique:
                self.unique_str_keep.append([struct_reps])
                self.unique_str_prop_keep.append([other_properties])

        if is_unique and len(self.unique_str) % 500 == 0:
            print(f"Reached {len(self.unique_str)} unique structures.")

        return is_unique, is_change_struct

    def _spg_count(self, spg_list):
        """Extract and sum space group counts from a list of space group strings."""
        sum_n_spg = 0
        for spg in spg_list:
            if spg is None:
                sum_n_spg += 1
            else:
                sum_n_spg += int(re.search(r"\((\d+)\)", spg).group(1))
        return sum_n_spg

    def _initialize_unique_structs(
        self,
        unique_structs: list[StructReps],
        unique_str_prop: Optional[list[dict]] = None,
    ):
        """Initialize unique structures and their associated properties."""
        self.unique_str = unique_structs
        if unique_str_prop is None:
            self.unique_str_prop = [{} for _ in unique_structs]
        else:
            self.unique_str_prop = unique_str_prop


def identify_unique_structure(
    struct_list: list[dict],
    axis_tol: float = 0.01,
    pos_tol: float = 0.01,
    assign_method: Literal["redrep", "greedy", "HK", "pymatgen"] = "redrep",
    keep_unique: bool = False,
    num_process: int = -1,
    backend: str = "loky",
    primitive_symprecs: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
    redrep_symprecs: list[float] = [1e-4, 1e-2, 1e-1],
    normalize_axis: bool = False,
    refine_cell: bool = False,
    pre_analyzer: Optional[UniqueStructIdentifier] = None,
    verbose: bool = False,
):
    """
    Parameters
    ----------
    struct_list : list of dict
        A list of dictionaries, each containing information about a structure.
        Required keys:
            - "struct_path": Path to the structure file.
        Optional keys:
            - "poscar_path": Path to the POSCAR file.
            - "structure": PolymlpStructure object.
            - "axis": Lattice vectors.
            - "positions": Atomic positions.
            - "elements": Chemical element symbols.
            - "struct_tag": Structure identifier (e.g., structure number).
            - "dupstr_paths": List of paths to duplicate structure files.
    """
    if pre_analyzer is None:
        analyzer = UniqueStructIdentifier(
            axis_tol=axis_tol,
            pos_tol=pos_tol,
            assign_method=assign_method,
            keep_unique=keep_unique,
        )
    else:
        analyzer = pre_analyzer

    if verbose:
        print("   - Converting reduced crystal structure representation...")
    generator = StructRepGenerator(
        primitive_symprecs=primitive_symprecs,
        redrep_symprecs=redrep_symprecs,
        normalize_axis=normalize_axis,
        refine_cell=refine_cell,
    )
    redreps_list = generator.generate_struct_reps_parallel(
        struct_list,
        num_process=num_process,
        backend=backend,
    )

    if verbose:
        print("   - Eliminating duplicate structures...")
    for idx, redreps in enumerate(redreps_list):
        analyzer.identify_duplicate_struct(
            redreps,
            other_properties=struct_list[idx],
        )
    return analyzer

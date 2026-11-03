import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np

from rsspolymlp.analysis.struct_matcher.primitive_transform import (
    PrimitiveCell,
    get_primitive_cell,
)
from rsspolymlp.analysis.struct_matcher.redrep_position import PositionRepReducer
from rsspolymlp.common.interface_spglib import SymCell
from rsspolymlp.common.property import PropUtil


@dataclass
class StructRep:
    reduced_reps: dict
    all_reps: dict
    elements: np.ndarray
    element_count: Counter[str]
    spg_number: int


@dataclass
class StructReps:
    struct_rep_list: list[StructRep]
    original_axis: np.ndarray
    original_positions: np.ndarray
    original_elements: np.ndarray
    spg_list: list[str]
    tol_sm: list[float]
    tol_redpos: list[float]
    normalize_axis: bool
    struct_path: Optional[str]
    struct_tag: Optional[int]
    dupstr_paths: set[str]


class StructRepGenerator:

    def __init__(
        self,
        tol_sm: list[float] = [1e-2],
        tol_redpos: list[float] = [1e-4, 1e-2, 1e-1],
        normalize_axis: bool = False,
        verbose: bool = False,
    ):
        self.tol_sm = tol_sm
        self.tol_redpos = tol_redpos
        self.normalize_axis = normalize_axis
        self.verbose = verbose

        self.iter_refine = 0
        self.time_refine = 0
        self.iter_redrep = 0
        self.time_redrep = 0

    def generate_struct_reps(
        self,
        axis: np.ndarray,
        positions: np.ndarray,
        elements: np.ndarray,
        struct_path: str,
        struct_tag: Optional[str] = None,
        dupstr_paths: Optional[set[str]] = None,
    ) -> tuple[StructReps, dict]:
        """
        Parameters
        ----------
        axis : (3, 3) array_like
            Lattice vectors defining the unit cell in Cartesian coordinates.
            Each row represents one lattice vector (a, b, or c).
            Equivalent to np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]).
        positions : (N, 3) array_like
            Fractional atomic coordinates within the unit cell.
            Each row represents the coordinates of one atom.
        elements : (N,) array_like
            Chemical element symbols corresponding to the atomic positions.
        struct_path: str
            Path to the structure file.
        struct_tag: str, Optional
            Structure identifier (e.g., structure number).
        dupstr_paths: set[str], Optional
            List of paths to duplicate structure files.
        """

        objprop = PropUtil(axis, positions)
        spg_list = objprop.analyze_space_group(elements)

        primitive_st_set, spg_number_set, log_refine = self.generate_primitive_cells(
            axis,
            positions,
            elements,
        )
        if primitive_st_set == []:
            return None

        struct_rep_list = []
        for i, primitive_st in enumerate(primitive_st_set):
            tol_redpos = sorted(
                self.tol_redpos,
                key=lambda x: x if isinstance(x, (int, float)) else sum(x) / len(x),
            )
            struct_rep, log_redrep = self.generate_struct_rep(
                primitive_st.axis,
                primitive_st.positions,
                primitive_st.elements,
                spg_number_set[i],
            )
            struct_rep_list.append(struct_rep)

        calc_log = {
            "iter_refine": log_refine[0],
            "time_refine": log_refine[1],
            "iter_redrep": log_redrep[0],
            "time_redrep": log_redrep[1],
        }

        if dupstr_paths is None:
            dupstr_paths = set([struct_path])
        return (
            StructReps(
                struct_rep_list=struct_rep_list,
                original_axis=axis,
                original_positions=positions,
                original_elements=np.asarray(elements, dtype=str),
                spg_list=spg_list,
                tol_sm=self.tol_sm,
                tol_redpos=tol_redpos,
                normalize_axis=self.normalize_axis,
                struct_path=struct_path,
                struct_tag=struct_tag,
                dupstr_paths=dupstr_paths,
            ),
            calc_log,
        )

    def generate_primitive_cells(
        self,
        axis: np.ndarray,
        positions: np.ndarray,
        elements: np.ndarray,
    ) -> tuple[list[PrimitiveCell], list[int], tuple]:
        """Generate primitive cells of a given structure under different symmetry tolerances."""
        primitive_st_set = []
        spg_number_set = []
        iter_refine = 0
        time_refine = 0
        for tol in self.tol_sm:
            t_start = time.time()
            primitive_cell = get_primitive_cell(
                axis=axis, positions=positions, elements=elements, tol=tol
            )
            symutil = SymCell(
                primitive_cell.axis,
                primitive_cell.positions,
                primitive_cell.elements,
                symprec=tol,
            )
            spg_str = symutil.get_spacegroup()
            if spg_str is None:
                spg_number = 1
            else:
                spg_number = int(re.search(r"\((\d+)\)", spg_str).group(1))

            primitive_st_set.append(primitive_cell)
            spg_number_set.append(spg_number)
            iter_refine += 1
            time_refine += time.time() - t_start

        self.iter_refine += iter_refine
        self.time_refine += time_refine

        return primitive_st_set, spg_number_set, (iter_refine, time_refine)

    def generate_struct_rep(
        self,
        axis: np.ndarray,
        positions: np.ndarray,
        elements: np.ndarray,
        spg_number: int,
    ) -> tuple[StructRep, tuple]:
        """Generate an StructRep object for a structure under different symmetry tolerances."""
        reduced_reps = {
            "metric_tensors": [],
            "axis_abc": [],
            "positions": [],
            "tol": [],
        }
        all_reps = {
            "metric_tensors": [],
            "axis_abc": [],
            "positions": [],
            "tol": [],
        }
        iter_redrep = 0
        time_redrep = 0

        tol_set = self.tol_redpos
        for tol in tol_set:
            if isinstance(tol, float):
                input_tol = [tol] * 3
            t_start = time.time()
            reducer = PositionRepReducer(
                axis,
                positions,
                elements,
                spg_number,
                tol=input_tol,
                normalize_axis=self.normalize_axis,
                verbose=self.verbose,
            )
            res = reducer.get_reduced_structure_representation()
            metric_tensor = res["metric_tensor"]
            axis_abc = res["axis_abc"]
            red_posrep = res["reduced_positions"]
            all_posrep = res["all_positions"]
            sorted_elements = res["elements"]

            iter_redrep += 1
            time_redrep += time.time() - t_start

            unique_metric = True
            for mt_ref in reduced_reps["metric_tensors"]:
                diffs = np.abs(mt_ref - metric_tensor)
                if np.all(diffs < 1e-4):
                    unique_metric = False
                    break
            unique_pos = True
            if not unique_metric and len(red_posrep) > 0:
                for ps_ref in reduced_reps["positions"]:
                    diffs = np.abs(ps_ref - red_posrep)
                    if np.all(diffs < 1e-4):
                        unique_pos = False
                        break
            if unique_metric:
                all_reps["metric_tensors"].append(metric_tensor)
                all_reps["axis_abc"].append(axis_abc)
                all_reps["positions"].append(all_posrep)
                all_reps["tol"].append(tol)
            if unique_metric or unique_pos:
                reduced_reps["metric_tensors"].append(metric_tensor)
                reduced_reps["axis_abc"].append(axis_abc)
                reduced_reps["positions"].append(red_posrep)
                reduced_reps["tol"].append(tol)

        self.iter_redrep += iter_redrep
        self.time_redrep += time_redrep

        return StructRep(
            reduced_reps=reduced_reps,
            all_reps=all_reps,
            elements=sorted_elements,
            element_count=Counter(sorted_elements),
            spg_number=spg_number,
        ), (iter_redrep, time_redrep)

    def generate_struct_reps_parallel(
        self,
        struct_list: list[dict],
        num_process: int = -1,
        backend: str = "loky",
        verbose: bool = False,
    ) -> list[StructReps]:
        """
        Generate a list of StructReps objects.

        Parameters
        ----------
        struct_list : list of dict
            A list of dictionaries, each containing information about a structure.
            Required keys:
                - "struct_path": Path to the structure file.
                - "axis": Lattice vectors.
                - "positions": Atomic positions.
                - "elements": Chemical element symbols.
            Optional keys:
                - "struct_tag": Structure identifier (e.g., structure number).
                - "dupstr_paths": List of paths to duplicate structure files.
        num_process : int, default=-1
            The number of parallel jobs. -1 means using all available processors.
        backend : str, default="loky"
            Backend used by joblib.
        """
        if num_process == 1:
            redreps_list = []
            for res in struct_list:
                redreps_list.append(
                    self.generate_struct_reps(
                        axis=res["axis"],
                        positions=res["positions"],
                        elements=res["elements"],
                        struct_path=res["struct_path"],
                        struct_tag=res.get("struct_tag", None),
                        dupstr_paths=res.get("dupstr_paths", None),
                    )[0]
                )
        else:
            if self.verbose:
                joblib_verbose = 10
            else:
                joblib_verbose = 0
            joblib_results = joblib.Parallel(
                n_jobs=num_process, backend=backend, verbose=joblib_verbose
            )(
                joblib.delayed(self.generate_struct_reps)(
                    axis=res["axis"],
                    positions=res["positions"],
                    elements=res["elements"],
                    struct_path=res["struct_path"],
                    struct_tag=res.get("struct_tag", None),
                    dupstr_paths=res.get("dupstr_paths", None),
                )
                for res in struct_list
            )
            redreps_list = []
            for res in joblib_results:
                if res[0] is not None:
                    redreps_list.append(res[0])
                self.iter_refine += res[1]["iter_refine"]
                self.time_refine += res[1]["time_refine"]
                self.iter_redrep += res[1]["iter_redrep"]
                self.time_redrep += res[1]["time_redrep"]

        if verbose:
            print("code: struct_matcher.gen_struct_rep")
            print("  time:")
            print("    Process   Iterations  Avg. time (ms)  Total time (s)")
            for name, iterations, total_time in [
                ("Refine", self.iter_refine, self.time_refine),
                ("Reduce", self.iter_redrep, self.time_redrep),
            ]:
                avg_time = total_time * 1000 / iterations
                print(
                    f"    {name:<7} {iterations:>12} "
                    f"{avg_time:>15.2f} {total_time:>15.2f}"
                )
        return redreps_list

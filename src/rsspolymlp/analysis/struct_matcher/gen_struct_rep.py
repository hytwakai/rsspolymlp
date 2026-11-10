import time
from typing import Optional

import joblib
import numpy as np

from rsspolymlp.analysis.struct_matcher.dataclass import StructRep
from rsspolymlp.analysis.struct_matcher.niggli_reduce import niggli_reduce
from rsspolymlp.analysis.struct_matcher.position_redrep import (
    PositionRepReducer,
)
from rsspolymlp.analysis.struct_matcher.primitive_transform import (
    PrimitiveCell,
    get_primitive_cell,
)
from rsspolymlp.analysis.struct_matcher.struct_match import matching_metric_tensor
from rsspolymlp.common.interface_spglib import SymCell
from rsspolymlp.common.property import PropUtil


class StructRepGenerator:

    def __init__(
        self,
        tol_sm: float = 1e-2,
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

    def generate_struct_rep(
        self,
        axis: np.ndarray,
        positions: np.ndarray,
        elements: np.ndarray,
        struct_path: str,
        struct_tag: Optional[str] = None,
        dupstr_paths: Optional[set[str]] = None,
    ) -> tuple[StructRep, dict]:
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

        primitive_st, spg_number, log_refine = self.generate_reduced_cell(
            axis,
            positions,
            elements,
        )
        if primitive_st is None:
            return None

        G = primitive_st.axis @ primitive_st.axis.T
        metric_tensor = (G[0, 0], G[1, 1], G[2, 2], G[1, 2], G[0, 2], G[0, 1])
        reduced_posreps, all_posreps, sorted_elements, log_redrep = (
            self.generate_reduced_position(
                primitive_st.axis,
                primitive_st.positions,
                primitive_st.elements,
                spg_number,
            )
        )

        calc_log = {
            "iter_refine": log_refine[0],
            "time_refine": log_refine[1],
            "iter_redrep": log_redrep[0],
            "time_redrep": log_redrep[1],
        }

        if dupstr_paths is None:
            dupstr_paths = set([struct_path])
        return (
            StructRep(
                axis=primitive_st.axis,
                metric_tensor=metric_tensor,
                reduced_posreps=reduced_posreps,
                all_posreps=all_posreps,
                elements=sorted_elements,
                spg_number=spg_number,
                spg_list=spg_list,
                original_axis=axis,
                original_positions=positions,
                original_elements=np.asarray(elements, dtype=str),
                tol_sm=self.tol_sm,
                tol_redpos=self.tol_redpos,
                normalize_axis=self.normalize_axis,
                struct_path=struct_path,
                struct_tag=struct_tag,
                dupstr_paths=dupstr_paths,
            ),
            calc_log,
        )

    def generate_reduced_cell(
        self,
        axis: np.ndarray,
        positions: np.ndarray,
        elements: np.ndarray,
    ) -> tuple[list[PrimitiveCell], list[int], tuple]:
        """Generate primitive cells of a given structure under different symmetry tolerances."""
        iter_refine = 0
        time_refine = 0

        t_start = time.time()
        primitive_cell = get_primitive_cell(
            axis=axis, positions=positions, elements=elements, tol=self.tol_sm
        )
        symutil = SymCell(
            primitive_cell.axis,
            primitive_cell.positions,
            primitive_cell.elements,
            symprec=self.tol_sm,
        )
        spg_number = symutil.get_spg_number()

        axis_niggli, positions_niggli = niggli_reduce(
            axis=primitive_cell.axis,
            positions=primitive_cell.positions,
            tol=self.tol_sm,
        )
        primitive_cell.axis = axis_niggli
        primitive_cell.positions = positions_niggli

        iter_refine += 1
        time_refine += time.time() - t_start

        self.iter_refine += iter_refine
        self.time_refine += time_refine

        return primitive_cell, spg_number, (iter_refine, time_refine)

    def generate_reduced_position(
        self,
        axis: np.ndarray,
        positions: np.ndarray,
        elements: np.ndarray,
        spg_number: int,
    ) -> tuple[StructRep, tuple]:
        """Generate an StructRep object for a structure under different symmetry tolerances."""
        G = axis @ axis.T
        transformation_matrices = matching_metric_tensor(
            G, G, spg_number, tol=self.tol_sm
        )

        reduced_posreps = {"positions": [], "tol": []}
        iter_redrep = 0
        time_redrep = 0
        tol_set = self.tol_redpos
        for tol in tol_set:
            t_start = time.time()
            reducer = PositionRepReducer(
                axis,
                positions,
                elements,
                normalize_axis=self.normalize_axis,
                transformation_matrices=transformation_matrices,
                verbose=self.verbose,
            )
            res = reducer.get_reduced_position_representation()
            red_posrep = res["reduced_positions"]
            all_posreps = res["all_positions"]
            sorted_elements = res["elements"]

            iter_redrep += 1
            time_redrep += time.time() - t_start

            unique_pos = True
            if len(red_posrep) > 0:
                for ps_ref in reduced_posreps["positions"]:
                    diffs = np.abs(ps_ref - red_posrep)
                    if np.all(diffs < 1e-4):
                        unique_pos = False
                        break
            if unique_pos:
                reduced_posreps["positions"].append(red_posrep)
                reduced_posreps["tol"].append(tol)

        self.iter_redrep += iter_redrep
        self.time_redrep += time_redrep

        return reduced_posreps, all_posreps, sorted_elements, (iter_redrep, time_redrep)

    def generate_struct_rep_parallel(
        self,
        struct_list: list[dict],
        num_process: int = -1,
        backend: str = "loky",
        verbose: bool = False,
    ) -> list[StructRep]:
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
            struct_rep_list = []
            for res in struct_list:
                struct_rep_list.append(
                    self.generate_struct_rep(
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
                joblib.delayed(self.generate_struct_rep)(
                    axis=res["axis"],
                    positions=res["positions"],
                    elements=res["elements"],
                    struct_path=res["struct_path"],
                    struct_tag=res.get("struct_tag", None),
                    dupstr_paths=res.get("dupstr_paths", None),
                )
                for res in struct_list
            )
            struct_rep_list = []
            for res in joblib_results:
                if res[0] is not None:
                    struct_rep_list.append(res[0])
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
        return struct_rep_list

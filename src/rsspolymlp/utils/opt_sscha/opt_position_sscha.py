import glob
import os
import shutil
from typing import Optional

import numpy as np

from pypolymlp.api.pypolymlp_sscha import PypolymlpSSCHA
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.structure_utils import refine_positions
from pypolymlp.utils.symfc_utils import construct_basis_fractional_coordinates
from rsspolymlp.utils.opt_sscha.sscha_property import SSCHAProperty


def optimization_fc2_position(
    cell: PolymlpStructure,
    pot: str,
    temperature: float,
    init_fc_algorithm: str = "harmonic",
    yamlfile: Optional[str] = None,
    fc2file: Optional[str] = None,
    tol: float = 0.01,
    mesh: list = [10, 10, 10],
    max_iter: int = 15,
    mixing: float = 0.5,
    n_samples: Optional[int] = None,
    threshold_disp: float = 0.01,
    verbose: bool = True,
):
    """Optimization of FCs and atomic positions using an iterative approach of SSCHA."""
    print("-------- sscha runs start --------", flush=True)
    for iteration in range(10):
        print(f"****** sscha iteration {iteration+1} ******", flush=True)
        sscha = PypolymlpSSCHA(verbose=verbose)
        sscha._unitcell = cell
        sscha._supercell_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        sscha.set_polymlp(pot)

        if fc2file is None:
            _init_fc_algorithm = init_fc_algorithm
        else:
            _init_fc_algorithm = "file"

        while True:
            sscha = sscha.run(
                temp=temperature,
                n_samples_init=n_samples,
                n_samples_final=n_samples,
                tol=tol,
                mesh=mesh,
                max_iter=max_iter,
                mixing=mixing,
                init_fc_algorithm=_init_fc_algorithm,
                init_fc_file=fc2file,
            )
            n_samples = sscha.sscha_params.n_samples_init
            converge_sscha = sscha.logs[-1].converge

            if converge_sscha is False:
                print(
                    "---- increasing the number of sample structures ----", flush=True
                )
                n_samples = n_samples * 2
                fc2file = "./fc2.hdf5"
            else:
                break

            if n_samples > 50000:
                print(
                    "<< Number of sample stuctures for FC optimization exceeds 50,000 >>"
                )
                return "not converged", cell, n_samples

        sscha_dir = glob.glob("./sscha/*")[0]
        if fc2file is None:
            os.makedirs("init", exist_ok=True)
            for name in ["fc2.hdf5", "sscha_results.yaml", "total_dos.dat"]:
                shutil.copy(os.path.join(sscha_dir, name), "init")
            fc2file = "./init/fc2.hdf5"
            yamlfile = "./init/sscha_results.yaml"
        else:
            fc2file = f"{sscha_dir}/fc2.hdf5"
            yamlfile = f"{sscha_dir}/sscha_results.yaml"

        _basis_f_withfc = construct_basis_fractional_coordinates(cell)
        if _basis_f_withfc is not None:
            for disp_iter in range(10):
                prp_sscha = SSCHAProperty(
                    cell,
                    pot,
                    n_samples=n_samples * 3,
                    yamlfile=yamlfile,
                    fc2file=fc2file,
                )
                move_eq_position, max_displacement = prp_sscha.position_opt(
                    _basis_f_withfc
                )
                cell.positions += move_eq_position
                cell = refine_positions(cell)
                if max_displacement < threshold_disp:
                    print(
                        "< Convergence of atomic positions is achieved >",
                        flush=True,
                    )
        else:
            print("< Convergence of sscha optimization is achieved >", flush=True)
            return "success", cell, n_samples

        if max_displacement < threshold_disp:
            print("disp_iteration =", disp_iter)
            if disp_iter == 0:
                print("< Convergence of sscha optimization is achieved >", flush=True)
                print("Positions :", flush=True)
                print(cell.positions, flush=True)
            return "success", cell, n_samples

    if iteration == 9:
        print(
            "< Maximum number of iterations of the sscha optimization has been exceeded >",
            flush=True,
        )
        return "not converged", cell, n_samples

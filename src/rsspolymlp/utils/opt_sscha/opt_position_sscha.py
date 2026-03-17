import glob
import os
import shutil
from typing import Optional

import numpy as np

from pypolymlp.api.pypolymlp_sscha import PypolymlpSSCHA
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.structure_utils import refine_positions
from pypolymlp.utils.symfc_utils import construct_basis_fractional_coordinates
from rsspolymlp.common.property import PropUtil
from rsspolymlp.utils.opt_sscha.sscha_property import SSCHAProperty


def fc2_optimization(
    cell: PolymlpStructure,
    pot: str,
    temperature: float,
    init_fc_algorithm: str = "harmonic",
    fc2file: Optional[str] = None,
    tol: float = 0.01,
    mesh: list = [10, 10, 10],
    max_iter: int = 15,
    mixing: float = 0.5,
    n_samples: Optional[int] = None,
    verbose: bool = True,
):
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
            precondition=False,
        )
        n_samples = sscha.sscha_params.n_samples_init
        converge_sscha = sscha.logs[-1].converge

        if converge_sscha is True:
            print("<< Convergence of FCs is achieved >>", flush=True)
            return True, n_samples
        else:
            print("---- increasing the number of sample structures ----", flush=True)
            n_samples = n_samples * 2
            fc2file = "./fc2.hdf5"

        if n_samples > 50000:
            print("<< Number of sample stuctures for FC optimization exceeds 50,000 >>")
            return False, n_samples


def position_optimization(
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
    print("------ position optimization run start ------", flush=True)
    for iteration in range(10):
        print(
            f"****** position optimization iteration {iteration+1} ******", flush=True
        )

        message = fc2_optimization(
            cell=cell,
            pot=pot,
            temperature=temperature,
            init_fc_algorithm=init_fc_algorithm,
            fc2file=fc2file,
            tol=tol,
            mesh=mesh,
            max_iter=max_iter,
            mixing=mixing,
            n_samples=n_samples,
            verbose=verbose,
        )
        if not message:
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
            objprop = PropUtil(cell.axis.T, cell.positions.T)
            least_distance = objprop.least_distance
            threshold_disp_ang = least_distance * threshold_disp
            print("threshold_disp_ang:", round(threshold_disp_ang, 4), "(Ang.)")

            for disp_iter in range(10):
                print("Number of samples:", n_samples * 3)
                prp_sscha = SSCHAProperty(
                    yamlfile,
                    fc2file,
                    cell=cell,
                    pot=pot,
                    n_samples=n_samples * 3,
                )
                prp_sscha.run()
                move_eq_position, max_displacement = prp_sscha.position_opt()
                cell.positions += move_eq_position
                cell = refine_positions(cell)
                if max_displacement < threshold_disp_ang:
                    print(
                        "<< Convergence of atomic positions is achieved >>",
                        flush=True,
                    )
                    break
        else:
            print("<< Convergence of LS optimization is achieved >>", flush=True)
            return "success", cell, n_samples

        if max_displacement < threshold_disp_ang:
            print("disp_iteration =", disp_iter)
            if disp_iter == 0:
                print("<< Convergence of LS optimization is achieved >>", flush=True)
                print("Positions :", flush=True)
                print(cell.positions, flush=True)
                break

    if iteration < 9:
        print("****** Final FC optimization run start ******", flush=True)
        message = fc2_optimization(
            cell=cell,
            pot=pot,
            temperature=temperature,
            init_fc_algorithm=init_fc_algorithm,
            fc2file=fc2file,
            tol=tol,
            mesh=mesh,
            max_iter=max_iter,
            mixing=mixing,
            n_samples=n_samples,
            verbose=True,
        )
        if not message:
            return "not converged", cell, n_samples
        return "success", cell, n_samples
    else:
        print(
            "<< Maximum number of iterations of the LS optimization has been exceeded >>",
            flush=True,
        )
        return "not converged", cell, n_samples

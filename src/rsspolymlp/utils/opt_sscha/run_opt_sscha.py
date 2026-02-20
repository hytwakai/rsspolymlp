import argparse
import glob
import os
import shutil
import subprocess
from typing import Optional

import numpy as np

from pypolymlp.api.pypolymlp_sscha import PypolymlpSSCHA
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.spglib_utils import SymCell
from pypolymlp.utils.supercell_utils import get_supercell
from rsspolymlp.utils.opt_sscha.opt_geometry_sscha import GeometryOptimization
from rsspolymlp.utils.opt_sscha.opt_position_sscha import optimization_fc2_position
from rsspolymlp.utils.opt_sscha.sscha_property import SSCHAProperty


def estimate_n_samples(
    cell: PolymlpStructure,
    pot: str,
    yamlfile: str,
    fc2file: str,
    gtol: float,
    n_samples: Optional[int] = None,
):
    if n_samples is None:
        n_samples = 2000

    prp_sscha = SSCHAProperty(
        cell,
        pot,
        n_samples=n_samples,
        yamlfile=yamlfile,
        fc2file=fc2file,
    )
    sd_e, sd_f, sd_s, sd_derivatives_f, sd_derivatives_s = (
        prp_sscha.estimate_derivatives_standard_deviation()
    )
    if sd_derivatives_f is None:
        max_dirivative = np.max(sd_derivatives_s)
    else:
        max_dirivative = np.max((np.max(sd_derivatives_f), np.max(sd_derivatives_s)))

    target_n_samples = int((max_dirivative * 5 / gtol) ** 2)
    max_sd_f = np.max(sd_f)
    max_sd_s = np.max(sd_s)
    se_e_average = sd_e / (target_n_samples**0.5)
    se_f_average = max_sd_f / (target_n_samples**0.5)
    se_s_average = max_sd_s / (target_n_samples**0.5)
    n_atoms = cell.n_atoms[0]

    print("units:")
    print("  sd_derivatives_f: meV/cell")
    print("  sd_derivatives_s: meV/Ang.")
    print("  sd_e: meV/cell")
    print("  max_sd_f: meV/Ang.")
    print("  max_sd_s: meV/cell")
    print("  se_e_average: meV/cell")
    print("  se_f_average: meV/Ang.")
    print("  se_s_average: meV/cell")
    print("properties:")
    print("  target_n_samples:", target_n_samples)
    if sd_derivatives_f is not None:
        print("  sd_derivatives_f:", sd_derivatives_f * 1000)
    print("  sd_derivatives_s:", sd_derivatives_s * 1000)
    print("  sd_e:", sd_e * 1000 / n_atoms)
    print("  max_sd_f:", max_sd_f * 1000 / n_atoms)
    print("  max_sd_s:", max_sd_s * 1000 / n_atoms)
    print("  se_e_average:", se_e_average * 1000 / n_atoms)
    print("  se_f_average:", se_f_average * 1000 / n_atoms)
    print("  se_s_average:", se_s_average * 1000 / n_atoms)


def run_opt_sscha(
    poscar: str,
    supercell: list = [2, 2, 2],
    pot: str = "polymlp.yaml",
    pressure: float = 0,
    temperature: float = 300,
    init_fc_algorithm: str = "harmonic",
    yamlfile: Optional[str] = None,
    fc2file: Optional[str] = None,
    tol: float = 0.01,
    dtol: float = 0.01,
    gtol: float = 0.01,
    n_samples: Optional[int] = None,
    max_iter: int = 15,
    mixing: float = 0.5,
    mesh: list = [10, 10, 10],
    relax_cell: bool = True,
    relax_volume: bool = True,
    relax_positions: bool = False,
    method: str = "BFGS",
):
    cell = Poscar(poscar).structure
    supercell_matrix = [
        [supercell[0], 0, 0],
        [0, supercell[1], 0],
        [0, 0, supercell[2]],
    ]
    cell = get_supercell(cell, supercell_matrix)

    os.makedirs("init", exist_ok=True)
    if fc2file is not None:
        fc2file = "./init/fc2.hdf5"
        if not os.path.samefile(fc2file, "./init/fc2.hdf5"):
            shutil.copy(fc2file, "./init/fc2.hdf5")
    if yamlfile is not None:
        yamlfile = "./init/sscha_results.yaml"
        if not os.path.samefile(yamlfile, "./init/sscha_results.yaml"):
            shutil.copy(yamlfile, "./init/sscha_results.yaml")

    max_iter_go = 10
    for iter_n in range(max_iter_go):
        os.makedirs(f"iteration_{iter_n+1}", exist_ok=True)
        judge, cell, n_samples = optimization_fc2_position(
            cell=cell,
            pot=pot,
            temperature=temperature,
            init_fc_algorithm=init_fc_algorithm,
            yamlfile=yamlfile,
            fc2file=fc2file,
            tol=tol,
            mesh=mesh,
            max_iter=max_iter,
            mixing=mixing,
            n_samples=n_samples,
            threshold_disp=dtol,
        )
        if judge == "not converged":
            print("< Convergence of fc optimization is failed >", flush=True)
            return

        sscha_dir = glob.glob("./sscha/*")[0]
        os.makedirs(f"iteration_{iter_n+1}", exist_ok=True)
        for name in ["fc2.hdf5", "sscha_results.yaml", "total_dos.dat"]:
            shutil.copy(os.path.join(sscha_dir, name), f"iteration_{iter_n+1}")
        yamlfile = f"iteration_{iter_n+1}/sscha_results.yaml"
        fc2file = f"iteration_{iter_n+1}/fc2.hdf5"

        if not relax_cell and not relax_volume and not relax_positions:
            print("No degree of freedom to be optimized.")
            break

        minobj = GeometryOptimization(
            cell=cell,
            relax_cell=relax_cell,
            relax_volume=relax_volume,
            relax_positions=relax_positions,
            sscha_opt=True,
            pressure=pressure,
            pot=pot,
            verbose=True,
        )
        minobj.run(
            method=method,
            gtol=gtol,
            maxiter=10,
            n_samples=n_samples,
            yamlfile=yamlfile,
            fc2file=fc2file,
        )

        if not minobj.relax_cell:
            print("Residuals (force):")
            print(minobj.residual_forces.T)
        else:
            res_f, res_s = minobj.residual_forces
            print("Residuals (force):")
            print(res_f.T)
            print("Residuals (stress):")
            print(res_s)

        print("Final structure")
        minobj.print_structure()
        minobj.write_poscar()

        sym = SymCell(poscar_name="POSCAR_eqm", symprec=1e-5)
        cell_copy = sym.refine_cell()
        minobj.structure = cell_copy
        minobj.write_poscar(filename="POSCAR_eqm.refine")
        spg_sets = []
        for tol in [10**-5, 10**-4, 10**-3, 10**-2]:
            sym = SymCell(poscar_name="POSCAR_eqm.refine", symprec=tol)
            spg = sym.get_spacegroup()
            spg_sets.append(spg)
            print(f"Space group ({tol}):", spg, flush=True)
        print("Space group set:", flush=True)
        print(spg_sets, flush=True)

        subprocess.run(
            f"mv POSCAR_eqm POSCAR_eqm.refine iteration_{iter_n+1}", shell=True
        )
        poscar = f"iteration_{iter_n+1}/POSCAR_eqm.refine"
        cell = Poscar(poscar).structure
        supercell_matrix = [
            [supercell[0], 0, 0],
            [0, supercell[1], 0],
            [0, 0, supercell[2]],
        ]
        cell = get_supercell(cell, supercell_matrix)

        iteration_go = minobj.n_iter
        print("itaration_go =", iteration_go, flush=True)
        if minobj.max_residual < gtol:
            if (iteration_go <= 1) and iter_n > 0:
                print("<< Convergence of sscha geometry optimization is achieved >>")
                break
        if iter_n == max_iter_go - 1:
            print("<< Maximum number of relaxation iterations has been exceeded >>")

        if iter_n == 4:
            print("---- increasing the number of sample structures ----", flush=True)
            n_samples_go = n_samples * 2

    shutil.copy(poscar, "./POSCAR_eqm.refine")
    print("------ final sscha calculation ------", flush=True)
    sscha = PypolymlpSSCHA(verbose=True)
    sscha._unitcell = cell
    sscha._supercell_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sscha.set_polymlp(pot)

    sscha = sscha.run(
        temp=temperature,
        n_samples_init=n_samples_go,
        n_samples_final=n_samples * 2,
        tol=tol,
        mesh=mesh,
        max_iter=max_iter,
        mixing=mixing,
        init_fc_algorithm="file",
        init_fc_file=fc2file,
    )
    for name in ["fc2.hdf5", "sscha_results.yaml", "total_dos.dat"]:
        shutil.copy(os.path.join(sscha_dir, name), "./")
    print("<<< Finished >>>")


if __name__ == "__main__":

    np.set_printoptions(legacy="1.21")
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate_n_samp", action="store_true")
    parser.add_argument("-p", "--poscar", type=str, default=None, help="poscar file")
    parser.add_argument(
        "--supercell", nargs=3, type=int, default=None, help="Supercell size"
    )
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default="polymlp.lammps",
        help="polymlp file",
    )
    parser.add_argument("--pressure", type=float, default=0, help="Pressure (GPa)")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature (K)")
    # Parameters for SSCHA calculation
    parser.add_argument(
        "--init",
        choices=["harmonic", "const", "random", "file"],
        default="harmonic",
        help="Initial FCs",
    )
    parser.add_argument(
        "--yamlfile", type=str, default=None, help="force constants file"
    )
    parser.add_argument(
        "--fc2file", type=str, default=None, help="force constants file"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.01,
        help="Tolerance parameter for FC optimization",
    )
    parser.add_argument(
        "--dtol",
        type=float,
        default=0.01,
        help="Tolerance parameter for atomic position optimization (ang)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of sample structures per iteration",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=15,
        help="Maximum number of SSCHA iterations",
    )
    parser.add_argument("--mixing", type=float, default=0.5, help="Mixing parameter")
    parser.add_argument(
        "--mesh",
        type=int,
        nargs=3,
        default=[10, 10, 10],
        help="q-mesh for phonon calculation",
    )
    # Parameters for geometry optimization
    parser.add_argument("--gtol", type=float, default=0.1, help="Tolerance parameter")
    parser.add_argument("--fix_cell", action="store_true", help="Fix cell parameters")
    parser.add_argument("--fix_volume", action="store_true", help="Fix volume")
    parser.add_argument(
        "--relax_positions",
        action="store_true",
        help="Relax atomic positions during the geometry optimization",
    )
    parser.add_argument(
        "--method", type=str, default="BFGS", help="Optimization method"
    )

    args = parser.parse_args()

    if args.estimate_n_samp:
        poscar = args.poscar
        cell = Poscar(poscar).structure
        estimate_n_samples(
            cell=cell,
            pot=args.pot,
            yamlfile=args.yamlfile,
            fc2file=args.fc2file,
            gtol=args.gtol,
            n_samples=args.n_samples,
        )
    else:
        run_opt_sscha(
            poscar=args.poscar,
            supercell=args.supercell,
            pot=args.pot,
            pressure=args.pressure,
            temperature=args.temperature,
            init_fc_algorithm=args.init,
            yamlfile=args.yamlfile,
            fc2file=args.fc2file,
            tol=args.tol,
            dtol=args.dtol,
            gtol=args.gtol,
            n_samples=args.n_samples,
            max_iter=args.max_iter,
            mixing=args.mixing,
            mesh=args.mesh,
            relax_cell=not args.fix_cell,
            relax_volume=not args.fix_volume,
            relax_positions=args.relax_positions,
            method=args.method,
        )

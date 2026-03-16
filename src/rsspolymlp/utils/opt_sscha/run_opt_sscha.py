import argparse
import glob
import os
import shutil
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
    se_ratio: float = 0.1,
    n_samples: Optional[int] = None,
    relax_positions: bool = False,
):
    if n_samples is None:
        n_samples = 2000

    prp_sscha = SSCHAProperty(
        yamlfile,
        fc2file,
        cell=cell,
        pot=pot,
        n_samples=n_samples,
        pressure=0,
    )
    sd_e, sd_f, sd_s, sd_derivatives_f, sd_derivatives_s = (
        prp_sscha.estimate_derivatives_standard_deviation()
    )
    if relax_positions is False or sd_derivatives_f is None:
        max_dirivative = np.max(sd_derivatives_s)
    else:
        max_dirivative = np.max((np.max(sd_derivatives_f), np.max(sd_derivatives_s)))

    target_n_samples = int((max_dirivative / (gtol * se_ratio)) ** 2)
    max_sd_f = np.max(sd_f)
    max_sd_s = np.max(sd_s)
    se_e_average = sd_e / (target_n_samples**0.5)
    se_f_average = max_sd_f / (target_n_samples**0.5)
    se_s_average = max_sd_s / (target_n_samples**0.5)
    n_atoms = cell.n_atoms[0]

    with open("estimate_n_samples.yaml", "w") as f:
        print("inputs:", file=f)
        print("  potentials:", pot, file=f)
        print("  yamlfile:", yamlfile, file=f)
        print("  fc2file:", fc2file, file=f)
        print("  gtol:", gtol, file=f)
        print("  se_ratio:", se_ratio, file=f)
        print("  n_samples:", n_samples, file=f)
        print("units:", file=f)
        print("  sd_derivatives_f: meV/cell", file=f)
        print("  sd_derivatives_s: meV/Ang.", file=f)
        print("  sd_e: meV/atom", file=f)
        print("  max_sd_f: meV/Ang.", file=f)
        print("  max_sd_s: meV/atom", file=f)
        print("  se_e_average: meV/atom", file=f)
        print("  se_f_average: meV/Ang.", file=f)
        print("  se_s_average: meV/atom", file=f)
        print("properties:", file=f)
        print("  target_n_samples:", target_n_samples, file=f)
        if sd_derivatives_f is not None:
            print("  sd_derivatives_f:", sd_derivatives_f * 1000, file=f)
        print("  sd_derivatives_s:", sd_derivatives_s * 1000, file=f)
        print("  sd_e:", sd_e * 1000 / n_atoms, file=f)
        print("  max_sd_f:", max_sd_f * 1000, file=f)
        print("  max_sd_s:", max_sd_s * 1000 / n_atoms, file=f)
        print("  se_e_average:", se_e_average * 1000 / n_atoms, file=f)
        print("  se_f_average:", se_f_average * 1000, file=f)
        print("  se_s_average:", se_s_average * 1000 / n_atoms, file=f)

    return target_n_samples


def run_opt_sscha(
    poscar: str,
    supercell_matrix: list = [2, 2, 2],
    pot: str = "polymlp.yaml",
    pressure: float = 0,
    temperature: float = 300,
    init_fc_algorithm: str = "harmonic",
    yamlfile: Optional[str] = None,
    fc2file: Optional[str] = None,
    tol: float = 0.01,
    dtol: float = 0.01,
    gtol: float = 0.01,
    se_ratio: float = 0.2,
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
    cell = get_supercell(cell, supercell_matrix)

    pots = pot if isinstance(pot, list) else [pot]
    _pot = [os.path.abspath(p) for p in pots]

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
        os.makedirs(f"iteration_{iter_n}", exist_ok=True)
        judge, cell, n_samples = optimization_fc2_position(
            cell=cell,
            pot=_pot,
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
        os.makedirs(f"iteration_{iter_n}", exist_ok=True)
        for name in ["fc2.hdf5", "sscha_results.yaml", "total_dos.dat"]:
            shutil.copy(os.path.join(sscha_dir, name), f"iteration_{iter_n}")

        if not relax_cell and not relax_volume and not relax_positions:
            print("No degree of freedom to be optimized.")
            break

        os.chdir(f"iteration_{iter_n}")
        yamlfile = os.path.abspath("./sscha_results.yaml")
        fc2file = os.path.abspath("./fc2.hdf5")

        if iter_n == 0:
            _gtol = gtol * 10
        else:
            _gtol = gtol
        _n_samples = estimate_n_samples(
            cell=cell,
            pot=_pot,
            yamlfile=yamlfile,
            fc2file=fc2file,
            gtol=_gtol,
            se_ratio=se_ratio,
            relax_positions=relax_positions,
        )
        if _n_samples > n_samples:
            print(f"Updating number of samples: {n_samples} -> {_n_samples}")
            n_samples = _n_samples

        minobj = GeometryOptimization(
            cell=cell,
            relax_cell=relax_cell,
            relax_volume=relax_volume,
            relax_positions=relax_positions,
            sscha_opt=True,
            pressure=pressure,
            pot=_pot,
            verbose=True,
        )
        reliable_gtol = _gtol * (1 - 1.96 * se_ratio)
        minobj.run(
            method=method,
            gtol=reliable_gtol,
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

        os.chdir("../")

        poscar = f"iteration_{iter_n}/POSCAR_eqm.refine"
        cell = Poscar(poscar).structure
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
            n_samples = n_samples * 2

    shutil.copy(poscar, "./POSCAR_eqm.refine")
    print("------ final sscha calculation ------", flush=True)
    sscha = PypolymlpSSCHA(verbose=True)
    sscha._unitcell = cell
    sscha._supercell_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sscha.set_polymlp(_pot)

    sscha = sscha.run(
        temp=temperature,
        n_samples_init=n_samples,
        n_samples_final=n_samples * 2,
        tol=tol,
        mesh=mesh,
        max_iter=max_iter,
        mixing=mixing,
        init_fc_algorithm="file",
        init_fc_file=fc2file,
        precondition=False,
    )
    for name in ["fc2.hdf5", "sscha_results.yaml", "total_dos.dat"]:
        shutil.copy(os.path.join(sscha_dir, name), "./")
    with open("optimization_status.dat", "a") as f:
        print("success", file=f)
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
    parser.add_argument("--gtol", type=float, default=0.02, help="Tolerance parameter")
    parser.add_argument("--se_ratio", type=float, default=0.1, help="")
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
            relax_positions=args.relax_positions,
        )
    else:
        run_opt_sscha(
            poscar=args.poscar,
            supercell_matrix=args.supercell,
            pot=args.pot,
            pressure=args.pressure,
            temperature=args.temperature,
            init_fc_algorithm=args.init,
            yamlfile=args.yamlfile,
            fc2file=args.fc2file,
            tol=args.tol,
            dtol=args.dtol,
            gtol=args.gtol,
            se_ratio=args.se_ratio,
            n_samples=args.n_samples,
            max_iter=args.max_iter,
            mixing=args.mixing,
            mesh=args.mesh,
            relax_cell=not args.fix_cell,
            relax_volume=not args.fix_volume,
            relax_positions=args.relax_positions,
            method=args.method,
        )

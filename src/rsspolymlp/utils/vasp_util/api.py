import shutil
from pathlib import Path

from rsspolymlp.utils.vasp_util.gen_incar import (
    generate_optimization_incar,
    generate_single_point_incar,
)
from rsspolymlp.utils.vasp_util.gen_script import (
    generate_opt_shell_script,
    generate_sp_shell_script,
)


def prepare_vasp_inputs(
    run_vaspmpi: str,
    mode: str = "sp",  # "opt" or "sp"
    poscar_path: str = "./POSCAR",
    potcar_path: str = "./POTCAR",
    script_name: str = "run_vasp.sh",
    ENCUT: float = 400,
    KSPACING: float = 0.09,
    PSTRESS: float = 0.0,
    EDIFF: float = 1e-6,
    NELM: int = 100,
    NELMIN: int = 5,
    ALGO: str = "Normal",
    PREC: str = "Accurate",
    ADDGRID: bool = True,
    LREAL: bool = False,
    ISMEAR: int = 1,
    SIGMA: float = 0.2,
    NCORE: int = 2,
    LCHARG: bool = False,
    LWAVE: bool = False,
    EDIFFG: float = -0.01,
    IBRION: int = 2,
    ISIF: int = 3,
    NSW: int = 50,
):

    # Generate INCAR file
    if mode == "sp":
        generate_single_point_incar(
            incar_name="INCAR-sp",
            ENCUT=ENCUT,
            KSPACING=KSPACING,
            PSTRESS=PSTRESS,
            EDIFF=EDIFF,
            NELM=NELM,
            NELMIN=NELMIN,
            ALGO=ALGO,
            PREC=PREC,
            ADDGRID=ADDGRID,
            LREAL=LREAL,
            ISMEAR=ISMEAR,
            SIGMA=SIGMA,
            NCORE=NCORE,
            LCHARG=LCHARG,
            LWAVE=LWAVE,
        )
    elif mode == "opt":
        generate_optimization_incar(
            incar_name="INCAR-first",
            EDIFFG=EDIFFG,
            IBRION=IBRION,
            ISIF=ISIF,
            NSW=1,
            ENCUT=ENCUT,
            KSPACING=KSPACING,
            PSTRESS=PSTRESS,
            EDIFF=EDIFF,
            NELM=NELM,
            NELMIN=NELMIN,
            ALGO=ALGO,
            PREC=PREC,
            ADDGRID=ADDGRID,
            LREAL=LREAL,
            ISMEAR=ISMEAR,
            SIGMA=SIGMA,
            NCORE=NCORE,
            LCHARG=LCHARG,
            LWAVE=LWAVE,
        )
        generate_optimization_incar(
            incar_name="INCAR-relax",
            EDIFFG=EDIFFG,
            IBRION=IBRION,
            ISIF=ISIF,
            NSW=NSW,
            ENCUT=ENCUT,
            KSPACING=KSPACING,
            PSTRESS=PSTRESS,
            EDIFF=EDIFF,
            NELM=NELM,
            NELMIN=NELMIN,
            ALGO=ALGO,
            PREC=PREC,
            ADDGRID=ADDGRID,
            LREAL=LREAL,
            ISMEAR=ISMEAR,
            SIGMA=SIGMA,
            NCORE=NCORE,
            LCHARG=LCHARG,
            LWAVE=LWAVE,
        )
    else:
        raise ValueError("Mode must be either `sp` or `opt`.")

    # Copy POSCAR and POTCAR files
    if poscar_path not in ["POSCAR", "./POSCAR"]:
        shutil.copy(poscar_path, "./POSCAR")
    if potcar_path not in ["POTCAR", "./POTCAR"]:
        shutil.copy(potcar_path, "./POTCAR")

    # Generate shell script
    if mode == "sp":
        script_str = generate_sp_shell_script(
            run_vaspmpi=run_vaspmpi,
        )
    elif mode == "opt":
        script_str = generate_opt_shell_script(
            run_vaspmpi=run_vaspmpi,
        )
    else:
        raise ValueError("Mode must be either `sp` or `opt`.")

    Path(script_name).write_text(script_str + "\n")
    print(f"Shell script written to: {script_name}")

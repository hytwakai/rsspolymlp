import argparse


def bool_to_vasp(val: bool) -> str:
    return ".TRUE." if val else ".FALSE."


def generate_single_point_incar(
    incar_name: str = "INCAR",
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
) -> list[str]:
    """
    Generate a VASP INCAR file for single-point calculation.

    Returns the list of lines written.
    """
    lines = [
        "ISTART = 0",
        "ICHARG = 2",
        f"ENCUT = {ENCUT}",
        f"KSPACING = {KSPACING}",
        f"PSTRESS = {PSTRESS}",
        f"EDIFF = {EDIFF:.1e}",
        f"NELM = {NELM}",
        f"NELMIN = {NELMIN}",
        f"ALGO = {ALGO}",
        f"PREC = {PREC}",
        f"ADDGRID = {bool_to_vasp(ADDGRID)}",
        f"LREAL = {bool_to_vasp(LREAL)}",
        f"ISMEAR = {ISMEAR}",
    ]

    if ISMEAR != -5:
        lines.append(f"SIGMA = {SIGMA}")

    lines += [
        f"NCORE = {NCORE}",
        f"LCHARG = {bool_to_vasp(LCHARG)}",
        f"LWAVE = {bool_to_vasp(LWAVE)}",
    ]

    if not incar_name == "__tmp__":
        with open(incar_name, "w") as f:
            f.write("\n".join(lines) + "\n")

    return lines


def generate_optimization_incar(
    incar_name: str = "INCAR",
    EDIFFG: float = -0.01,
    IBRION: int = 2,
    ISIF: int = 3,
    NSW: int = 50,
    **kwargs,  # passed to generate_single_point_incar
) -> list[str]:
    """
    Generate a VASP INCAR file for geometry optimization.

    Returns the list of lines written.
    """
    lines = generate_single_point_incar(incar_name="__tmp__", **kwargs)
    opt_lines = [
        f"EDIFFG = {EDIFFG}",
        f"IBRION = {IBRION}",
        f"ISIF = {ISIF}",
        f"NSW = {NSW}",
    ]
    lines.extend(opt_lines)

    with open(incar_name, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate VASP INCAR file.")

    parser.add_argument("--sp", action="store_true")
    parser.add_argument("--opt", action="store_true")
    parser.add_argument("--incar_name", default="INCAR", type=str)

    parser.add_argument("--ENCUT", default=400, type=float)
    parser.add_argument("--KSPACING", default=0.09, type=float)
    parser.add_argument("--PSTRESS", default=0.0, type=float)
    parser.add_argument("--EDIFF", default=1e-6, type=float)
    parser.add_argument("--NELM", default=100, type=int)
    parser.add_argument("--NELMIN", default=5, type=int)
    parser.add_argument("--ALGO", default="Normal", type=str)
    parser.add_argument("--PREC", default="Accurate", type=str)
    parser.add_argument("--ADDGRID", default=True, type=bool)
    parser.add_argument("--LREAL", action="store_true")
    parser.add_argument("--ISMEAR", default=1, type=int)
    parser.add_argument("--SIGMA", default=0.2, type=float)
    parser.add_argument("--NCORE", default=2, type=int)
    parser.add_argument("--LCHARG", action="store_true")
    parser.add_argument("--LWAVE", action="store_true")

    parser.add_argument("--EDIFFG", default=-0.01, type=float)
    parser.add_argument("--IBRION", default=2, type=int)
    parser.add_argument("--ISIF", default=3, type=int)
    parser.add_argument("--NSW", default=50, type=int)

    args = parser.parse_args()

    if args.sp:
        generate_single_point_incar(
            incar_name=args.incar_name,
            ADDGRID=args.ADDGRID,
            ALGO=args.ALGO,
            EDIFF=args.EDIFF,
            LCHARG=args.LCHARG,
            LREAL=args.LREAL,
            LWAVE=args.LWAVE,
            NELM=args.NELM,
            NELMIN=args.NELMIN,
            PREC=args.PREC,
            ISMEAR=args.ISMEAR,
            SIGMA=args.SIGMA,
            ENCUT=args.ENCUT,
            KSPACING=args.KSPACING,
            PSTRESS=args.PSTRESS,
            NCORE=args.NCORE,
        )

    if args.opt:
        generate_optimization_incar(
            incar_name=args.incar_name,
            EDIFFG=args.EDIFFG,
            IBRION=args.IBRION,
            ISIF=args.ISIF,
            NSW=args.NSW,
            ADDGRID=args.ADDGRID,
            ALGO=args.ALGO,
            EDIFF=args.EDIFF,
            LCHARG=args.LCHARG,
            LREAL=args.LREAL,
            LWAVE=args.LWAVE,
            NELM=args.NELM,
            NELMIN=args.NELMIN,
            PREC=args.PREC,
            ISMEAR=args.ISMEAR,
            SIGMA=args.SIGMA,
            ENCUT=args.ENCUT,
            KSPACING=args.KSPACING,
            PSTRESS=args.PSTRESS,
            NCORE=args.NCORE,
        )

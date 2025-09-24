import os
import re
import shutil

from pypolymlp.utils.vasprun_compress import compress_vaspruns


def max_iteration_reached(vasp_path: str) -> bool:
    """Return True if the number of electronic steps equals NELM."""
    # Count DAV or RMM steps in OSZICAR
    pattern = re.compile(r"^(DAV|RMM)")
    iteration = 0
    with open(os.path.join(vasp_path, "OSZICAR")) as f:
        iteration = sum(1 for line in f if pattern.match(line))

    # Extract NELM from INCAR
    nelm = None
    with open(os.path.join(vasp_path, "INCAR")) as f:
        for line in f:
            if "NELM" in line and "NELMIN" not in line:
                try:
                    nelm = int(line.split("=")[-1].strip())
                except ValueError:
                    pass

    return nelm is not None and iteration == nelm


def check_convergence(
    vasp_paths: list[str],
    vasprun_status={"fail": 0, "fail_iteration": 0, "parse": 0, "success": 0},
):
    valid_paths = []
    for vasp_path in vasp_paths:
        if not os.path.isfile(f"{vasp_path}/OSZICAR"):
            vasprun_status["fail"] += 1
            print(vasp_path, "failed")
            continue
        if "E0=" not in open(f"{vasp_path}/OSZICAR").read():
            vasprun_status["fail"] += 1
            print(vasp_path, "failed")
            continue
        if max_iteration_reached(vasp_path):
            vasprun_status["fail_iteration"] += 1
            print(vasp_path, "failed_iteration")
            continue
        valid_paths.append(vasp_path)

    return valid_paths, vasprun_status


def compress(vasprun_path, output_dir: str = "compress_dft_data"):
    cwd_path = os.getcwd()
    if os.path.isfile(f"{output_dir}/{'.'.join(vasprun_path.split('/'))}"):
        return True

    if os.path.isfile(vasprun_path):
        os.chdir(os.path.dirname(vasprun_path))
        if not os.path.isfile("vasprun.xml.polymlp"):
            judge = compress_vaspruns("vasprun.xml")
        else:
            judge = True
        os.chdir(cwd_path)
        if judge:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(
                vasprun_path + ".polymlp",
                f"{output_dir}/{'.'.join(vasprun_path.split('/'))}",
            )
        else:
            print(vasprun_path, "failed_parse")
            return False
    else:
        return False

    print(vasprun_path)
    return True

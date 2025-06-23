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
            if "NELM" in line:
                try:
                    nelm = int(line.split("=")[-1].strip())
                except ValueError:
                    pass

    return nelm is not None and iteration == nelm


def convert(vasprun_path, output_dir: str):
    if os.path.isfile(f"{output_dir}/{'.'.join(vasprun_path.split('/'))}"):
        return True

    os.chdir(os.path.dirname(vasprun_path))
    if os.path.isfile(vasprun_path):
        judge = compress_vaspruns(vasprun_path)
        if judge:
            shutil.copy(
                vasprun_path + ".polymlp",
                f"{output_dir}/{'.'.join(vasprun_path.split('/'))}",
            )
        else:
            return False
    else:
        return False

    print(vasprun_path)
    return True


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        nargs="+",
        required=True,
        help="Directory paths containing vasp results.",
    )
    args = parser.parse_args()

    vasp_paths = args.path

    vasprun_status = {"fail": 0, "fail_iteration": 0, "parse": 0, "success": 0}
    for vasp_path in vasp_paths:
        if not os.path.isfile(f"{vasp_path}/OSZICAR"):
            vasprun_status["fail"] += 1
            continue
        if "E0=" not in open(f"{vasp_path}/OSZICAR").read():
            vasprun_status["fail"] += 1
            continue
        if max_iteration_reached(vasp_path):
            vasprun_status["fail_iteration"] += 1
            continue

        judge = convert(vasp_path)
        if judge:
            vasprun_status["success"] += 1
        else:
            vasprun_status["parse"] += 1

        print(f"<< input {len(vasp_paths)} structure >>")
        print(f"<< success {vasprun_status['success']} structure >>")
        print(f"<< failed calclation {vasprun_status['fail']} structure >>")
        print(f"<< failed iteration {vasprun_status['fail_iteration']} structure >>")
        print(f"<< failed parse {vasprun_status['parse']} structure >>")

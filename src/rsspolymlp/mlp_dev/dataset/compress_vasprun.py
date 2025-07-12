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
            continue
        if "E0=" not in open(f"{vasp_path}/OSZICAR").read():
            vasprun_status["fail"] += 1
            continue
        if max_iteration_reached(vasp_path):
            vasprun_status["fail_iteration"] += 1
            continue
        valid_paths.append(vasp_path)

    return valid_paths, vasprun_status


def compress(vasprun_path, output_dir: str = "dft_dataset"):
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
            return False
    else:
        return False

    print(vasprun_path)
    return True


if __name__ == "__main__":

    import argparse

    from joblib import Parallel, delayed

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        nargs="+",
        required=True,
        help="Directory paths containing vasp results.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dft_dataset",
        help="Output directory path.",
    )
    parser.add_argument(
        "--num_process",
        type=int,
        default=4,
        help="Number of processes to use with joblib.",
    )
    args = parser.parse_args()

    vasp_paths = args.path

    valid_paths, vasprun_status = check_convergence(vasp_paths=vasp_paths)

    judge_list = Parallel(n_jobs=args.num_process)(
        delayed(compress)(vasp_path + "/vasprun.xml", args.output_dir)
        for vasp_path in valid_paths
    )
    vasprun_status["success"] += sum(judge_list)
    vasprun_status["parse"] += len(judge_list) - sum(judge_list)

    with open("dataset_status.yaml", "a") as f:
        print(f"{os.path.dirname(vasp_paths[0])}:", file=f)
        print(f" - input:             {len(vasp_paths)}", file=f)
        print(f"   success:           {vasprun_status['success']}", file=f)
        print(f"   failed_calclation: {vasprun_status['fail']}", file=f)
        print(f"   failed_iteration:  {vasprun_status['fail_iteration']}", file=f)
        print(f"   failed_parse:      {vasprun_status['parse']}", file=f)

    print(f"{os.path.dirname(vasp_paths[0])}:")
    print(f" - input:             {len(vasp_paths)} structure")
    print(f"   success:           {vasprun_status['success']} structure")
    print(f"   failed calclation: {vasprun_status['fail']} structure")
    print(f"   failed iteration:  {vasprun_status['fail_iteration']} structure")
    print(f"   failed parse:      {vasprun_status['parse']} structure")

# VASP calculation utility

## Description

This utility prepares the necessary input files and execution script for a VASP calculation.
Specifically, it generates:

 - INCAR file with the specified calculation settings

 - copies of the provided POSCAR and POTCAR files

 - `run_vasp.sh` shell script to execute the VASP job

The calculation mode (e.g., single-point or structure optimization) and input parameters can be flexibly specified in the `prepare_vasp_inputs` function.

## Single-point calculation
```python
import subprocess
from rsspolymlp.utils.vasp_util.api import prepare_vasp_inputs

# Generate input files for a VASP single-point calculation
prepare_vasp_inputs(
    mode="sp",                         # Mode: "sp" for single-point calculation
    poscar_path="./POSCAR",            # Path to the POSCAR file
    potcar_path="{your_potcar_path}",  # Path to the POTCAR file
    run_vaspmpi="srun {your_vasp_binary_path}",  # MPI command to run VASP

    # INCAR parameters
    ENCUT=400,       # Plane-wave energy cutoff (eV)
    KSPACING=0.09,   # k-point mesh spacing
    EDIFF=1e-6,      # Electronic convergence criterion
)

# Run the generated shell script
subprocess.run(["zsh", "run_vasp.sh"], check=True)
```

## Local geometry optimization
```python
import subprocess
from rsspolymlp.utils.vasp_util.api import prepare_vasp_inputs

# Generate input files for a VASP geometry optimization
prepare_vasp_inputs(
    mode="opt",  # Mode: "opt" for local geometry optimization
    poscar_path="./POSCAR",
    potcar_path="{your_potcar_path}",
    run_vaspmpi="srun {your_vasp_binary_path}",
    ENCUT=400,
    KSPACING=0.09,
    EDIFF=1e-6,
    EDIFFG=-0.01,
    ISIF=3,
    PSTRESS=100,
)
subprocess.run(["zsh", "run_vasp.sh"], check=True)

# Generate input files for calculating highly accurate energy
prepare_vasp_inputs(
    mode="sp",
    poscar_path="./POSCAR",
    potcar_path="{your_potcar_path}",
    run_vaspmpi="srun {your_vasp_binary_path}",
    script_name="e_final_vasp.sh",
    ENCUT=400,
    KSPACING=0.09,
    EDIFF=1e-6,
    ISMEAR=-5,
    PSTRESS=100,
)
subprocess.run(["zsh", "e_final_vasp.sh"], check=True)
```

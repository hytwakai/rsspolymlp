# Python API (RSS)

## Initial structure generation
```python
from rsspolymlp.rss.random_struct import GenerateRandomStructure

gen_str = GenerateRandomStructure(
    elements=["Al", "Cu"],
    atom_counts=[4, 4],
    num_init_str=2000,
    least_distance=0.0,
)
gen_str.random_structure(min_volume=0.0, max_volume=100)
```


## Global RSS with polynomial MLPs
```python
import glob
import multiprocessing
import os

from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

from rsspolymlp.rss.optimization_mlp import RandomStructureSearch

# Check which structures have already been optimized
poscar_path_all = sorted(
    glob.glob("initial_struct/*"), key=lambda x: int(x.split("_")[-1])
)
if os.path.isfile("rss_result/finish.dat"):
    with open("rss_result/finish.dat") as f:
        finished_set = set(line.strip() for line in f)
    poscar_path_all = [
        p for p in poscar_path_all if os.path.basename(p) not in finished_set
    ]

num_process = multiprocessing.cpu_count()
backend = "loky"

rssobj = RandomStructureSearch(
    pot="polymlp.yaml",
    pressure=0.0,
    solver_method="CG",
    maxiter=100,
    num_opt_str=1000,
)

# Perform parallel optimization with joblib
Parallel(n_jobs=num_process, backend=backend)(
    delayed(rssobj.run_optimization)(poscar) for poscar in poscar_path_all
)
executor = get_reusable_executor(max_workers=num_process)
executor.shutdown(wait=True)
```

`srun` can also used for parallel execution (default: `joblib`), which is suitable for high-performance computing environments. 
By specifying `--parallel_method srun`, a script named `multiprocess.sh` will be automatically generated for execution with `srun`. 

```bash
rss-parallel --parallel_method srun --pot polymlp.yaml --num_opt_str 1000
srun -n $SLURM_CPUS_ON_NODE ./multiprocess.sh
```

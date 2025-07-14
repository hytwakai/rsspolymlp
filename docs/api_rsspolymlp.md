# Python API (RSS)

## Initial structure generation
```python
from rsspolymlp.rss.random_struct import GenerateRandomStructure

gen_str = GenerateRandomStructure(
    element_list=["Al", "Cu"],
    atom_counts=[4, 4],
    min_volume=0,
    max_volume=100,
    least_distance=1.0,
)
gen_str.random_structure(max_init_struct=2000)
```


## Global RSS with polynomial MLPs
```python
import glob
import multiprocessing
import os

from joblib import Parallel, delayed

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
    c_maxiter=100,
    n_opt_str=1000,
)

# Perform parallel optimization with joblib
Parallel(n_jobs=num_process, backend=backend)(
    delayed(rssobj.run_optimization)(poscar) for poscar in poscar_path_all
)
```

## Unique structure identification and RSS summary generation
```python
from rsspolymlp.rss.eliminate_duplicates import RSSResultAnalyzer

analyzer = RSSResultAnalyzer()
analyzer.run_rss_uniq_struct(
    num_str=-1,
    use_joblib=True,
    num_process=-1,
    backend="loky",
)
```


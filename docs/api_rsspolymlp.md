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

## Identification of unique structures across atom counts `n` or pressures `p`
```python
import glob
import os
from rsspolymlp.analysis.rss_summarize import RSSResultSummarizer

os.makedirs("rss_summary", exist_ok=True)
os.chdir("rss_summary")

target_paths = glob.glob("../rss_mlp/Al-Cu/0.0GPa/*")
analyzer = RSSResultSummarizer(
    result_paths=paths,
    use_joblib=True,
    num_process=-1,
    backend="loky,
    symprec_set=[1e-5, 1e-4, 1e-3, 1e-2],
    output_poscar=False,
    threshold=None,
)

summarize_p = False
if not summarize_p:
    # Identify unique structures across atom numbers `n`
    analyzer.run_summarize()
else:
    # Identify unique structures across pressures `p`
    analyzer.run_summarize_p()
```

## Elimination of ghost minima
To identify ghost minimum structure candidates:
```python
import glob
from rsspolymlp.analysis.ghost_minima import ghost_minima_candidates

result_paths = glob.glob("./json/*")
dir_path = os.path.dirname(result_paths[0])
os.makedirs(f"{dir_path}/../ghost_minima/ghost_minima_candidates", exist_ok=True)
os.chdir(f"{dir_path}/../")

ghost_minima_candidates(result_paths=result_paths)
```

After performing DFT calculations for the ghost minimum structure candidates:
```python
import glob
from rsspolymlp.analysis.ghost_minima import detect_actual_ghost_minima

detect_actual_ghost_minima(dft_dir="./ghost_minima_dft")
```

## Phase stability analysis
```python
from rsspolymlp.analysis.phase_analysis import ConvexHullAnalyzer

ch_analyzer = ConvexHullAnalyzer(elements=["Al", "Cu"])
ch_analyzer.parse_results(
    input_paths="./json/*",
    ghost_minima_file="./ghost_minima/ghost_minima_detection.yaml",
    # or None
)
ch_analyzer.run_analysis()

threshold_list = [10, 30, 50]
for threshold in threshold_list:
    ch_analyzer.get_structures_near_hull(threshold)
```

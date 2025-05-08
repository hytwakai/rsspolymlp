## Additional Options

`rss-init-struct`

- `--least_distance`: Minimum interatomic distance in the initial structure, in angstroms. *(default: 0.0)*
- `--max_volume`: Maximum volume per atom for the initial structure (Å³/atom). *(default: 100.0)*
- `--min_volume`: Minimum volume per atom for the initial structure (Å³/atom). *(default: 0.0)*


`rss-parallel`

- `--pressure`: Pressure to apply during optimization (in GPa). *(default: 0.0)*
- `--solver_method`: Optimization solver to use. *(default: CG)*
- `--maxiter`: Maximum number of iterations for parameter tuning (e.g., `c1`, `c2`). *(default: 100)*
- `--parallel_method`: Parallelization method. *(choices: joblib, srun; default: joblib)*
- `--num_process`: Number of processes for `joblib`; `-1` uses all available cores. *(default: -1)*
- `--backend`: Backend used by `joblib`. *(default: loky)*

You can also use `srun` for parallel execution (default: `joblib`), which is suitable for high-performance computing environments. 
By specifying `--parallel_method srun`, a script named `multiprocess.sh` will be automatically generated for execution with `srun`. 

```bash
rss-parallel --parallel_method srun --pot polymlp.yaml --num_opt_str 1000
srun -n $SLURM_CPUS_ON_NODE ./multiprocess.sh
```
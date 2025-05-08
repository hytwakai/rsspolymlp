# A framework for random structure search (RSS) using polynomial MLPs

## Citation of rsspolymlp

If you use `rsspolymlp` in your study, please cite the following articles.

“Efficient global crystal structure prediction using polynomial machine learning potential in the binary Al–Cu alloy system”, [J. Ceram. Soc. Jpn. 131, 762 (2023)](https://www.jstage.jst.go.jp/article/jcersj2/131/10/131_23053/_article/-char/ja/)
```
@article{HayatoWakai202323053,
  title="{Efficient global crystal structure prediction using polynomial machine learning potential in the binary Al–Cu alloy system}",
  author={Hayato Wakai and Atsuto Seko and Isao Tanaka},
  journal={J. Ceram. Soc. Jpn.},
  volume={131},
  number={10},
  pages={762-766},
  year={2023},
  doi={10.2109/jcersj2.23053}
}
```

## Installation

### Required libraries and python modules

- python >= 3.9
- pypolymlp
- symfc
- spglib
- joblib

### How to install

```shell
git clone https://github.com/hytwakai/rsspolymlp.git
cd rsspolymlp
conda create -n rsspolymlp
conda activate rsspolymlp
conda install -c conda-forge pypolymlp symfc spglib joblib
pip install .
```

## Usage

The command-line interface of `rsspolylmp` is organized into three sections, each corresponding to a different phase of the workflow:
1. Generating initial structures (`rss-init-struct`)
2. Performing parallel geometry optimization (`rss-parallel`)
3. Analyzing RSS results (`rss-analysis`)

### Example Commands

```shell
rss-init-struct --elements Al Cu --atom_counts 4 4 --num_init_str 2000
rss-parallel --pot polymlp.yaml --num_opt_str 1000
rss-analysis
```

| Argument            | Description |
|---------------------|-------------|
| `--elements`        | List of element symbols (e.g., `Al Cu`). |
| `--atom_counts`     | Number of atoms for each element (must match the order of `--elements`). |
| `--num_init_str`    | Number of random initial structures to generate. |
| `--pot`             | Path to the polynomial MLP potential file. |
| `--num_opt_str`     | Maximum number of optimized structures to obtain from RSS. |

### Additional Options

`rss-init-struct`

| Argument           | Description |
|--------------------|-------------|
| `--least_distance` | Minimum interatomic distance in the initial structure, in angstroms. **Default**: `0.0` |
| `--max_volume`     | Maximum volume per atom for the initial structure (Å³/atom). **Default**: `100.0` |
| `--min_volume`     | Minimum volume per atom for the initial structure (Å³/atom). **Default**: `0.0` |

`rss-parallel`

| Argument          | Description |
|-------------------|-------------|
| `--pressure`      | Pressure to apply during optimization (in GPa). **Default**: `0.0` |
| `--solver_method` | Optimization solver to use. **Default**: `CG` |
| `--maxiter`       | Maximum number of iterations for parameter tuning (e.g., `c1`, `c2`). **Default**: `100` |
| `--parallel_method`  | Parallelization method. <br>**Choices**: `joblib`, `srun` **Default**: `joblib` |
| `--num_process`      | Number of processes for `joblib`; `-1` uses all available cores. **Default**: `-1` |
| `--backend`          | Backend used by `joblib`. **Default**: `loky` |

You can also use `srun` for parallel execution (default: `joblib`), which is suitable for high-performance computing environments. 
By specifying `--parallel_method srun`, a script named `multiprocess.sh` will be automatically generated for execution with `srun`. 

```bash
rss-parallel --parallel_method srun --pot polymlp.yaml --num_opt_str 1000
srun -n $SLURM_CPUS_ON_NODE ./multiprocess.sh
```
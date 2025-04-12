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

The command-line interface of `rsspolylmp` is divided into three sections.
Each section corresponds to a different phase of the workflow:
1. Generating initial structures (`gen-rand-struct`)
2. Geometry optimization performed in parallel (`rss-parallel`)
3. Sorting optimization results (`sort-struct`)

### Example Commands

#### Generating Initial Structures
```shell
gen-rand-struct --elements Al Cu --atom_counts 4 4 --num_init_str 2000
```
#### Running Geometry Optimization
```shell
rss-parallel --num_opt_str 1000 --pot polymlp.yaml
```
#### Sorting the optimization results
```shell
sort-struct
```

## Extended Usage

### 1. Random Structure Generation Arguments

These options configure the generation of random initial structures:

- `--elements`  
  **Type**: string (list)  
  **Description**: List of chemical element symbols (e.g., `Al, Cu`).

- `--atom_counts`  
  **Type**: int (list)  
  **Description**: Number of atoms for each element specified in `--elements`.

- `--num_init_str`  
  **Type**: int  
  **Default**: 5000  
  **Description**: Number of randomly generated initial structures.

- `--least_distance`  
  **Type**: float  
  **Default**: 0.0  
  **Description**: Minimum interatomic distance in the initial structure, in angstroms.

- `--max_volume`  
  **Type**: float  
  **Default**: 100.0  
  **Description**: Maximum volume per atom for the initial structure (A³/atom).

- `--min_volume`  
  **Type**: float  
  **Default**: 0.0  
  **Description**: Minimum volume per atom for the initial structure (A³/atom).

### 2.1. Geometry Optimization Arguments
These options control the settings for geometry optimizations:

- `--pot`  
  **Type**: string (list)
  **Default**: polymlp.yaml
  **Description**: Potential file used by the polynomial MLP.

- `--num_opt_str`  
  **Type**: int  
  **Default**: 1000  
  **Description**: Maximum number of optimized structures obtained from RSS.

- `--pressure`  
  **Type**: float  
  **Default**: 0.0  
  **Description**: Pressure term to be applied during optimization (in GPa).

- `--solver_method`  
  **Type**: string  
  **Default**: CG  
  **Description**: Type of solver used during the optimization process.

- `--maxiter`  
  **Type**: int  
  **Default**: 100  
  **Description**: Maximum number of iterations allowed when adjusting optimization parameters (e.g., c1 and c2 values).

### 2.2. Parallelization Arguments
These options the geometry optimization settings to enable parallel processing:

- `--parallel_method`  
  **Type**: string (choice)  
  **Choices**: joblib, srun  
  **Default**: joblib  
  **Description**: Selects the parallelization method.

- `--num_process`  
  **Type**: int  
  **Default**: -1  
  **Description**: Number of processes to use with joblib; use -1 to use all available CPU cores.

- `--backend`  
  **Type**: string (choice)  
  **Choices**: loky, threading, multiprocessing  
  **Default**: loky
  **Description**: Specifies the backend for joblib parallelization.

You can also use `srun` for parallel execution (default: `joblib`), which is suitable for high-performance computing environments. By specifying `--parallel_method srun`, a script named `multiprocess.sh` will be automatically generated for execution with `srun`. For example:

```shell
rss-parallel --parallel_method srun --num_opt_str 1000 --pot polymlp.yaml
srun -n $SLURM_CPUS_ON_NODE ./multiprocess.sh
```

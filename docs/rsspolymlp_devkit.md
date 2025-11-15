# Development kit for polynomial MLPs

The command-line tool `rsspolymlp-devkit` provides utilities for developing polynomial MLPs.

It supports structure generation, DFT dataset classification, MLP input preparation, and model selection based on Pareto-optimality.

## MLP dataset generation

Generates displecement structures from given POSCAR files to be used as training data for MLPs.

```bash
rsspolymlp-devkit --gen_data --poscars POSCAR1 POSCAR2 \
    --per_volume 1.0 --disp_max 40 --disp_grid 2
```

**Key options:**

* `--poscars`: List of input POSCAR files
* `--per_volume`: Volume scaling factor (default: 1.0)
* `--disp_max`: Maximum displacement ratio (default: 30)
* `--disp_grid`: Displacement grid step size (default: 1)
* `--natom_lb`, `--natom_ub`: Minimum and maximum number of atoms in generated structures

## DFT dataset division

Classifies DFT results based on the magnitude of force and stress components.
Datasets are categorized as `f_small`, `f_normal`, `f_large`, `f_exlarge`, or `s_large`.

```bash
rsspolymlp-devkit --divide_data --paths ./<vaspruns_dir>
```

**Threshold options:**

* `--th_e_high`: Energy threshold (eV/atom) for structures classified as high energy (`-e_high` tag) (default: 10.0).
* `--th_s_large`: Stress tensor threshold (GPa) for structures with large stress tensor (`s_large`) (default: 200.0).
* `--th_f_small`: Force threshold (eV/ang.) for structures with small forces (`f_small`) (default: 3.0).
* `--th_f_normal`: Force threshold for structures with normal forces (`f_normal`) (default: 10.0).
* `--th_f_large`: Force threshold for structures with large forces (`f_large`) (default: 100.0).

## Polynomial MLP development

Creates MLP input files with specified training/test datasets and custom weights for data containing large forces or stresses.

```bash
rsspolymlp-devkit --mlp_dev --input_path ./polymlp-0001 --elements Al Cu \
    --train_data ./train/* --test_data ./test/*
```

**Key options:**

* `--input_path`: Directory containing `polymlp*.in` templates
* `--elements`: List of element symbols
* `--train_data`, `--test_data`: Paths to training and test datasets

## Pareto-optimal MLP detection

Identifies MLP models that lie on the Pareto front with respect to energy RMSE and computational cost.

```bash
rsspolymlp-devkit --pareto_opt --paths ./polymlp-* --rmse_path test/f_small
```

**Key options:**

* `--error_path`: Path to the YAML file containing RMSE values (default: `polymlp_error.yaml`)
* `--rmse_path`: Subset of the dataset used to compute energy RMSE (e.g., `test/f_small`)

## Cost estimation

Estimates the computational cost (e.g., training or inference time) for each MLP model.

```bash
rsspolymlp-devkit --calc_cost --paths ./polymlp-*
```

Use `--param_input` to read cost estimates from `polymlp.in` files instead of `polymlp.yaml` files.

## Compressing DFT data

Compresses `vasprun.xml` files and checks whether each VASP calculation has converged.

```bash
rsspolymlp-devkit --compress_data --paths <vasp_calc_dirs> --output_dir compress_dft_data
```

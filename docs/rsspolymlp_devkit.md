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
* `--disp_max`: Maximum displacement ratio (default: 40)
* `--disp_grid`: Displacement grid step size (default: 2)
* `--natom_lb`, `--natom_ub`: Minimum and maximum number of atoms in generated structures
* `--str_name`: Index of the POSCAR filename to extract structure names

## DFT dataset division

Classifies DFT results based on the magnitude of force and stress components.
Datasets are categorized as `minima-close`, `force-normal`, `force-large`, `force-very-large`, or `stress-very-large`.

```bash
rsspolymlp-devkit --divide_data --paths ./<vaspruns_dir>
```

**Threshold options:**

* `--threshold_vlarge_s`: Stress threshold (GPa) for `stress-very-large` (default: 300.0)
* `--threshold_vlarge_f`: Force threshold (eV/Å) for `force-very-large` (default: 100.0)
* `--threshold_large_f`: Force threshold for `force-large` (default: 10.0)
* `--threshold_close_minima`: Force threshold for `minima-close` (default: 1.0)

## Polynomial MLP development

Creates MLP input files with specified training/testing datasets and custom weights for data containing large forces or stresses.

```bash
rsspolymlp-devkit --mlp_dev --input_path ./polymlp-0001 --elements Al Cu \
    --train_data ./train/* --test_data ./test/* --w_vlarge_f 0.1 --alpha_param -4 3 8
```

**Key options:**

* `--input_path`: Directory containing `polymlp*.in` templates
* `--elements`: List of element symbols
* `--train_data`, `--test_data`: Paths to training and test datasets
* `--w_large_f`, `--w_vlarge_f`, `--w_vlarge_s`: Weights for `force-large`, `force-very-large`, and `stress-very-large` datasets
* `--include_vlarge_f`, `--include_vlarge_s`: Include data from very large force/stress cases in training
* `--alpha_param`: Regularization parameter range, given as three integers (e.g., `-4 3 8`)

## Pareto-optimal MLP detection

Identifies MLP models that lie on the Pareto front with respect to energy RMSE and computational cost.

```bash
rsspolymlp-devkit --pareto_opt --paths ./polymlp-* --rmse_path test/minima-close
```

**Key options:**

* `--error_path`: Path to the YAML file containing RMSE values (default: `polymlp_error.yaml`)
* `--rmse_path`: Subset of the dataset used to compute energy RMSE (e.g., `test/minima-close`)

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

**Key options:**

* `--num_process`: Number of parallel processes used for compression (default: 4)
* `--output_dir`: Output directory for compressed data

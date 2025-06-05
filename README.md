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

[Optional]
- matplotlib (if plotting RSS results)
- seaborn (if plotting RSS results)

### How to install
- Install from conda-forge

| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-rsspolymlp-green.svg)](https://anaconda.org/conda-forge/rsspolymlp) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/rsspolymlp.svg)](https://anaconda.org/conda-forge/rsspolymlp) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/rsspolymlp.svg)](https://anaconda.org/conda-forge/rsspolymlp) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/rsspolymlp.svg)](https://anaconda.org/conda-forge/rsspolymlp) |

```shell
conda create -n rsspolymlp
conda activate rsspolymlp
conda install -c conda-forge rsspolymlp
```

- Install from PyPI
```shell
conda create -n rsspolymlp
conda activate rsspolymlp
conda install -c conda-forge pypolymlp symfc spglib joblib
pip install rsspolymlp
```

## Usage

### Example Commands

```shell
# Step 1–3: Execute for each (p, c, n) condition
rss-init-struct --elements Al Cu --atom_counts 4 4 --num_init_str 2000
rss-parallel --pot polymlp.yaml --num_opt_str 1000
rss-analysis

# Steps 4–6: Execute after the above steps and analyze the results aggregated by (p, c) conditions.
rss-summarize --elements Al Cu --use_joblib --rss_paths ./*
rss-outlier --result_paths ./Al*.log ./Cu*.log
plot-binary --elements Al Cu --result_paths Al*.log Cu*.log
```

### Workflow

The command-line interface of `rsspolymlp` is organized into several sections, each corresponding to a different phase of the workflow:

1. **Generating initial structures (`rss-init-struct`)**
   
   Random structures are generated under specified conditions of pressure (`p`), composition (`c`), and number of atoms (`n`).

2. **Performing parallel geometry optimization (`rss-parallel`)**
   
   Each generated structure is optimized in parallel using polynomial MLPs. These optimizations are performed independently for each (`p`, `c`, `n`) condition.

3. **Analyzing RSS results (`rss-analysis`)**
   
   This step processes the optimized structures for each (`p`, `c`, `n`) condition individually. It includes:

   * Removing duplicate structures and extracting unique optimized structures
   * Detecting outliers based on energy values

4. **Summarizing RSS results across atom numbers (`rss-summarize`)**
   
   This step aggregates the RSS results across different numbers of atoms `n` under the same pressure and composition conditions. It performs the same operations as `rss-analysis`, but across multiple `n` values.

5. **Outlier detection (`rss-outlier`)**
   
   Provides utilities for identifying and filtering out anomalous structures based on energy values.

6. **Binary phase diagram plotting (`plot-binary`)**
   
   Visualizes the convex hull and stability of binary systems based on the summarized results.

[Additional information is here](docs/rss.md)
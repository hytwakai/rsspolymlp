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
- scikit-learn
- joblib
- pypolymlp
- spglib
- symfc

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
conda install -c conda-forge scikit-learn joblib pypolymlp spglib symfc
pip install rsspolymlp
```

## Workflow

<img src="docs/workflow.png" alt="workflow" width="70%" />

### The command-line interface of `rsspolymlp`

First, RSS using the polynomial MLP is independently performed for each condition defined by pressure (`p`), composition (`c`), and number of atoms (`n`).

1. Generating initial random structures
   
   ```shell
   rss-init-struct --elements Al Cu --atom_counts 4 4 --num_init_str 2000
   ```

2. Performing parallel geometry optimization using the polynomial MLP
   
   ```shell
   rss-parallel --pot polymlp.yaml --pressure 0.0 --num_opt_str 1000
   ```

3. Eliminating duplicate structures
   
   This step processes the optimized structures. It includes:

   * Parsing optimization logs, filtering out failed or unconverged cases, and generating detailed computational summaries.
   * Removing duplicate structures and extracting unique optimized structures.

   ```shell
   rss-uniq-struct
   ```

Next, RSS results aggregated for each (`p`, `c`) condition are analyzed.

4. Identifying unique structures across atom numbers `n`

   ```shell
   rss-summarize --elements Al Cu --result_paths <rss_directory>/*
   # <rss_directory>: parent directory of RSS runs at the same pressure
   ```

5. Eliminating ghost minimum structures
   
   Identifying and filtering out ghost minimum structures based on nearest-neighbor distance are performed.

   ```shell
   rss-ghost-minima --result_paths <summary_dir>/json/*
   rss-ghost-minima --compare_dft --dft_dir <summary_dir>/ghost_minima_dft
   # <summary_dir>: output directory from rss-summarize, storing RSS results
   ```

6. Phase stability analysis

   This step computes the relative or formation energies of structures obtained from the RSS and outputs the global minimum structures. It also identifies metastable structures near the convex hull based on a energy threshold (e.g., 30 meV/atom).

   ```shell
   rss-phase-analysis --elements Al Cu --result_paths <summary_dir>/json/* 
   --thresholds 10 30 50 --ghost_minima_file <summary_dir>/ghost_minima/ghost_minima_detection.yaml
   ```

7. (Optional) Plotting RSS results (e.g., `plot-binary`)
   
   The energy distribution of structures obtained through this RSS workflow is visualized.
   ```shell
   plot-binary --elements Al Cu --threshold 30
   ```

## Additional information

 - [Python API (RSS)](docs/api_rss.md)
   - Initial structure generation
   - Global RSS with polynomial MLPs
   - Unique structure identification and RSS summary generation
 - [VASP calculation utility](src/rsspolymlp/utils/vasp_util/readme.md)
   - Single-point calculation
   - Local geometry optimizaion
 - [matplotlib utility](src/rsspolymlp/utils/matplot_util/readme.md)
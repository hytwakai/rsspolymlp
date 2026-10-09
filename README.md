# Framework for random structure search using polynomial MLPs

## Citation of rsspolymlp

If you use `rsspolymlp` in your study, please cite the following articles.

“Systematic global structure search of bismuth-based binary systems under pressure using machine learning potentials”, [Phys. Rev. Materials 10, 093801 (2026)](https://doi.org/10.1103/k9sg-v7hl)
```
@article{Phys.Rev.Mater.10.093801,
  title = {Systematic global structure search of bismuth-based binary systems under pressure using machine learning potentials},
  author = {Wakai, Hayato and Ishiwata, Shintaro and Seko, Atsuto},
  journal = {Phys. Rev. Mater.},
  volume = {10},
  issue = {9},
  pages = {093801},
  numpages = {28},
  year = {2026},
  month = {Sep},
  publisher = {American Physical Society},
  doi = {10.1103/k9sg-v7hl},
  url = {https://link.aps.org/doi/10.1103/k9sg-v7hl}
}
```

## Installation

### Required libraries and python modules

- python >= 3.10
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

## How to use rsspolymlp

 - [Workflow of RSS with polynomial MLPs](docs/rsspolymlp.md)
   - Initial structure generation
   - Global RSS with polynomial MLPs
   - Unique structure identification and RSS result summarization
   - Ghost minimum structure elimination
   - Phase stability analysis
 - [Development kit for polynomial MLPs](docs/rsspolymlp_dev.md)
   - MLP dataset generation
   - DFT dataset division
   - Polynomial MLP development
   - Pareto-optimal MLP selection
 - Python API
   - [RSS workflow](docs/api_rsspolymlp.md)
   - [VASP calculation utilities](src/rsspolymlp/utils/vasp_util/readme.md)
     - Single-point calculation
     - Local geometry optimization
   - [Matplotlib utilities](src/rsspolymlp/utils/matplot_util/readme.md)
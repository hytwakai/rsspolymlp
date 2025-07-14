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

## How to use rsspolymlp

 - [Workflow of RSS with polynomial MLPs](docs/rsspolymlp.md)
   - Initial structure generation
   - Global RSS with polynomial MLPs
   - Unique structure identification and RSS result summarization
   - Ghost minimum structure elimination
   - Phase stability analysis
 - [Development kit for polynomial MLPs](docs/rsspolymlp_devkit.md)
   - MLP dataset generation
   - DFT dataset division
   - Polynomial MLP development
   - Pareto-optimal MLP detection
 - [Python API for RSS](docs/api_rsspolymlp.md)
 - [VASP calculation utilities](src/rsspolymlp/utils/vasp_util/readme.md)
   - Single-point calculation
   - Local geometry optimizaion
 - [Matplotlib utilities](src/rsspolymlp/utils/matplot_util/readme.md)
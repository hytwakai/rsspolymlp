# rss_polymlp

## A framework for random structure search (RSS) using polynomial MLPs

### Required libraries and python modules

- python >= 3.9
- pypolymlp
- symfc
- spglib
- joblib

### Installation

```shell
git clone https://github.com/hytwakai/rss_polymlp.git
cd rss_polymlp
conda create -n rss_polymlp
conda activate rss_polymlp
conda install -c conda-forge pypolymlp symfc spglib joblib
pip install .
```

### Usage
```shell
python run_rss_parallel.py --elements Al Cu --n_atoms 4 4 --max_str 1000 --pot polymlp.yaml
```

### How to cite rss_polymlp

If you use `rss_polymlp` in your study, please cite the following articles.

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

# rsspolymlp

## A framework for random structure search (RSS) using polynomial MLPs

### Required libraries and python modules

- python >= 3.9
- pypolymlp
- symfc
- spglib
- joblib

### Installation

```shell
git clone https://github.com/hytwakai/rsspolymlp.git
cd rsspolymlp
conda create -n rsspolymlp
conda activate rsspolymlp
conda install -c conda-forge pypolymlp symfc spglib joblib
pip install .
```

### Usage
```shell
gen-rand-struct --elements Al Cu --atom_counts 4 4 --num_init_str 2000
rss-parallel --num_opt_str 1000 --pot polymlp.yaml
sort-struct
```

### How to cite rsspolymlp

If you use `rsspolymlp` in your study, please cite the following articles.

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

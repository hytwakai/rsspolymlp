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

```shell
conda create -n rsspolymlp python=3.11
conda activate rsspolymlp
conda install -c conda-forge pypolymlp symfc spglib joblib
pip install rsspolymlp
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

#### Arguments
- `--elements`: List of element symbols (e.g., `Al Cu`).
- `--atom_counts`: Number of atoms for each element (must match the order of `--elements`).
- `--num_init_str`: Number of random initial structures to generate. *(default: 5000)*
- `--pot`: Path to the polynomial MLP potential file. *(default: polymlp.yaml)*
- `--num_opt_str`: Maximum number of optimized structures to obtain from RSS. *(default: 1000)*
- [Additional information is here](docs/rss.md)
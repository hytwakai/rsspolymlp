```shell
python gen_mlp_dataset.py --poscars ./* --per_volume 1.0 --disp_max 40 --disp_grid 2
python compress_vasprun.py --path <dft_calculated_directries> --output_dir <output_dir>
python divide_dataset.py --path <dft_dataset_directory>
```
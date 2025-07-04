```shell
python gen_rand_struct.py --poscars ./* --per_volume 1.0 --disp_max 40 --disp_grid 2
python gen_dft_dataset.py --path <dft_calculated_directries> --output_dir <output_dir>
python divide_dataset.py --path <dft_dataset_directory>
```
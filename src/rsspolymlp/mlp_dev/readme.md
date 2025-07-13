```shell
python polymlp_dev.py --input_path model_candidates/single/polymlp-0001 --elements Li 
--train_data dft_dataset_divided/train/* --test_data dft_dataset_divided/test/*
```

```shell
python estimate_cost.py --path polymlp-0001
```

```shell
python pareto_opt_mlp.py --path polymlp-* --plot
```

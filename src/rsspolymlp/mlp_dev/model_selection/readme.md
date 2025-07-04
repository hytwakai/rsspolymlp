Generating model parameter candidates of single polynomial MLPs.
```shell
python ./gen_single_model.py --model_type 2
python ./gen_single_model.py --model_type 3
python ./gen_single_model.py --model_type 4
python ./gen_single_model.py --model_type pair
```

Generating model parameter candidates of hybrid polynomial MLPs.
```shell
python ./gen_hybrid_model.py --model_type 2
python ./gen_hybrid_model.py --model_type 4
```

Estimeting the number of polynomial invariants.
```shell
python ./n_feature.py --path ./polymlps_single_m2
python ./n_feature.py --path ./polymlps_hybrid_m2
```

Employing only model parameter candidates that use fewer than 80,000 polynomial invariants.
```shell
python ./reduce_grid.py --path ./polymlps_single_m2
python ./reduce_grid.py --path ./polymlps_hybrid_m2
```
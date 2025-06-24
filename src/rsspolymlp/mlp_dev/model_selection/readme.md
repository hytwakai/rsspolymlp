Generating model parameter candidates of single polynomial MLPs.
```python
python ./gen_single_model.py --model_type 2
python ./gen_single_model.py --model_type 3
python ./gen_single_model.py --model_type 4
python ./gen_single_model.py --model_type pair
```

Generating model parameter candidates of hybrid polynomial MLPs.
```python
python ./gen_hybrid_model.py --model_type 2
python ./gen_hybrid_model.py --model_type 4
```

Employing only model parameter candidates that use fewer than 80,000 polynomial invariants.
```ptyhon
python ./reduce_grid.py --path ./polymlps_single_m2
python ./reduce_grid.py --path ./polymlps_hybrid_m2
```

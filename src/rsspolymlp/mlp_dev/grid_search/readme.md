python ./gen_single_model.py --model_type 2
python ./gen_single_model.py --model_type 3
python ./gen_single_model.py --model_type 4
python ./gen_single_model.py --model_type pair

python ./gen_hybrid_model.py --model_type 2
python ./gen_hybrid_model.py --model_type 4

python ./reduce_grid.py --path ./polymlps_single_m2
python ./reduce_grid.py --path ./polymlps_hybrid_m2

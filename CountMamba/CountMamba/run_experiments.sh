#!/bin/bash

for dataset in CW_top200 CW_top300 CW_top400
do
  python main.py --dataset ${dataset} --log_transform --maximum_load_time 120 --max_matrix_len 2700 >> log_${dataset} 2>&1
  python test.py --dataset ${dataset} --log_transform --load_ratio 100 --result_file test_p100 --maximum_load_time 120 --max_matrix_len 2700 --mapping_file mapping >> log_${dataset} 2>&1
done


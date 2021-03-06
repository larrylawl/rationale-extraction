#!/usr/bin/env bash
set -e

dataset_name=esnli_fr
exp_name=gen_only/sup
root_data_dir=/temp/larry
lab_data_dir=$root_data_dir/data/esnli/$dataset_name
config=../models/config.json
out_dir=../out/$dataset_name/$exp_name
repeat=0
# model_dir=../out/$dataset_name/gen_only/partial_sup/0

# for ((i = 0; i < $repeat ; i++)); do   # forked if use ( &)
python main.py --lab_data_dir $lab_data_dir \
    --config $config \
    --out_dir $out_dir/$repeat \
    --seed $(( 100 + $repeat )) \
    --gen_only
# done
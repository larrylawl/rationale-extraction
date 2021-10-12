#!/usr/bin/env bash
set -e

dataset_name=esnli_lite_fr
exp_name=sup_weighted
data_dir=../data/esnli/$dataset_name
config=../models/config.json
out_dir=../out/$dataset_name/$exp_name
repeat=3

for ((i = 0; i < $repeat ; i++)); do   # forked if use ( &)
    python main.py --data_dir $data_dir \
        --config $config \
        --out_dir $out_dir/$i \
        --seed $(( 100 + $i )) \
        --tune_hp
done

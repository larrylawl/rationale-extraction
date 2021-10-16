#!/usr/bin/env bash
set -e

dataset_name=esnli_2000
exp_name=gen_only/converge
lab_data_dir=../data/esnli/$dataset_name
config=../models/config.json
out_dir=../out/$dataset_name/$exp_name
repeat=1
model_dir=../out/$dataset_name/gen_only/partial_sup/0

for ((i = 0; i < $repeat ; i++)); do   # forked if use ( &)
    python main.py --lab_data_dir $lab_data_dir \
        --config $config \
        --out_dir $out_dir/$i \
        --seed $(( 100 + $i )) \
        --gen_only
done

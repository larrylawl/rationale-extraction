#!/usr/bin/env bash
set -e

dataset_name=esnli_2000
exp_name=gen_only/partial_sup
data_dir=../data/esnli/$dataset_name
model_dir=../out/$dataset_name/$exp_name
config=../models/config.json
repeat=1

for ((i = 0; i < $repeat ; i++)); do   # forked if use ( &)
    python predict.py --data_dir $data_dir \
        --config $config \
        --model_dir $model_dir/$i \
        --seed $(( 100 + $i ))
        # --sup
        # --tune_hp
done

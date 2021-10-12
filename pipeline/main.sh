#!/usr/bin/env bash
set -e

dataset_name=esnli_lite_fr
exp_name=finetune
data_dir=../data/esnli/$dataset_name
config=../models/config.json
out_dir=../out/$dataset_name/$exp_name
model_dir=../out/esnli_lite/sup_pn/1
repeat=1

for ((i = 0; i < $repeat ; i++)); do   # forked if use ( &)
    python main.py --data_dir $data_dir \
        --config $config \
        --out_dir $out_dir/$i \
        --seed $(( 100 + $i )) \
        --model_dir $model_dir
done

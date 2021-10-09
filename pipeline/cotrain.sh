#!/usr/bin/env bash
set -e

dataset_name=esnli
exp_name=toy_exp

src_data_dir=../data/esnli/$dataset_name
tgt_data_dir=../data/esnli/${dataset_name}_fr
src_model_dir=../out/esnli_20k/self_sup/0
tgt_model_dir=../out/esnli_20k_fr/self_sup/0
config=../models/config.json
out_dir=../out/$dataset_name/$exp_name
repeat=1

for ((i = 0; i < $repeat ; i++)); do   # forked if use ( &)
    python cotrain.py --src_data_dir $src_data_dir \
        --tgt_data_dir $tgt_data_dir \
        --src_model_dir $src_model_dir \
        --tgt_model_dir $tgt_model_dir \
        --config $config \
        --out_dir $out_dir/$i \
        --seed $(( 100 + $i )) \
        # --tune_hp
done
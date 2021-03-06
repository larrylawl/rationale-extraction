#!/usr/bin/env bash
set -e

lab_dataset_name=esnli_1000
unlab_dataset_name=esnli_20000
exp_name=toy_exp
root_data_dir=..

repeat=0
src_lab_data_dir=$root_data_dir/data/esnli/$lab_dataset_name
tgt_lab_data_dir=$root_data_dir/data/esnli/${lab_dataset_name}_fr
src_unlab_data_dir=$root_data_dir/data/esnli/$unlab_dataset_name
tgt_unlab_data_dir=$root_data_dir/data/esnli/${unlab_dataset_name}_fr
# src_model_dir=/home/l/larrylaw/dso-fyp/cotrain/out/esnli_20000/cotrain/instance/0/src_0
# tgt_model_dir=/home/l/larrylaw/dso-fyp/cotrain/out/esnli_20000/cotrain/instance/0/tgt_0
src_model_dir=../out/$lab_dataset_name/gen_only/sup/$repeat
tgt_model_dir=../out/${lab_dataset_name}_fr/gen_only/sup/$repeat
config=../models/config.json
out_dir=../out/$unlab_dataset_name/$exp_name

python cotrain.py --src_lab_data_dir $src_lab_data_dir \
    --tgt_lab_data_dir $tgt_lab_data_dir \
    --src_unlab_data_dir $src_unlab_data_dir \
    --tgt_unlab_data_dir $tgt_unlab_data_dir \
    --src_model_dir $src_model_dir \
    --tgt_model_dir $tgt_model_dir \
    --config $config \
    --out_dir $out_dir/$repeat \
    --seed $(( 100 + $repeat )) \
    --overwrite_cache
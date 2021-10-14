#!/usr/bin/env bash
set -e

lab_dataset_name=esnli_1k
unlab_dataset_name=esnli_19k
exp_name=toy_exp

src_lab_data_dir=../data/esnli/$lab_dataset_name
tgt_lab_data_dir=../data/esnli/${dataset_name}_fr
src_unlab_data_dir=../data/esnli/$unlab_dataset_name
tgt_unlab_data_dir=../data/esnli/${unlab_dataset_name}_fr
src_model_dir=../out/esnli_20k/gen_only/partial_sup_only/0
tgt_model_dir=../out/esnli_20k_fr/gen_only/partial_sup_only/0
config=../models/config.json
out_dir=../out/$dataset_name/$exp_name
cotrain_rate=0.01
cotrain_epochs=1
repeat=1

for ((i = 0; i < $repeat ; i++)); do   # forked if use ( &)
    python cotrain.py --src_data_dir $src_data_dir \
        --tgt_data_dir $tgt_data_dir \
        --src_model_dir $src_model_dir \
        --tgt_model_dir $tgt_model_dir \
        --config $config \
        --out_dir $out_dir/$i \
        --seed $(( 100 + $i )) \
        --cotrain_epochs $cotrain_epochs \
        --cotrain_rate $cotrain_rate \
        --cotrain_perfect
        # --tune_hp
done

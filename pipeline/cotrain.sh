#!/usr/bin/env bash
set -e

lab_dataset_name=esnli_2000
unlab_dataset_name=esnli_lightest
exp_name=toy_exp

src_lab_data_dir=../data/esnli/esnli_lightest
tgt_lab_data_dir=../data/esnli/esnli_lightest_fr
src_unlab_data_dir=../data/esnli/$unlab_dataset_name
tgt_unlab_data_dir=../data/esnli/${unlab_dataset_name}_fr
src_model_dir=../out/esnli_2000/gen_only/partial_sup/0
tgt_model_dir=../out/esnli_2000/gen_only/partial_sup/0
config=../models/config.json
out_dir=../out/$dataset_name/$exp_name
repeat=1

for ((i = 0; i < $repeat ; i++)); do   # forked if use ( &)
    python cotrain.py --src_lab_data_dir $src_lab_data_dir \
        --tgt_lab_data_dir $tgt_lab_data_dir \
        --src_unlab_data_dir $src_unlab_data_dir \
        --tgt_unlab_data_dir $tgt_unlab_data_dir \
        --src_model_dir $src_model_dir \
        --tgt_model_dir $tgt_model_dir \
        --config $config \
        --out_dir $out_dir/$i \
        --seed $(( 100 + $i ))
        # --tune_hp
done

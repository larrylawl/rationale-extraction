#!/usr/bin/env bash
set -e

repo_path=/home/l/larrylaw/dso-fyp/cotrain
src_ipd=$repo_path/data/esnli/esnli
tgt_ipd=$repo_path/data/esnli/esnli_fr
train_size=1500
val_size=-1
test_size=-1
src_opd=$repo_path/data/esnli/esnli_${train_size}
tgt_opd=$repo_path/data/esnli/esnli_${train_size}_fr
seed=100
src_split_opd=$repo_path/data/esnli/esnli_${train_size}_comp
tgt_split_opd=$repo_path/data/esnli/esnli_${train_size}_comp_fr

python subset.py --src_ipd $src_ipd \
        --src_opd $src_opd \
        --tgt_ipd $tgt_ipd \
        --tgt_opd $tgt_opd \
        --train_size $train_size \
        --val_size $val_size \
        --test_size $test_size \
        --seed $seed \
        --split \
        --src_split_opd $src_split_opd \
        --tgt_split_opd $tgt_split_opd 

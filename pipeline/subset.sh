#!/usr/bin/env bash
set -e

repo_path=/home/l/larrylaw/dso-fyp/cotrain
src_ipd=$repo_path/data/esnli/esnli
tgt_ipd=$repo_path/data/esnli/esnli_fr
train_size=2000
val_size=1000
test_size=1000
src_opd=$repo_path/data/esnli/esnli_$train_size
tgt_opd=$repo_path/data/esnli/esnli_${train_size}_fr
seed=100

python subset.py $src_ipd $src_opd $tgt_ipd $tgt_opd $train_size $val_size $test_size $seed
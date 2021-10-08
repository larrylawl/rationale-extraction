#!/usr/bin/env bash
set -e

repo_path=/home/l/larrylaw/dso-fyp/cotrain
src_ipd=$repo_path/data/esnli/esnli
tgt_ipd=$repo_path/data/esnli/esnli_fr
train_size=20000
val_size=2000
test_size=2000
src_opd=$repo_path/data/esnli/esnli_20k
tgt_opd=$repo_path/data/esnli/esnli_20k_fr
seed=100

python subset.py $src_ipd $src_opd $tgt_ipd $tgt_opd $train_size $val_size $test_size $seed
#!/usr/bin/env bash
set -e
dataset_1=esnli_2000
dataset_2=esnli_19000

src_ipd_1=../data/esnli/$dataset_1
src_ipd_2=../data/esnli/$dataset_2
tgt_ipd_1=../data/esnli/${dataset_1}_fr
tgt_ipd_2=../data/esnli/${dataset_2}_fr
src_opd=../data/esnli/esnli_21000
tgt_opd=../data/esnli/esnli_21000_fr

python merge.py --src_ipd_1 $src_ipd_1 \
                --src_ipd_2 $src_ipd_2 \
                --tgt_ipd_1 $tgt_ipd_1 \
                --tgt_ipd_2 $tgt_ipd_2 \
                --src_opd $src_opd \
                --tgt_opd $tgt_opd
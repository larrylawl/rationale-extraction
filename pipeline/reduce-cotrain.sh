#!/usr/bin/env bash
set -e
ipd=../out/esnli_1000_comp/cotrain/vanilla/
opd=../out/esnli_1000_comp/cotrain/vanilla/avg
cotrain_repeats=2

rm -rf opd
for ((i = 0 ; i < $cotrain_repeats ; i++)); do
    mkdir -p $opd/$i
    tb-reducer -i "$ipd/*/src_$i" -o $opd/$i/src -r mean --lax-steps -f
    tb-reducer -i "$ipd/*/tgt_$i" -o $opd/$i/tgt -r mean --lax-steps -f
done
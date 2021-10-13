src_dir="../data/esnli/esnli_20k"
tgt_dir="../data/esnli/esnli_20k_fr"
op_dir="../data/esnli/esnli_20k_fr"

python translate_annotation.py --src_dir $src_dir \
    --tgt_dir $tgt_dir \
    --op_dir $op_dir
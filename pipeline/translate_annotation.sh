src_dir="../data/esnli/esnli_lightest"
tgt_dir="../data/esnli/esnli_lightest_fr"
op_dir="../data/esnli/esnli_lightest_fr_fixed"

python translate_annotation.py --src_dir $src_dir \
    --tgt_dir $tgt_dir \
    --op_dir $op_dir
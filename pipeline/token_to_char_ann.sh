set -e

repo_path=/home/l/larrylaw/dso-fyp/cotrain
data_dir=$repo_path/data/esnli/esnli_fr
output_dir=$repo_path/data/esnli_char/esnli_fr

python token_to_char_ann.py --data_dir $data_dir --output_dir $output_dir
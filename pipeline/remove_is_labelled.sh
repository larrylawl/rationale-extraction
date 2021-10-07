set -e

repo_path=/home/l/larrylaw/dso-fyp/cotrain
data_dir=$repo_path/data/esnli_old/esnli_fr
output_dir=$repo_path/data/esnli/esnli_fr

python remove_is_labelled.py --data_dir $data_dir --output_dir $output_dir
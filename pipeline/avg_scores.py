# This script takes in a dir of repeats and given json and outputs the avg of given json in the same directory.
# Run script using `avg_test_scores.sh`
import sys; sys.path.insert(0, "..")
import os
import argparse
from utils import read_json, write_json

def parse_args():
    parser = argparse.ArgumentParser("This script takes in a dir of repeats and outputs the all_scores.json. Assumes each json is metric:float pair.")
    parser.add_argument("exp_d", help="Experiment directory of repeats.")
    parser.add_argument("scores_fn", help="Filename of scores (e.g. poc_scores.json).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    all_scores = {}
    idx = 0
    for fn in os.listdir(args.exp_d):
        if not os.path.isdir(os.path.join(args.exp_d, fn)): continue

        js_fp = os.path.join(args.exp_d, fn, args.scores_fn)
        assert os.path.exists(js_fp), js_fp
        js = read_json(js_fp)
        for key, value in js.items():
            assert isinstance(key, str) and (isinstance(value, float) or isinstance(value, int))
            if key not in all_scores:
                assert idx == 0, f"{key} in {os.path.join(args.exp_d, fn)} is problematic."  # only need to create for first json. assumes all json to be the same.
                all_scores[key] = [value]
            else: all_scores[key].append(value)
        idx += 1  # not sure why enumerate jumps oddly (0, 3, 5)...
    
    avg_scores = {key: sum(value) / len(value) for (key,value) in all_scores.items()}
    for key, value in avg_scores.items():
        # ensuring results across repeats are not duplicated
        if value == all_scores[key][0]: print(f"WARNING: {key} has same value across repeats.")

    print(avg_scores)
    
    if os.path.exists(os.path.join(args.exp_d, f"avg_{args.scores_fn}")): os.remove(os.path.join(args.exp_d, f"avg_{args.scores_fn}"))
    write_json(avg_scores, os.path.join(args.exp_d, f"avg_{args.scores_fn}"))

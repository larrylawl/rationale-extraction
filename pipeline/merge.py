import sys; sys.path.insert(0, "..")
from copy import copy
import random
import os
import argparse
from utils import load_datasets, load_id_jsonl_as_dict, load_jsonl, write_jsonl, annotations_from_jsonl, annotations_to_jsonl
import shutil

def parse_args():
    parser = argparse.ArgumentParser("Outputs a subset of (train|val|test) directory.")
    parser.add_argument("--src_ipd_1", required=True, help="Input directory of (train|val|test).jsonl.")
    parser.add_argument("--src_ipd_2", required=True, help="Output directory of (train|val|test).jsonl.")
    parser.add_argument("--tgt_ipd_1", required=True, help="Input directory of (train|val|test).jsonl.")
    parser.add_argument("--tgt_ipd_2", required=True, help="Output directory of (train|val|test).jsonl.")
    parser.add_argument("--src_opd", required=True, help="Output directory of (train|val|test).jsonl.")
    parser.add_argument("--tgt_opd", required=True, help="Output directory of (train|val|test).jsonl.")


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if os.path.exists(args.src_opd): shutil.rmtree(args.src_opd)
    os.makedirs(args.src_opd)
    if os.path.exists(args.tgt_opd): shutil.rmtree(args.tgt_opd)
    os.makedirs(args.tgt_opd)

    # for each split
    ## load everything in, then extend then write
    splits = ["train", "val", "test", "docs", "wa"]
    for split in splits:
        tgt_1 = load_jsonl(os.path.join(args.tgt_ipd_1, f"{split}.jsonl"))
        tgt_2 = load_jsonl(os.path.join(args.tgt_ipd_2, f"{split}.jsonl"))
        tgt_res = tgt_1 + tgt_2
        write_jsonl(tgt_res, os.path.join(args.tgt_opd, f"{split}.jsonl"))

        if split != "wa":
            src_1 = load_jsonl(os.path.join(args.src_ipd_1, f"{split}.jsonl"))
            src_2 = load_jsonl(os.path.join(args.src_ipd_2, f"{split}.jsonl"))
            src_res = src_1 + src_2
            write_jsonl(src_res, os.path.join(args.src_opd, f"{split}.jsonl"))
            assert len(tgt_res) == len(src_res), f"{len(tgt_res)} != {len(src_res)}"
        
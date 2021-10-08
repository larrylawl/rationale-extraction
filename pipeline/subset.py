import sys; sys.path.insert(0, "..")
from copy import copy
import random
import os
import argparse
from utils import load_jsonl, write_jsonl
from shutil import copyfile

def parse_args():
    parser = argparse.ArgumentParser("Outputs a subset of (train|val|test) directory.")
    parser.add_argument("src_ipd", help="Input directory of (train|val|test).jsonl.")
    parser.add_argument("src_opd", help="Output directory of (train|val|test).jsonl.")
    parser.add_argument("tgt_ipd", help="Input directory of (train|val|test).jsonl.")
    parser.add_argument("tgt_opd", help="Output directory of (train|val|test).jsonl.")
    parser.add_argument("train_size")
    parser.add_argument("val_size")
    parser.add_argument("test_size")
    parser.add_argument("seed")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    assert os.path.isfile(os.path.join(args.src_ipd, "train.jsonl"))
    assert os.path.isfile(os.path.join(args.src_ipd, "val.jsonl"))
    assert os.path.isfile(os.path.join(args.src_ipd, "test.jsonl"))
    assert not os.path.exists(args.src_opd)
    assert not os.path.isfile(os.path.join(args.src_opd, "train.jsonl"))
    assert not os.path.isfile(os.path.join(args.src_opd, "val.jsonl"))
    assert not os.path.isfile(os.path.join(args.src_opd, "test.jsonl"))
    assert os.path.isfile(os.path.join(args.tgt_ipd, "train.jsonl"))
    assert os.path.isfile(os.path.join(args.tgt_ipd, "val.jsonl"))
    assert os.path.isfile(os.path.join(args.tgt_ipd, "test.jsonl"))
    assert not os.path.exists(args.tgt_opd)
    assert not os.path.isfile(os.path.join(args.tgt_opd, "train.jsonl"))
    assert not os.path.isfile(os.path.join(args.tgt_opd, "val.jsonl"))
    assert not os.path.isfile(os.path.join(args.tgt_opd, "test.jsonl"))

    src_train_jsl = load_jsonl(os.path.join(args.src_ipd, "train.jsonl"))
    src_val_jsl = load_jsonl(os.path.join(args.src_ipd, "val.jsonl"))
    src_test_jsl = load_jsonl(os.path.join(args.src_ipd, "test.jsonl"))
    tgt_train_jsl = load_jsonl(os.path.join(args.tgt_ipd, "train.jsonl"))
    tgt_val_jsl = load_jsonl(os.path.join(args.tgt_ipd, "val.jsonl"))
    tgt_test_jsl = load_jsonl(os.path.join(args.tgt_ipd, "test.jsonl"))

    assert len(src_train_jsl) == len(tgt_train_jsl)
    assert len(src_val_jsl) == len(tgt_val_jsl)
    assert len(src_test_jsl) == len(tgt_test_jsl)

    train_ids = set(random.sample(range(0, len(src_train_jsl)), int(args.train_size)))
    val_ids = set(random.sample(range(0, len(src_val_jsl)), int(args.val_size)))
    test_ids = set(random.sample(range(0, len(src_test_jsl)), int(args.test_size)))
    
    src_sub_train_jsl = []
    src_sub_val_jsl = []
    src_sub_test_jsl = []
    tgt_sub_train_jsl = []
    tgt_sub_val_jsl = []
    tgt_sub_test_jsl = []

    for i in range(len(src_train_jsl)):
        if i in train_ids:
            src_sub_train_jsl.append(src_train_jsl[i])
            tgt_sub_train_jsl.append(tgt_train_jsl[i])
    
    for i in range(len(src_val_jsl)):
        if i in val_ids:
            src_sub_val_jsl.append(src_val_jsl[i])
            tgt_sub_val_jsl.append(tgt_val_jsl[i])
    
    for i in range(len(src_test_jsl)):
        if i in test_ids:
            src_sub_test_jsl.append(src_test_jsl[i])
            tgt_sub_test_jsl.append(tgt_test_jsl[i])

    os.mkdir(args.src_opd)
    os.mkdir(args.tgt_opd)
    write_jsonl(src_sub_train_jsl, os.path.join(args.src_opd, "train.jsonl"))
    write_jsonl(src_sub_test_jsl, os.path.join(args.src_opd, "val.jsonl"))
    write_jsonl(src_sub_val_jsl, os.path.join(args.src_opd, "test.jsonl"))
    write_jsonl(tgt_sub_train_jsl, os.path.join(args.tgt_opd, "train.jsonl"))
    write_jsonl(tgt_sub_val_jsl, os.path.join(args.tgt_opd, "val.jsonl"))
    write_jsonl(tgt_sub_test_jsl, os.path.join(args.tgt_opd, "test.jsonl"))
    copyfile(os.path.join(args.src_ipd, "docs.jsonl"), os.path.join(args.src_opd, "docs.jsonl"))
    copyfile(os.path.join(args.tgt_ipd, "docs.jsonl"), os.path.join(args.tgt_opd, "docs.jsonl"))
    copyfile(os.path.join(args.tgt_ipd, "wa.jsonl"), os.path.join(args.tgt_opd, "wa.jsonl"))

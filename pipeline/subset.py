import sys; sys.path.insert(0, "..")
from copy import copy
import random
import os
import argparse
from utils import load_datasets, load_id_jsonl_as_dict, load_jsonl, write_jsonl, annotations_from_jsonl, annotations_to_jsonl
import shutil

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
    if os.path.exists(args.src_opd): shutil.rmtree(args.src_opd)
    os.makedirs(args.src_opd)
    if os.path.exists(args.tgt_opd): shutil.rmtree(args.tgt_opd)
    os.makedirs(args.tgt_opd)
    assert os.path.isfile(os.path.join(args.src_ipd, "train.jsonl"))
    assert os.path.isfile(os.path.join(args.src_ipd, "val.jsonl"))
    assert os.path.isfile(os.path.join(args.src_ipd, "test.jsonl"))
    assert not os.path.isfile(os.path.join(args.src_opd, "train.jsonl"))
    assert not os.path.isfile(os.path.join(args.src_opd, "val.jsonl"))
    assert not os.path.isfile(os.path.join(args.src_opd, "test.jsonl"))
    assert os.path.isfile(os.path.join(args.tgt_ipd, "train.jsonl"))
    assert os.path.isfile(os.path.join(args.tgt_ipd, "val.jsonl"))
    assert os.path.isfile(os.path.join(args.tgt_ipd, "test.jsonl"))
    assert not os.path.isfile(os.path.join(args.tgt_opd, "train.jsonl"))
    assert not os.path.isfile(os.path.join(args.tgt_opd, "val.jsonl"))
    assert not os.path.isfile(os.path.join(args.tgt_opd, "test.jsonl"))

    # load datasets
    src_all_ds = load_datasets(args.src_ipd)
    tgt_all_ds = load_datasets(args.tgt_ipd)
    src_docs = load_id_jsonl_as_dict(os.path.join(args.src_ipd, "docs.jsonl"))
    tgt_docs = load_id_jsonl_as_dict(os.path.join(args.tgt_ipd, "docs.jsonl"))
    wa = load_id_jsonl_as_dict(os.path.join(args.tgt_ipd, "wa.jsonl"))

    sub_wa = []
    src_sub_docs = []
    tgt_sub_docs = []
    sizes = [args.train_size, args.val_size, args.test_size]
    splits = ["train", "val", "test"]
    for i, split in enumerate(splits):
        ids = set(random.sample(range(0, len(src_all_ds[i])), int(sizes[i])))
        src_sub_ds = [src_all_ds[i][id] for id in ids]
        tgt_sub_ds = [tgt_all_ds[i][id] for id in ids]

        for j in range(len(src_sub_ds)):
            src_ann_id = src_sub_ds[j].annotation_id
            tgt_ann_id = tgt_sub_ds[j].annotation_id
            assert src_ann_id == tgt_ann_id
            doc_ids = [f"{tgt_ann_id}_hypothesis", f"{tgt_ann_id}_premise"]
            for doc_id in doc_ids:
                sub_wa.append({"docid": doc_id, "alignment": wa[doc_id]["alignment"]})
                src_sub_docs.append({"docid": doc_id, "document": src_docs[doc_id]["document"]})
                tgt_sub_docs.append({"docid": doc_id, "document": tgt_docs[doc_id]["document"]})
        
        annotations_to_jsonl(src_sub_ds, os.path.join(args.src_opd, f"{split}.jsonl"))
        annotations_to_jsonl(tgt_sub_ds, os.path.join(args.tgt_opd, f"{split}.jsonl"))
    write_jsonl(sub_wa, os.path.join(args.tgt_opd, "wa.jsonl"))
    write_jsonl(tgt_sub_docs, os.path.join(args.tgt_opd, "docs.jsonl"))
    write_jsonl(src_sub_docs, os.path.join(args.src_opd, "docs.jsonl"))
            
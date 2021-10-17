import sys; sys.path.insert(0, "..")
from copy import copy
import random
import os
import argparse
from utils import load_datasets, load_id_jsonl_as_dict, load_jsonl, write_jsonl, annotations_from_jsonl, annotations_to_jsonl
import shutil

def parse_args():
    parser = argparse.ArgumentParser("Outputs a subset of (train|val|test) directory.")
    parser.add_argument("--src_ipd", required=True, help="Input directory of (train|val|test).jsonl.")
    parser.add_argument("--src_opd", required=True, help="Output directory of (train|val|test).jsonl.")
    parser.add_argument("--tgt_ipd", required=True, help="Input directory of (train|val|test).jsonl.")
    parser.add_argument("--tgt_opd", required=True, help="Output directory of (train|val|test).jsonl.")
    parser.add_argument("--train_size", required=True)
    parser.add_argument("--val_size", required=True)
    parser.add_argument("--test_size", required=True)
    parser.add_argument("--seed", required=True, default=100)
    parser.add_argument("--split", action="store_true", default=False, help="If true, will output the complement set as well.")
    parser.add_argument("--src_split_opd")
    parser.add_argument("--tgt_split_opd")

    return parser.parse_args()

def get_sub_wa_and_docs(src_sub_ds, tgt_sub_ds, wa, src_docs, tgt_docs):
    sub_wa = []
    src_sub_docs = []
    tgt_sub_docs = []
    for j in range(len(src_sub_ds)):
        src_ann_id = src_sub_ds[j].annotation_id
        tgt_ann_id = tgt_sub_ds[j].annotation_id
        assert src_ann_id == tgt_ann_id
        doc_ids = [f"{tgt_ann_id}_hypothesis", f"{tgt_ann_id}_premise"]
        for doc_id in doc_ids:
            sub_wa.append({"docid": doc_id, "alignment": wa[doc_id]["alignment"]})
            src_sub_docs.append({"docid": doc_id, "document": src_docs[doc_id]["document"]})
            tgt_sub_docs.append({"docid": doc_id, "document": tgt_docs[doc_id]["document"]})
    return sub_wa, src_sub_docs, tgt_sub_docs
        

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    if os.path.exists(args.src_opd): shutil.rmtree(args.src_opd)
    os.makedirs(args.src_opd)
    if os.path.exists(args.tgt_opd): shutil.rmtree(args.tgt_opd)
    os.makedirs(args.tgt_opd)
    if args.split:
        if os.path.exists(args.tgt_split_opd): shutil.rmtree(args.tgt_split_opd)
        os.makedirs(args.tgt_split_opd)
        if os.path.exists(args.src_split_opd): shutil.rmtree(args.src_split_opd)
        os.makedirs(args.src_split_opd)

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
    if args.split:
        comp_sub_wa = []
        comp_src_sub_docs = []
        comp_tgt_sub_docs = []
    
    sizes = [args.train_size, args.val_size, args.test_size]
    splits = ["train", "val", "test"]
    for i, split in enumerate(splits):
        ids = set(random.sample(range(0, len(src_all_ds[i])), int(sizes[i])))
        src_sub_ds = [src_all_ds[i][id] for id in ids]
        tgt_sub_ds = [tgt_all_ds[i][id] for id in ids]

        annotations_to_jsonl(src_sub_ds, os.path.join(args.src_opd, f"{split}.jsonl"))
        annotations_to_jsonl(tgt_sub_ds, os.path.join(args.tgt_opd, f"{split}.jsonl"))

        s_wa, s_s_d, t_s_d = get_sub_wa_and_docs(src_sub_ds, tgt_sub_ds, wa, src_docs, tgt_docs)
        sub_wa.extend(s_wa)
        src_sub_docs.extend(s_s_d)
        tgt_sub_docs.extend(t_s_d)

        # do for complement set
        if args.split:
            comp_src_sub_ds = [ann for j, ann in enumerate(src_all_ds[i]) if j not in ids]
            comp_tgt_sub_ds = [ann for j, ann in enumerate(tgt_all_ds[i]) if j not in ids]               
            annotations_to_jsonl(comp_src_sub_ds, os.path.join(args.src_split_opd, f"{split}.jsonl"))
            annotations_to_jsonl(comp_tgt_sub_ds, os.path.join(args.tgt_split_opd, f"{split}.jsonl"))
            c_s_wa, c_s_s_d, c_t_s_d = get_sub_wa_and_docs(comp_src_sub_ds, comp_tgt_sub_ds, wa, src_docs, tgt_docs)
            comp_sub_wa.extend(c_s_wa)
            comp_src_sub_docs.extend(c_s_s_d)
            comp_tgt_sub_docs.extend(c_t_s_d)
            

    write_jsonl(sub_wa, os.path.join(args.tgt_opd, "wa.jsonl"))
    write_jsonl(tgt_sub_docs, os.path.join(args.tgt_opd, "docs.jsonl"))
    write_jsonl(src_sub_docs, os.path.join(args.src_opd, "docs.jsonl"))

    if args.split:
        write_jsonl(comp_sub_wa, os.path.join(args.tgt_split_opd, "wa.jsonl"))
        write_jsonl(comp_tgt_sub_docs, os.path.join(args.tgt_split_opd, "docs.jsonl"))
        write_jsonl(comp_src_sub_docs, os.path.join(args.src_split_opd, "docs.jsonl"))
            
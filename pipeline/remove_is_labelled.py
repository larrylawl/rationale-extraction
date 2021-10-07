import sys; sys.path.insert(0, "..")

import os
from typing import Dict, List
from utils import load_documents, Annotation, Evidence, annotations_to_jsonl, preprocess_line, load_jsonl, write_jsonl
from old_utils import load_flattened_documents
import nltk
import argparse
import unicodedata
import json

def parse_args():
    parser = argparse.ArgumentParser("Converts annotation from token level to character level. Useful when annotating using your own tokenizer.")
    parser.add_argument("--data_dir", required=True, help="Input directory to data.")
    parser.add_argument("--output_dir", required=True, help="Output directory of data.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    splits = ["train", "test", "val"]
    for split in splits:
        print(f"Converting {split}.jsonl...")
        jsl = load_jsonl(os.path.join(args.data_dir, f"{split}.jsonl"))
        for js in jsl: 
            del js['is_labelled']
            del js['is_perfect']
        write_jsonl(jsl, os.path.join(args.output_dir, f"{split}.jsonl"))

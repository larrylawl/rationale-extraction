import sys; sys.path.insert(0, "..")

import os
from typing import Dict, List
from utils import load_documents, Annotation, Evidence, annotations_to_jsonl, preprocess_line
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

    # load dict: doc id -> string
    documents: Dict[str, str] = load_documents(args.data_dir)
    flattened_documents: Dict[str, List[str]] = load_flattened_documents(args.data_dir)
    tokenizer = nltk.tokenize.WhitespaceTokenizer()

    splits = ["train", "test", "val"]
    for split in splits:
        print(f"Converting {split}.jsonl...")
        annotations_char: List[Annotation] = []
        # essentially copies annotations_from_jsonl, but mutates evidences
        with open(os.path.join(args.data_dir, f"{split}.jsonl"), 'r', encoding='utf-8') as inf:
            for line in inf:
                line = preprocess_line(line)
                content = json.loads(line)
                ev_groups = []
                # mutates evidences
                for ev_group in content['evidences']:
                    for ev in ev_group:
                        doc = documents[ev["docid"]]
                        doc_tokens = flattened_documents[ev["docid"]]
                        # sanity check that start and end tokens from flattened document matches text
                        doc_ev = " ".join(doc_tokens[ev["start_token"]: ev["end_token"]])
                        if doc_ev != ev["text"]: print(f"WARNING: {doc_ev} != {ev['text']}")  # multirc fails this due to encoding differences (e.g. ... vs …)

                        token_spans = list(tokenizer.span_tokenize(doc))
                        ev["start_char"] = token_spans[ev["start_token"]][0]
                        ev["end_char"] = token_spans[ev["end_token"] - 1][1]  # already end-exclusive
                        del ev["start_token"]
                        del ev["end_token"]

                        # sanity check that start and end characters match previously retrieved text
                        doc_chars = list(doc)
                        doc_ev_chars = "".join(doc_chars[ev["start_char"]: ev["end_char"]])
                        assert doc_ev_chars == doc_ev, f"{doc_ev_chars} != {doc_ev}"

                for ev_group in content['evidences']:
                    ev_group = tuple([Evidence(**ev) for ev in ev_group])
                    ev_groups.append(ev_group)
                content['evidences'] = frozenset(ev_groups)
                del content['is_labelled']
                del content['is_perfect']

                annotations_char.append(Annotation(**content))

        annotations_to_jsonl(annotations_char, os.path.join(args.output_dir, f"{split}.jsonl"), 'w+')

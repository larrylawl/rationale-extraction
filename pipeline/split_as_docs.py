import argparse

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

    # load in splits
    splits = ["train", "test", "val"]
    for split in splits:
        print(f"Converting {split}.jsonl...")
        
        # combine both documents
        # get word alignment


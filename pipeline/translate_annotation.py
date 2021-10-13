""" Example usage:
python translate/translate_annotation.py data/esnli_lite/docs.jsonl data/esnli_lite_fr/docs.jsonl data/esnli_lite/val.jsonl data/esnli_lite_fr/wa.jsonl data/esnli_lite_fr/val.jsonl
"""
import sys; sys.path.insert(0, "..")
import os
import shutil
import argparse
from utils import load_jsonl, write_jsonl, load_id_jsonl_as_dict, parse_alignment, closest_number

approx_label = 0

def parse_args():
    parser = argparse.ArgumentParser("Translates a (train|val|test).jsonl using 1) parallel docs.jsonl, and 2) word alignments of these docs.jsonl in i-j Pharaoh format.")
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--tgt_dir", required=True)
    parser.add_argument("--op_dir", required=True)

    return parser.parse_args()

def get_substring_from_tokens(str, start_tkn, end_tkn):
    """ Returns substring the spans from the start token index of given string (inclusive)
    to end token index (exclusive).
    """
    str_lst = str.strip().split()
    substr_lst = str_lst[start_tkn: end_tkn]
    substr = " ".join(substr_lst)
    return substr

def translate_tokens(algn, start_tkn, end_tkn, src="", tgt="", src_ann=""):
    """ Returns a tuple of translated start and end tokens based on word alignment.
    Assumes that words between start and end tokens will be between translated start and end tokens.
    Given that rationales tend to be short, this approximating assumption should have low margin of error.

    To check correctness, print out the translated text.
    """
    global approx_label
    assert start_tkn <= end_tkn
    p_algn = parse_alignment(algn)

    tgt_start_tkn = 10000
    tgt_end_tkn = -1
    for k in range(start_tkn, end_tkn):
        if k not in p_algn: continue  # not in alignment
        for v in p_algn[k]:
            if v <= tgt_start_tkn: tgt_start_tkn = v
            if v >= tgt_end_tkn: tgt_end_tkn = v

    if tgt_start_tkn == 10000 and tgt_end_tkn == -1:    # alignment may not include start or end token
        start_tkn = closest_number(p_algn.keys(), start_tkn)  # alignment may not include start or end         tokens
        end_tkn = closest_number(p_algn.keys(), end_tkn - 1)
        span = p_algn[start_tkn] + p_algn[end_tkn]
        tgt_start_tkn = min(span)  # min of both values as aligned word order may be swapped: crowded area => zone bondé
        tgt_end_tkn = max(span)

        approx_label += 1
    assert tgt_start_tkn <= tgt_end_tkn, f"{tgt_start_tkn} > {tgt_end_tkn}. {src} => {tgt} {src_ann}"
    tgt_end_tkn += 1  # end-exclusive

    return (tgt_start_tkn, tgt_end_tkn)


if __name__ == "__main__":
    args = parse_args()
    raise NotImplementedError("results worsened when I fixed the translate tokens... not too sure. might be better to use the old translate tokens")
    # if os.path.exists(args.op_dir): shutil.rmtree(args.op_dir)
    # os.makedirs(args.op_dir)
    # load parallel corpus as dict
    src_dict = load_id_jsonl_as_dict(os.path.join(args.src_dir, "docs.jsonl"))
    tgt_dict = load_id_jsonl_as_dict(os.path.join(args.tgt_dir, "docs.jsonl"))
    wa_dict = load_id_jsonl_as_dict(os.path.join(args.tgt_dir, "wa.jsonl"))

    for split in ["train", "val", "test"]:
        # load target jsonl
        tgt_ann_jsonl = load_jsonl(os.path.join(args.src_dir, f"{split}.jsonl"))  # loading as target as we'll be modifying it

        # translating start and end tokens; start and end sentences are always -1 for esnli
        for js in tgt_ann_jsonl:
            assert len(js["evidences"]) == 1  # weird nested list of only 1 elt
            evds = js["evidences"][0]
            for evd in evds:
                # sanity check that evidence text matches start and end tokens of source corpus
                src_doc = src_dict[evd["docid"]]
                src_txt = get_substring_from_tokens(src_doc["document"], evd["start_token"], evd["end_token"])
                assert src_txt == evd["text"], f"Start and end tokens of source corpus are not the same as text in annotation jsonl: {src_txt} != {evd['text']}"

                # get translated start, end tokens and text
                alignment = wa_dict[evd["docid"]]["alignment"]
                tgt_doc = tgt_dict[evd["docid"]]
                tgt_start_tkn, tgt_end_tkn = translate_tokens(alignment, evd["start_token"], evd["end_token"], src_doc['document'], tgt_doc['document'], src_txt)
                tgt_txt = get_substring_from_tokens(tgt_doc["document"], tgt_start_tkn, tgt_end_tkn)

                # modifying js
                # print(f"({evd['start_token']},{evd['end_token']}) => ({tgt_start_tkn},{tgt_end_tkn}) | {src_txt} => {tgt_txt} | {src_doc['document']} => {tgt_doc['document']}")  # checking correctness
                evd["start_token"] = tgt_start_tkn
                evd["end_token"] = tgt_end_tkn
                evd["text"] = tgt_txt
                assert tgt_txt, f"Target text should not be empty: {tgt_txt}"

        print(f"approx labelled pn: {approx_label / len(tgt_ann_jsonl)}")
        approx_label = 0

        # write out jsonl
        write_jsonl(tgt_ann_jsonl, os.path.join(args.op_dir, f"{split}.jsonl"))

import sys; sys.path.insert(0, "..")
from copy import deepcopy
import os
import time
import math
import copy
import datetime
import logging
import shutil
import torch.multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from pipeline.cotrain_utils import *
from utils import get_top_k_prob_mask, higher_conf, load_datasets, load_documents, load_id_jsonl_as_dict, get_optimizer, parse_alignment, read_json, same_label, write_json, add_offsets

logging.basicConfig(level=logging.INFO, format='%(relativeCreated)6d %(threadName)s %(message)s')

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Cotraining.")
    parser.add_argument("--src_lab_data_dir", required=True, help="Input directory to data.")
    parser.add_argument("--tgt_lab_data_dir", required=True, help="Input directory to data.")
    parser.add_argument("--src_unlab_data_dir", required=True, help="Input directory to data.")
    parser.add_argument("--tgt_unlab_data_dir", required=True, help="Input directory to data.")
    parser.add_argument("--src_model_dir", required=True, help="Model weights file path.")
    parser.add_argument("--tgt_model_dir", required=True, help="Model weights file path.")
    parser.add_argument("--config", required=True, help="Model config file.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tune_hp", action="store_true")
    parser.add_argument("--cotrain_perfect", action="store_true")
    parser.add_argument("--cotrain_rate", default=0.05, type=float, help="Train with [0, 1]% of supervised labels")
    parser.add_argument("--cotrain_patience", default=2, type=int)
    parser.add_argument("--cotrain_epochs", default=5, type=int)
    parser.add_argument("--seed", required=True, type=int, default=100)

    return parser.parse_args()

def add_wa_to_anns(src_train_anns, tgt_train_anns, src_was, tgt_was, src_documents, tgt_documents):
    for i in range(len(src_train_anns)):
        src_ann = src_train_anns[i]
        tgt_ann = tgt_train_anns[i]
        assert src_ann.annotation_id == tgt_ann.annotation_id

        # load in alignment for doc_h and doc_p 
        src_doc_h_wa: Dict[int, List[int]] = src_was[f"{src_ann.annotation_id}_hypothesis"]
        src_doc_p_wa: Dict[int, List[int]] = src_was[f"{src_ann.annotation_id}_premise"]
        tgt_doc_h_wa: Dict[int, List[int]] = tgt_was[f"{tgt_ann.annotation_id}_hypothesis"]
        tgt_doc_p_wa: Dict[int, List[int]] = tgt_was[f"{tgt_ann.annotation_id}_premise"]
        
        # add offsets based on premise alignments based on hypothesis docs
        src_offset = len(src_documents[f"{src_ann.annotation_id}_hypothesis"].split())
        tgt_offset = len(tgt_documents[f"{tgt_ann.annotation_id}_hypothesis"].split())

        src_ann.alignment = {**src_doc_h_wa, **add_offsets(src_doc_p_wa, src_offset, tgt_offset)}
        tgt_ann.alignment = {**tgt_doc_h_wa, **add_offsets(tgt_doc_p_wa, tgt_offset, src_offset)}
    return src_train_anns, tgt_train_anns

def get_algn_mask(anns, max_tokens):
    """ (i, j) indicates if the ith token of jth annotation has an alignment. 
    """

    algn_mask = torch.zeros(max_tokens, len(anns))
    for i, ann in enumerate(anns):
        for k in ann.alignment.keys(): algn_mask[k][i] = 1

    return algn_mask
    
def compute_top_k_prob_mask(gen, dataset, algn_mask, args, config):
    """ Returns prob mask tensor with only the top r% most confident tokens retained; remaining tokens are zeroed out.
    Size (L, N), where L denotes the longest sequence length and N denotes the training size.  """
    gen.eval()
    # shuffle false for prob_mask[:, 0] to correspond to annotation 0.
    dataloader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=False, collate_fn=pad_collate)
    running_scalar_labels = [f"tok_p", f"tok_r", f"tok_f1"]
    running_scalar_metrics = torch.zeros(len(running_scalar_labels))

    with torch.no_grad():
        # forward pass to obtain prob of all tokens
        prob_mask = torch.zeros(config["max_tokens"], len(dataset))
        r_mask = torch.zeros(config["max_tokens"], len(dataset))
        bs = config["train"]["batch_size"]
        for batch, (t_e_pad, t_e_lens, r_pad, _, ann_ids, _, is_l) in enumerate(tqdm(dataloader)): 
            mask = gen(t_e_pad, t_e_lens)  # (L, bs)
            assert mask.size() == r_pad.size()
            # excluding rationales which are labelled
            mask[:, is_l == 1] = 0.51  # 0.51 instead of 0.50 so that it's selected over tokens with no alignment
            t_e_lens[is_l == 1] = 0

            prob_mask[:, batch*bs:(batch+1)*bs] = F.pad(mask.T, (0, config["max_tokens"] - len(mask)), value=0.5).T # (max_tokens, bs), pad to max_tokens
            r_mask[:, batch*bs:(batch+1)*bs] = F.pad(r_pad.T, (0, config["max_tokens"] - len(r_pad)), value=-1).T # (max_tokens, bs), -1 to not clash with 0 denoting not rationale

            mask_hard = (mask > 0.5).float()
            tok_p, tok_r, tok_f1 = score_hard_rationale_predictions(mask_hard, r_pad, t_e_lens, average="micro")  # micro for valid comparison with top k scores
            running_scalar_metrics += torch.tensor([tok_p, tok_r, tok_f1])

        # label top k% of most confident tokens
        prob_mask_dup = prob_mask.clone()
        prob_mask_dup[algn_mask == 0] = 0.5 # ensure that tokens with no alignment (including padding) are not selected

        total_tokens = torch.count_nonzero(algn_mask)
        k = math.ceil(args.cotrain_rate * total_tokens)
        top_k_prob_mask = get_top_k_prob_mask(prob_mask_dup, k)# (max_tokens, trg_size)
        del prob_mask_dup

        # perfect labelling instead
        if args.cotrain_perfect: 
            top_k_r_mask = r_mask.clone()
            top_k_r_mask[top_k_prob_mask == -1] = -1
            assert torch.equal((top_k_r_mask + 1).nonzero(), (top_k_prob_mask + 1).nonzero()), f"r_mask should take index from top_k_prob_mask"
            top_k_prob_mask = top_k_r_mask
        
        # micro token f1 of chosen tokens
        top_k_pred = (top_k_prob_mask[top_k_prob_mask != -1] > 0.5).float()
        p, r, f1 = PRFScore(average="binary")(r_mask[top_k_prob_mask != -1], top_k_pred)
        # if config["train"]["cotrain_perfect"]: assert p == r == f1 == 1, f"{r_mask[top_k_prob_mask != -1]}, {top_k_prob_mask[top_k_prob_mask != -1]}"
        scalar_metrics = {  
            "top_k_p": p,
            "top_k_r": r,
            "top_k_f1": f1
        }

        total_scalar_metrics = running_scalar_metrics / (batch + 1)
        for i in range(len(running_scalar_labels)): scalar_metrics[running_scalar_labels[i]] = total_scalar_metrics[i]

    return top_k_prob_mask, scalar_metrics, r_mask, prob_mask

def cotrain(src_gen, tgt_gen, src_train_dataset, tgt_train_dataset, src_algn_mask, tgt_algn_mask, args, config, device, label_fns=[same_label, higher_conf]) -> DataLoader:
    """ Augments Eraserdataset with self-labels. """

    # src_future = torch.jit.fork(compute_top_k_prob_mask, src_gen, src_train_dataset, src_algn_mask, rate)
    # tgt_future = torch.jit.fork(compute_top_k_prob_mask, tgt_gen, tgt_train_dataset, tgt_algn_mask, rate)
    # src_top_k_prob_mask, src_scalar_metrics, src_r_mask, src_prob_mask = torch.jit.wait(src_future)
    # tgt_top_k_prob_mask, tgt_scalar_metrics, tgt_r_mask, tgt_prob_mask = torch.jit.wait(tgt_future)

    # p = mp.Process(target=compute_top_k_prob_mask, args=(src_gen, src_train_dataset, src_algn_mask, args, config))
    # p.start()
    # print("parallelising!")
    # compute_top_k_prob_mask(src_gen, src_train_dataset, src_algn_mask, args, config)
    # p.join()
    # exit(1)

    src_top_k_prob_mask, src_scalar_metrics, src_r_mask, src_prob_mask = compute_top_k_prob_mask(src_gen, src_train_dataset, src_algn_mask, args, config)
    logger.info(src_scalar_metrics)
    tgt_top_k_prob_mask, tgt_scalar_metrics, tgt_r_mask, tgt_prob_mask = compute_top_k_prob_mask(tgt_gen, tgt_train_dataset, tgt_algn_mask, args, config)
    logger.info(tgt_scalar_metrics)


    # Co-Training
    # Removes labels which are conflicting.
    # NOTE: looping is fine since number of self labels are small. if r == 1, should vectorize
    src_idxs = (src_top_k_prob_mask + 1).nonzero()
    tgt_idxs = (tgt_top_k_prob_mask + 1).nonzero()
    denied_labels = 0
    success_labels = 0
    for tkn_idx, ann_idx in src_idxs:
        src_wa = src_train_dataset.anns[ann_idx].alignment
        for v in src_wa[tkn_idx.item()]:  # implicitly asserts tkn in alignment
            src_label = src_r_mask[tkn_idx, ann_idx]
            tgt_label = tgt_r_mask[v, ann_idx]
            src_prob = src_top_k_prob_mask[tkn_idx, ann_idx]
            tgt_prob = tgt_prob_mask[v, ann_idx]  # NOTE: not top_k to emphasise on easier examples
            tgt_has_label = tgt_top_k_prob_mask[v, ann_idx] != -1

            if (src_label != tgt_label) or (tgt_has_label and not label(src_prob, tgt_prob, label_fns)):
                src_top_k_prob_mask[tkn_idx, ann_idx] = -1
                denied_labels += 1
            else: 
                tgt_top_k_prob_mask[v, ann_idx] = src_prob
                success_labels += 1       

    for tkn_idx, ann_idx in tgt_idxs:
        tgt_wa = tgt_train_dataset.anns[ann_idx].alignment
        for v in tgt_wa[tkn_idx.item()]: 
            tgt_label = tgt_r_mask[tkn_idx, ann_idx]
            src_label = src_r_mask[v, ann_idx]
            tgt_prob = tgt_top_k_prob_mask[tkn_idx, ann_idx]
            src_prob = src_prob_mask[v, ann_idx]
            src_has_label = src_top_k_prob_mask[v, ann_idx] != -1

            if (src_label != tgt_label) or (src_has_label and not label(tgt_prob, src_prob, label_fns)):
                tgt_top_k_prob_mask[tkn_idx, ann_idx] = -1  # skip self-supervising own labels that are problematic
                denied_labels += 1
            else:
                src_top_k_prob_mask[v, ann_idx] = tgt_prob
                success_labels += 1
    
    src_top_k_pred = (src_top_k_prob_mask[src_top_k_prob_mask != -1] > 0.5).float()
    src_p, src_r, src_f1 = PRFScore(average="binary")(src_r_mask[src_top_k_prob_mask != -1], src_top_k_pred)

    tgt_top_k_pred = (tgt_top_k_prob_mask[tgt_top_k_prob_mask != -1] > 0.5).float()
    tgt_p, tgt_r, tgt_f1 = PRFScore(average="binary")(tgt_r_mask[tgt_top_k_prob_mask != -1], tgt_top_k_pred)
    overall_scalar_metrics = {
        "src_top_k_p": src_p, "src_top_k_r": src_r, "src_top_k_f1": src_f1,
        "tgt_top_k_p": tgt_p, "tgt_top_k_r": tgt_r, "tgt_top_k_f1": tgt_f1,
        "denied_labels": denied_labels / (len(src_idxs) + len(tgt_idxs)), "success_labels_pn": success_labels / (len(src_idxs) + len(tgt_idxs))
    }
    logger.info(overall_scalar_metrics)

    assert src_top_k_prob_mask.size(1) == len(src_train_dataset), f"Col i of prob mask corresponds to self labels for ith annotation. {len(src_top_k_prob_mask)} != {len(src_train_dataset)}"
    assert tgt_top_k_prob_mask.size(1) == len(tgt_train_dataset), f"Col i of prob mask corresponds to self labels for ith annotation. {len(tgt_top_k_prob_mask)} != {len(tgt_train_dataset)}"
    assert src_top_k_prob_mask.size() == tgt_top_k_prob_mask.size()
    src_train_dataset.cotrain_mask = src_top_k_prob_mask.to(device)
    tgt_train_dataset.cotrain_mask = tgt_top_k_prob_mask.to(device)
    return src_train_dataset, tgt_train_dataset, overall_scalar_metrics

def main():
    start_time = time.time()
    args = parse_args()
    logger.info(args)
    set_seed(args)
    
    if os.path.exists(args.out_dir): shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)
    write_json(vars(args), os.path.join(args.out_dir, "exp_args.json"))

    config = read_json(args.config)
    config["generator"]["selection_lambda"] = 0  # we have supervision here
    config["generator"]["continuity_lambda"] = 0
    if args.tune_hp:
        config = tune_hp(config)
    assert 0 <= args.cotrain_rate <= 1
    write_json(config, os.path.join(args.out_dir, "config.json"))
    config["encoder"]["num_classes"] = len(dataset_mapping)

    tokenizer = AutoTokenizer.from_pretrained(config["embedding_model_name"])
    embedding_model = AutoModel.from_pretrained(config["embedding_model_name"], output_hidden_states=True)
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    embedding_model.to(device)
    embedding_model.eval()  # only extracting pre-trained embeddings

    # setting up data
    was = load_id_jsonl_as_dict(os.path.join(args.tgt_data_dir, "wa.jsonl"))
    src_was: Dict[str, List[int]] = {}  # maps from src to tgt
    tgt_was: Dict[str, List[int]] = {}  # maps from tgt to src
    for k, v in was.items(): 
        src_was[k] = parse_alignment(v["alignment"])
        tgt_was[k] = parse_alignment(v["alignment"], reverse=True)

    # purely creates datasets for me
    # TODO: val and tgt should combine labelled and unlabelled
    # create ul ds
    # create l ds
    # augment l ds
    src_l_ds = create_datasets_features(load_datasets(args.src_lab_data_dir), load_documents(args.src_lab_data_dir), device)
    tgt_l_ds = create_datasets_features(load_datasets(args.tgt_lab_data_dir), load_documents(args.tgt_lab_data_dir), device)

    src_ul_documents: Dict[str, str] = load_documents(args.src_unlab_data_dir)
    tgt_ul_documents: Dict[str, str] = load_documents(args.tgt_unlab_data_dir)
    src_ul_ds = create_datasets_features(load_datasets(args.src_unlab_data_dir), src_ul_documents, device)
    tgt_ul_ds = create_datasets_features(load_datasets(args.tgt_unlab_data_dir), tgt_ul_documents, device)
    src_train_feat, tgt_train_feat = add_wa_to_anns(src_ul_ds[0], tgt_ul_ds[0], src_was, tgt_was, src_ul_documents, tgt_ul_documents)
    src_ul_ds[0] = src_train_feat
    tgt_ul_ds[0] = tgt_train_feat
    src_algn_mask = get_algn_mask(src_train_feat, config["max_tokens"])
    tgt_algn_mask = get_algn_mask(tgt_train_feat, config["max_tokens"])

    # create train dataloader later
    # NOTE: for train, need to indicate which are sup since we'll be labelling other examples too
    src_l_ds = [EraserDataset(ds, tokenizer, embedding_model, True) for ds in src_l_ds]
    src_ul_ds = [EraserDataset(ds, tokenizer, embedding_model, False) for ds in src_ul_ds]
    tgt_l_ds = [EraserDataset(ds, tokenizer, embedding_model, True) for ds in tgt_l_ds]
    tgt_ul_ds = [EraserDataset(ds, tokenizer, embedding_model, False) for ds in tgt_ul_ds]

    src_val_ds = 

    src_train_dataset = EraserDataset(src_train_feat, tokenizer, embedding_model, config["train"]["sup_pn"])  
    src_val_dataset = EraserDataset(src_val_feat, tokenizer, embedding_model)
    src_test_dataset = EraserDataset(src_test_feat, tokenizer, embedding_model)
    src_val_dataloader = DataLoader(src_val_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    src_test_dataloader = DataLoader(src_test_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)

    tgt_train_dataset = EraserDataset(tgt_train_feat, tokenizer, embedding_model, config["train"]["sup_pn"])
    tgt_val_dataset = EraserDataset(tgt_val_feat, tokenizer, embedding_model)
    tgt_test_dataset = EraserDataset(tgt_test_feat, tokenizer, embedding_model)
    tgt_val_dataloader = DataLoader(tgt_val_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    tgt_test_dataloader = DataLoader(tgt_test_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    exit(1)
    
    # instantiate models
    src_gen = instantiate_generator(config, device, os.path.join(args.src_model_dir, "best_gen_weights.pth"))
    tgt_gen = instantiate_generator(config, device, os.path.join(args.tgt_model_dir, "best_gen_weights.pth"))
    best_src_gen = copy.deepcopy(src_gen)  # won't be trained
    best_tgt_gen = copy.deepcopy(tgt_gen)

    # instantiate optimiser
    src_optimizer = get_optimizer([src_gen], config["train"]["lr"])
    tgt_optimizer = get_optimizer([tgt_gen], config["train"]["lr"])

    co_best_src_val_metrics = get_best_val_metrics(read_json(os.path.join(args.src_model_dir, "results.json")))
    co_best_tgt_val_metrics = get_best_val_metrics(read_json(os.path.join(args.tgt_model_dir, "results.json")))
    co_best_val_target_metric = co_best_src_val_metrics + co_best_tgt_val_metrics
    # co_epochs = math.ceil((1 - config["train"]["sup_pn"]) / (args.cotrain_rate * 2))  # NOTE: *2 since combining both src and tgt labels
    co_es_count = 0
    co_writer = SummaryWriter(args.out_dir)
    for co_t in range(args.cotrain_epochs):
        logger.info(f"Cotrain Epochs {co_t+1}\n-------------------------------")
        # augment train datasets with cotrain masks
        src_train_dataset, tgt_train_dataset, cotrain_scalar_metrics = cotrain(best_src_gen, best_tgt_gen, src_train_dataset, tgt_train_dataset, src_algn_mask, tgt_algn_mask, args, config, device)
        for tag, val in cotrain_scalar_metrics.items():
            co_writer.add_scalar(tag, val, co_t)

        # fine-tuning C_k 
        # src_out_dir = os.path.join(args.out_dir, f"src_{co_t}")
        # src_writer = SummaryWriter(src_out_dir)
        # src_train_dataloader = DataLoader(src_train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
        # src_gen, _, src_best_val_scalar_metrics = train_loop(src_train_dataloader, src_val_dataloader, src_gen, None, src_optimizer, config["train"]["num_epochs"], 
        #             src_out_dir, src_writer, device, config["train"]["patience"], logger)
        # src_best_val_scalar_metrics = get_best_val_metrics(src_best_val_scalar_metrics)
        
        tgt_out_dir = os.path.join(args.out_dir, f"tgt_{co_t}")
        tgt_writer = SummaryWriter(tgt_out_dir)
        tgt_train_dataloader = DataLoader(tgt_train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
        tgt_gen, _, tgt_best_val_scalar_metrics = train_loop(tgt_train_dataloader, tgt_val_dataloader, tgt_gen, None, tgt_optimizer, config["train"]["num_epochs"], 
                    tgt_out_dir, tgt_writer, device, config["train"]["patience"], logger)
        tgt_best_val_scalar_metrics = get_best_val_metrics(tgt_best_val_scalar_metrics)
        
        # saving best models
        # if src_best_val_scalar_metrics > co_best_src_val_metrics: 
        #     co_best_src_val_metrics = src_best_val_scalar_metrics
        #     best_src_gen = copy.deepcopy(src_gen)
        # if tgt_best_val_scalar_metrics > co_best_tgt_val_metrics:
        #     co_best_tgt_val_metrics = tgt_best_val_scalar_metrics
        #     best_tgt_gen = copy.deepcopy(tgt_gen)
        
        # co-train early stopping
        # val_target_metric = src_best_val_scalar_metrics + tgt_best_val_scalar_metrics
        # if val_target_metric > co_best_val_target_metric: 
        #     co_best_val_target_metric = val_target_metric
        #     co_es_count = 0
        # else:
        #     co_es_count += 1
        #     if co_es_count >= args.cotrain_patience:
        #         logger.info("Early stopping co-training!")
        #         break
        # src_writer.close()
        tgt_writer.close()


    co_src_test_scalar_metrics = test(src_test_dataloader, None, best_src_gen, logger, split="src_test")
    co_tgt_test_scalar_metrics = test(tgt_test_dataloader, None, best_tgt_gen, logger, split="tgt_test")
    co_test_scalar_metrics = {**co_src_test_scalar_metrics, **co_tgt_test_scalar_metrics}
    for tag, val in co_test_scalar_metrics.items(): 
        co_writer.add_scalar(tag, val)
    co_test_scalar_metrics["total_time"] = str(datetime.timedelta(seconds=time.time() - start_time))
    write_json(co_test_scalar_metrics, os.path.join(args.out_dir, str(co_t), "results.json"))

if __name__ == "__main__":
    main()
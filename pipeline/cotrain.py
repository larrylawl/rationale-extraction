import sys; sys.path.insert(0, "..")
import os
import time
import math
import datetime
import logging
import shutil
import multiprocessing as mp
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
from utils import get_top_k_prob_mask, instantiate_models, load_datasets, create_instance, load_documents, load_id_jsonl_as_dict, get_optimizer, parse_alignment, read_json, top_k_idxs_multid, write_json, add_offsets

logging.basicConfig(level=logging.INFO, format='%(relativeCreated)6d %(threadName)s %(message)s')

logger = logging.getLogger(__name__)
args = None
device = None
writer = None
config = None
avg_tokens = 16  # from ERASER paper
max_tokens = 113  # assume all sequences ≤ 113 tokens (correct for esnli)

def parse_args():
    parser = argparse.ArgumentParser("Cotraining.")
    parser.add_argument("--src_data_dir", required=True, help="Input directory to data.")
    parser.add_argument("--tgt_data_dir", required=True, help="Input directory to data.")
    parser.add_argument("--src_model_dir", required=True, help="Model weights file path.")
    parser.add_argument("--tgt_model_dir", required=True, help="Model weights file path.")
    parser.add_argument("--config", required=True, help="Model config file.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tune_hp", action="store_true")
    parser.add_argument("--cotrain_perfect", action="store_true")
    parser.add_argument("--cotrain_rate", default=0.05, type=float, help="Train with [0, 1]% of supervised labels")
    parser.add_argument("--cotrain_patience", default=2, type=int)
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

def get_algn_mask(anns):
    """ (i, j) indicates if the ith token of jth annotation has an alignment. """

    algn_mask = torch.zeros(max_tokens, len(anns))
    for i, ann in enumerate(anns):
        for k in ann.alignment.keys(): algn_mask[k][i] = 1
        # print(algn_mask[:, i])
        # print(ann.alignment.keys())
    return algn_mask
    
def compute_top_k_prob_mask(gen, dataset, algn_mask, k):
    """ Returns prob mask tensor with only the top k most confident tokens retained; remaining tokens are zeroed out.
    Size (L, N), where L denotes the longest sequence length and N denotes the training size.  """

    gen.eval()
    # shuffle false for prob_mask[:, 0] to correspond to annotation 0.
    dataloader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=False, collate_fn=pad_collate)
    running_scalar_labels = [f"tok_p", f"tok_r", f"tok_f1"]
    running_scalar_metrics = torch.zeros(len(running_scalar_labels))

    with torch.no_grad():
        # forward pass to obtain prob of all tokens
        prob_mask = torch.zeros(max_tokens, len(dataset))
        r_mask = torch.zeros(max_tokens, len(dataset))
        bs = config["train"]["batch_size"]
        for batch, (t_e_pad, t_e_lens, r_pad, _, ann_ids, _) in enumerate(tqdm(dataloader)): 
            mask = gen(t_e_pad, t_e_lens)  # (L, bs)
            assert mask.size() == r_pad.size()
            prob_mask[:, batch*bs:(batch+1)*bs] = F.pad(mask.T, (0, max_tokens - len(mask)), value=0.5).T # (max_tokens, bs), pad to max_tokens
            r_mask[:, batch*bs:(batch+1)*bs] = F.pad(r_pad.T, (0, max_tokens - len(r_pad)), value=0.5).T # (max_tokens, bs)

            mask_hard = (mask.detach() > 0.5).float() - mask.detach() + mask  
            tok_p, tok_r, tok_f1 = score_hard_rationale_predictions(mask_hard.detach(), r_pad.detach(), t_e_lens, average="micro")  # micro for valid comparison with top k scores
            running_scalar_metrics += torch.tensor([tok_p, tok_r, tok_f1])

        # label top k% of most confident tokens
        prob_mask_dup = prob_mask.clone()
        prob_mask_dup[algn_mask == 0] = 0.5 # ensure that tokens with no alignment (including padding) are not selected
        top_k_prob_mask = get_top_k_prob_mask(prob_mask_dup, k)  # (max_tokens, trg_size)
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
        
def cotrain(src_gen, tgt_gen, src_train_dataset, tgt_train_dataset, src_algn_mask, tgt_algn_mask, rate) -> DataLoader:
    """ Augments Eraserdataset with self-labels. """
    src_total_tokens = torch.count_nonzero(src_algn_mask)
    src_size = math.ceil(rate * src_total_tokens)
    tgt_total_tokens = torch.count_nonzero(tgt_algn_mask)
    tgt_size = math.ceil(rate * tgt_total_tokens)

    # src_future = torch.jit.fork(compute_top_k_prob_mask, src_gen, src_train_dataset, src_algn_mask, src_size)
    # tgt_future = torch.jit.fork(compute_top_k_prob_mask, tgt_gen, tgt_train_dataset, tgt_algn_mask, tgt_size)
    # src_top_k_prob_mask, src_scalar_metrics, src_r_mask, src_prob_mask = torch.jit.wait(src_future)
    # tgt_top_k_prob_mask, tgt_scalar_metrics, tgt_r_mask, tgt_prob_mask = torch.jit.wait(tgt_future)

    src_top_k_prob_mask, src_scalar_metrics, src_r_mask, src_prob_mask = compute_top_k_prob_mask(src_gen, src_train_dataset, src_algn_mask, src_size)
    tgt_top_k_prob_mask, tgt_scalar_metrics, tgt_r_mask, tgt_prob_mask = compute_top_k_prob_mask(tgt_gen, tgt_train_dataset, tgt_algn_mask, tgt_size)
    print(src_scalar_metrics)
    print(tgt_scalar_metrics)


    # Co-Training
    # Removes labels which are conflicting.
    # NOTE: looping is fine since number of self labels are small
    src_idxs = (src_top_k_prob_mask + 1).nonzero()
    tgt_idxs = (tgt_top_k_prob_mask + 1).nonzero()
    conflicting_labels = 0
    for tkn_idx, ann_idx in src_idxs:
        src_wa = src_train_dataset.anns[ann_idx].alignment
        for v in src_wa[tkn_idx.item()]:  # implicitly asserts tkn in alignment
            src_label = src_top_k_prob_mask[tkn_idx, ann_idx] > 0.5
            tgt_label = tgt_top_k_prob_mask[v, ann_idx] > 0.5
            tgt_no_label = tgt_top_k_prob_mask[v, ann_idx] == -1
            if tgt_no_label or src_label == tgt_label:
                tgt_top_k_prob_mask[v, ann_idx] = src_top_k_prob_mask[tkn_idx, ann_idx]
            else:
                conflicting_labels += 1

            # if src_label == tgt_label: 
            #     tgt_top_k_prob_mask[v, ann_idx] = src_top_k_prob_mask[tkn_idx, ann_idx]
            # else: 
            #     conflicting_labels += 1            

    for tkn_idx, ann_idx in tgt_idxs:
        tgt_wa = tgt_train_dataset.anns[ann_idx].alignment
        for v in tgt_wa[tkn_idx.item()]: 
            tgt_label = tgt_top_k_prob_mask[tkn_idx, ann_idx] > 0.5
            src_label = src_top_k_prob_mask[v, ann_idx] > 0.5
            src_no_label = src_top_k_prob_mask[v, ann_idx] == -1

            if src_no_label or src_label == tgt_label:
                src_top_k_prob_mask[v, ann_idx] = tgt_top_k_prob_mask[tkn_idx, ann_idx]
            else:
                conflicting_labels += 1

            # src_top_k_prob_mask[v, ann_idx] = tgt_top_k_prob_mask[tkn_idx, ann_idx]
            # if src_label == tgt_label: 
            #     src_top_k_prob_mask[v, ann_idx] = tgt_top_k_prob_mask[tkn_idx, ann_idx]
            # else: 
            #     conflicting_labels += 1
    
    src_top_k_pred = (src_top_k_prob_mask[src_top_k_prob_mask != -1] > 0.5).float()
    src_p, src_r, src_f1 = PRFScore(average="binary")(src_r_mask[src_top_k_prob_mask != -1], src_top_k_pred)

    tgt_top_k_pred = (tgt_top_k_prob_mask[tgt_top_k_prob_mask != -1] > 0.5).float()
    tgt_p, tgt_r, tgt_f1 = PRFScore(average="binary")(tgt_r_mask[tgt_top_k_prob_mask != -1], tgt_top_k_pred)
    overall_scalar_metrics = {
        "src_top_k_p": src_p, "src_top_k_r": src_r, "src_top_k_f1": src_f1,
        "tgt_top_k_p": tgt_p, "tgt_top_k_r": tgt_r, "tgt_top_k_f1": tgt_f1,
        "conflicting_labels": conflicting_labels
    }

    assert src_top_k_prob_mask.size(1) == len(src_train_dataset), f"Col i of prob mask corresponds to self labels for ith annotation. {len(src_top_k_prob_mask)} != {len(src_train_dataset)}"
    assert tgt_top_k_prob_mask.size(1) == len(tgt_train_dataset), f"Col i of prob mask corresponds to self labels for ith annotation. {len(tgt_top_k_prob_mask)} != {len(tgt_train_dataset)}"
    assert src_top_k_prob_mask.size() == tgt_top_k_prob_mask.size()
    src_train_dataset.cotrain_mask = src_top_k_prob_mask.to(device)
    tgt_train_dataset.cotrain_mask = tgt_top_k_prob_mask.to(device)
    return src_train_dataset, tgt_train_dataset, overall_scalar_metrics

def main():
    start_time = time.time()
    global args, device, writer, config
    args = parse_args()
    logger.info(args)
    set_seed(args)
    
    if os.path.exists(args.out_dir): shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    write_json(vars(args), os.path.join(args.out_dir, "exp_args.json"))

    config = read_json(args.config)
    if args.tune_hp:
        config = tune_hp(config)
    assert 0 <= config["train"]["sup_pn"] <= 1
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

    src_documents: Dict[str, str] = load_documents(args.src_data_dir, docids=None)
    tgt_documents: Dict[str, str] = load_documents(args.tgt_data_dir, docids=None)
    src_train_feat, src_val_feat, src_test_feat = create_datasets_features(load_datasets(args.src_data_dir), src_documents, device)
    tgt_train_feat, tgt_val_feat, tgt_test_feat = create_datasets_features(load_datasets(args.tgt_data_dir), tgt_documents, device)
    src_train_feat, tgt_train_feat = add_wa_to_anns(src_train_feat, tgt_train_feat, src_was, tgt_was, src_documents, tgt_documents)
    src_algn_mask = get_algn_mask(src_train_feat)
    tgt_algn_mask = get_algn_mask(tgt_train_feat)

    # create train dataloader later
    src_train_dataset = EraserDataset(src_train_feat, tokenizer, embedding_model, logger)
    src_val_dataset = EraserDataset(src_val_feat, tokenizer, embedding_model, logger)
    src_test_dataset = EraserDataset(src_test_feat, tokenizer, embedding_model, logger)
    src_val_dataloader = DataLoader(src_val_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    src_test_dataloader = DataLoader(src_test_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)

    tgt_train_dataset = EraserDataset(tgt_train_feat, tokenizer, embedding_model, logger)
    tgt_val_dataset = EraserDataset(tgt_val_feat, tokenizer, embedding_model, logger)
    tgt_test_dataset = EraserDataset(tgt_test_feat, tokenizer, embedding_model, logger)
    tgt_val_dataloader = DataLoader(tgt_val_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    tgt_test_dataloader = DataLoader(tgt_test_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    
    # instantiate models
    src_enc, src_gen = instantiate_models(config, device, os.path.join(args.src_model_dir, "best_enc_weights.pth"), os.path.join(args.src_model_dir, "best_gen_weights.pth"))
    tgt_enc, tgt_gen = instantiate_models(config, device, os.path.join(args.tgt_model_dir, "best_enc_weights.pth"), os.path.join(args.tgt_model_dir, "best_gen_weights.pth"))

    # instantiate optimiser
    src_optimizer = get_optimizer([src_gen, src_enc], config["train"]["lr"])
    src_scheduler = ReduceLROnPlateau(src_optimizer, 'max', patience=2)
    tgt_optimizer = get_optimizer([tgt_gen, tgt_enc], config["train"]["lr"])
    tgt_scheduler = ReduceLROnPlateau(tgt_optimizer, 'max', patience=2)

    epochs = config["train"]["num_epochs"]
    # src_val_metrics = read_json(os.path.join(args.src_model_dir, "results.json"))
    # tgt_val_metrics = read_json(os.path.join(args.tgt_model_dir, "results.json"))
    # best_val_target_metric = src_val_metrics["best_val_f1"] + src_val_metrics["best_val_tok_f1"] + tgt_val_metrics["best_val_f1"] + tgt_val_metrics["best_val_tok_f1"]
    best_val_target_metric = 0
    es_count = 0
    cotrain_patience_count = 0
    growth_rate = args.cotrain_rate
    best_cotrain_rate = 0
    for t in range(epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        # augment train datasets with cotrain masks
        cur_cotrain_rate = best_cotrain_rate + growth_rate
        src_train_dataset, tgt_train_dataset, cotrain_scalar_metrics = cotrain(src_gen, tgt_gen, src_train_dataset, tgt_train_dataset, src_algn_mask, tgt_algn_mask, cur_cotrain_rate)
        src_train_dataloader = DataLoader(src_train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
        tgt_train_dataloader = DataLoader(tgt_train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)

        # vanilla training loops
        src_train_scalar_metrics, _ = train(src_train_dataloader, src_enc, src_gen, src_optimizer, args, device, config)
        src_val_scalar_metrics = test(src_val_dataloader, src_enc, src_gen, device)
        src_overall_scalar_metrics = {**src_train_scalar_metrics, **src_val_scalar_metrics}
        src_val_target_metric = src_overall_scalar_metrics["val_f1"] + src_overall_scalar_metrics["val_tok_f1"]
        src_scheduler.step(src_val_target_metric)

        tgt_train_scalar_metrics, _ = train(tgt_train_dataloader, tgt_enc, tgt_gen, tgt_optimizer, args, device, config)
        tgt_val_scalar_metrics = test(tgt_val_dataloader, tgt_enc, tgt_gen, device)
        tgt_overall_scalar_metrics = {**tgt_train_scalar_metrics, **tgt_val_scalar_metrics}
        tgt_val_target_metric = tgt_overall_scalar_metrics["val_f1"] + tgt_overall_scalar_metrics["val_tok_f1"]
        tgt_scheduler.step(tgt_val_target_metric)

        # logging metrics
        for tag, val in cotrain_scalar_metrics.items():
            writer.add_scalar(tag, val, t)
        for tag, val in src_overall_scalar_metrics.items():
            writer.add_scalar(f"src_{tag}", val, t)
        writer.add_scalar('learning_rate', src_optimizer.param_groups[0]['lr'], t)
        for tag, val in tgt_overall_scalar_metrics.items():
            writer.add_scalar(f"tgt_{tag}", val, t)
        writer.add_scalar('learning_rate', tgt_optimizer.param_groups[0]['lr'], t)
        writer.add_scalar('cotrain_rate', cur_cotrain_rate, t)

        # early stopping and adjusting cotraining step
        val_target_metric = src_val_target_metric + tgt_val_target_metric
        if val_target_metric > best_val_target_metric: 
            best_val_target_metric = val_target_metric
            es_count = 0
            torch.save(src_gen.state_dict(), os.path.join(args.out_dir, "best_src_gen_weights.pth"))
            torch.save(src_enc.state_dict(), os.path.join(args.out_dir, "best_src_enc_weights.pth"))
            torch.save(tgt_gen.state_dict(), os.path.join(args.out_dir, "best_tgt_gen_weights.pth"))
            torch.save(tgt_enc.state_dict(), os.path.join(args.out_dir, "best_tgt_enc_weights.pth"))

            best_cotrain_rate = cur_cotrain_rate
            cotrain_patience_count = 0
        else: 
            es_count += 1
            if es_count >= config["train"]["patience"]: 
                logger.info("Early stopping!")
                break
            
            cotrain_patience_count += 1
            if cotrain_patience_count >= args.cotrain_patience:
                growth_rate /= 2
                cotrain_patience_count = 0
            
    logger.info("Done training!")
    logger.info("Evaluating best model on test set")
    src_gen.load_state_dict(torch.load(os.path.join(args.out_dir, "best_src_gen_weights.pth")))
    src_enc.load_state_dict(torch.load(os.path.join(args.out_dir, "best_src_enc_weights.pth")))
    tgt_gen.load_state_dict(torch.load(os.path.join(args.out_dir, "best_tgt_gen_weights.pth")))
    tgt_enc.load_state_dict(torch.load(os.path.join(args.out_dir, "best_tgt_enc_weights.pth")))
    src_test_scalar_metrics = test(src_test_dataloader, src_enc, src_gen, device, split="src_test")
    tgt_test_scalar_metrics = test(tgt_test_dataloader, tgt_enc, tgt_gen, device, split="tgt_test")
    test_scalar_metrics = {**src_test_scalar_metrics, **tgt_test_scalar_metrics}
    for tag, val in test_scalar_metrics.items(): 
        writer.add_scalar(tag, val)

    test_scalar_metrics["total_time"] = str(datetime.timedelta(seconds=time.time() - start_time))
    write_json(test_scalar_metrics, os.path.join(args.out_dir, "results.json"))

if __name__ == "__main__":
    main()
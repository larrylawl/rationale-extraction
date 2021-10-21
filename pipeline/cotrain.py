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
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
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
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--subset_val", action="store_true")
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
    
def compute_top_k_prob_mask(gen, ds, args, config):
    """ Returns prob mask tensor with only the top r% most confident tokens retained; remaining tokens are zeroed out.
    Size (L, N), where L denotes the longest sequence length and N denotes the training size.  """
    # TODO: assert shuffle false
    dataloader = DataLoader(ds, batch_size=config["train"]["batch_size"], shuffle=False, collate_fn=pad_collate)  
    gen.eval()
    
    # shuffle false for prob_mask[:, 0] to correspond to annotation 0.
    running_scalar_labels = [f"tok_p", f"tok_r", f"tok_f1"]
    running_scalar_metrics = torch.zeros(len(running_scalar_labels))

    with torch.no_grad():
        # forward pass to obtain prob of all tokens
        prob_mask = torch.zeros(config["max_tokens"], len(dataloader.dataset))
        r_mask = torch.zeros(config["max_tokens"], len(dataloader.dataset))
        bs = config["train"]["batch_size"]
        for batch, (t_e_pad, t_e_lens, r_pad, _, _, _) in enumerate(tqdm(dataloader)): 
            mask = gen(t_e_pad, t_e_lens)  # (L, bs)
            assert mask.size() == r_pad.size()

            prob_mask[:, batch*bs:(batch+1)*bs] = F.pad(mask.T, (0, config["max_tokens"] - len(mask)), value=0.5).T # (max_tokens, bs), pad to max_tokens
            r_mask[:, batch*bs:(batch+1)*bs] = F.pad(r_pad.T, (0, config["max_tokens"] - len(r_pad)), value=-1).T # (max_tokens, bs), -1 to not clash with 0 denoting not rationale

            mask_hard = (mask > 0.5).float()
            tok_p, tok_r, tok_f1 = score_hard_rationale_predictions(mask_hard, r_pad, t_e_lens, average="micro")  # micro for valid comparison with top k scores
            running_scalar_metrics += torch.tensor([tok_p, tok_r, tok_f1])

        # label top k% of most confident tokens
        top_k_prob_mask = get_top_k_prob_mask(prob_mask, ds.get_algn_mask(), config["cotrain"]["token_pn"], strategy="token")# (max_tokens, trg_size)

        # perfect labelling instead
        if config["cotrain"]["perfect"]: 
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

def cotrain(src_gen, tgt_gen, src_ds, tgt_ds, args, config, device, label_fns=[same_label, higher_conf]) -> Dataset:
    """ Augments ds with self-labels. """
    src_ds.cotrain_mask = None
    tgt_ds.cotrain_mask = None

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


    src_top_k_prob_mask, src_scalar_metrics, src_r_mask, src_prob_mask = compute_top_k_prob_mask(src_gen, src_ds, args, config)
    logger.info(src_scalar_metrics)
    tgt_top_k_prob_mask, tgt_scalar_metrics, tgt_r_mask, tgt_prob_mask = compute_top_k_prob_mask(tgt_gen, tgt_ds, args, config)
    logger.info(tgt_scalar_metrics)


    # Co-Training
    # Removes labels which are conflicting.
    # NOTE: looping is fine since number of self labels are small. if r == 1, should vectorize
    src_idxs = (src_top_k_prob_mask + 1).nonzero()
    tgt_idxs = (tgt_top_k_prob_mask + 1).nonzero()
    denied_labels = 0
    success_labels = 0
    for tkn_idx, ann_idx in src_idxs:
        src_wa = src_ds.anns[ann_idx].alignment
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
        tgt_wa = tgt_ds.anns[ann_idx].alignment
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
    
    src_eg_idx = torch.unique((src_top_k_prob_mask + 1).nonzero(as_tuple = True)[1])
    src_eg_pn = len(src_eg_idx) / src_top_k_prob_mask.size(1)
    src_top_k_pos_pn = torch.count_nonzero(src_top_k_pred) / len(src_top_k_pred)

    tgt_eg_idx = torch.unique((tgt_top_k_prob_mask + 1).nonzero(as_tuple = True)[1])
    tgt_eg_pn = len(tgt_eg_idx) / tgt_top_k_prob_mask.size(1)
    tgt_top_k_pos_pn = torch.count_nonzero(tgt_top_k_pred) / len(tgt_top_k_pred)

    overall_scalar_metrics = {
        "src_top_k_p": src_p, "src_top_k_r": src_r, "src_top_k_f1": src_f1,
        "tgt_top_k_p": tgt_p, "tgt_top_k_r": tgt_r, "tgt_top_k_f1": tgt_f1,
        "denied_labels": denied_labels / (len(src_idxs) + len(tgt_idxs)), "success_labels_pn": success_labels / (len(src_idxs) + len(tgt_idxs)),
        "src_eg_pn": src_eg_pn, "tgt_eg_pn": tgt_eg_pn, "src_top_k_pos_pn": src_top_k_pos_pn, "tgt_top_k_pos_pn": tgt_top_k_pos_pn
    }
    logger.info(overall_scalar_metrics)

    assert src_top_k_prob_mask.size(1) == len(src_ds), f"Col i of prob mask corresponds to self labels for ith annotation. {len(src_top_k_prob_mask)} != {len(src_train_dataset)}"
    assert tgt_top_k_prob_mask.size(1) == len(tgt_ds), f"Col i of prob mask corresponds to self labels for ith annotation. {len(tgt_top_k_prob_mask)} != {len(tgt_train_dataset)}"
    assert src_top_k_prob_mask.size() == tgt_top_k_prob_mask.size()

    src_ds.cotrain_mask = src_top_k_prob_mask.to(device)
    tgt_ds.cotrain_mask = tgt_top_k_prob_mask.to(device)
    src_subset_ds = Subset(src_ds, src_eg_idx)
    tgt_subset_ds = Subset(tgt_ds, tgt_eg_idx)

    return src_subset_ds, tgt_subset_ds, overall_scalar_metrics

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
    assert 0 <= config["cotrain"]["instance_pn"] <= 1
    assert 0 <= config["cotrain"]["token_pn"] <= 1
    write_json(config, os.path.join(args.out_dir, "config.json"))
    config["encoder"]["num_classes"] = len(dataset_mapping)

    tokenizer = AutoTokenizer.from_pretrained(config["embedding_model_name"])
    embedding_model = AutoModel.from_pretrained(config["embedding_model_name"], output_hidden_states=True)
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    embedding_model.to(device)
    embedding_model.eval()  # only extracting pre-trained embeddings

    # setting up datasets
    if not args.overwrite_cache and os.path.exists(os.path.join(args.src_lab_data_dir, "src_l_feats.pkl")) and os.path.exists(os.path.join(args.src_unlab_data_dir, "src_ul_feats.pkl")):
        logger.info("Loading cached features")
        src_l_feats = torch.load(os.path.join(args.src_lab_data_dir, "src_l_feats.pkl"))
        src_ul_feats = torch.load(os.path.join(args.src_unlab_data_dir, "src_ul_feats.pkl"))
        tgt_l_feats = torch.load(os.path.join(args.tgt_lab_data_dir, "tgt_l_feats.pkl"))
        tgt_ul_feats = torch.load(os.path.join(args.tgt_unlab_data_dir, "tgt_ul_feats.pkl"))
    else:
        logger.info("Caching features")
        src_l_feats = create_datasets_features(load_datasets(args.src_lab_data_dir), load_documents(args.src_lab_data_dir), device)
        tgt_l_feats = create_datasets_features(load_datasets(args.tgt_lab_data_dir), load_documents(args.tgt_lab_data_dir), device)

        src_ul_documents: Dict[str, str] = load_documents(args.src_unlab_data_dir)
        tgt_ul_documents: Dict[str, str] = load_documents(args.tgt_unlab_data_dir)
        src_ul_feats = create_datasets_features(load_datasets(args.src_unlab_data_dir), src_ul_documents, device)
        tgt_ul_feats = create_datasets_features(load_datasets(args.tgt_unlab_data_dir), tgt_ul_documents, device)

        # augmenting unlabelled datasets with word alignment
        was = load_id_jsonl_as_dict(os.path.join(args.tgt_unlab_data_dir, "wa.jsonl"))
        src_was: Dict[str, List[int]] = {}  # maps from src to tgt
        tgt_was: Dict[str, List[int]] = {}  # maps from tgt to src
        for k, v in was.items(): 
            src_was[k] = parse_alignment(v["alignment"])
            tgt_was[k] = parse_alignment(v["alignment"], reverse=True)
        src_train_feat, tgt_train_feat = add_wa_to_anns(src_ul_feats[0], tgt_ul_feats[0], src_was, tgt_was, src_ul_documents, tgt_ul_documents)
        src_ul_feats[0] = src_train_feat
        tgt_ul_feats[0] = tgt_train_feat
        torch.save(src_l_feats, os.path.join(args.src_lab_data_dir, "src_l_feats.pkl"))
        torch.save(src_ul_feats, os.path.join(args.src_unlab_data_dir, "src_ul_feats.pkl"))
        torch.save(tgt_l_feats, os.path.join(args.tgt_lab_data_dir, "tgt_l_feats.pkl"))
        torch.save(tgt_ul_feats, os.path.join(args.tgt_unlab_data_dir, "tgt_ul_feats.pkl"))

    # create train dataloader later
    # NOTE: for train, need to indicate which are sup since we'll be labelling other examples too
    src_l_ds = [EraserDataset(feat, tokenizer, embedding_model) for feat in src_l_feats]
    src_ul_ds = [EraserDataset(feat, tokenizer, embedding_model) for feat in src_ul_feats]
    tgt_l_ds = [EraserDataset(feat, tokenizer, embedding_model) for feat in tgt_l_feats]
    tgt_ul_ds = [EraserDataset(feat, tokenizer, embedding_model) for feat in tgt_ul_feats]
    src_l_ds[0].is_labelled = True
    tgt_l_ds[0].is_labelled = True
    
    src_val_ds = ConcatDataset([src_l_ds[1], src_ul_ds[1]])
    src_test_ds = ConcatDataset([src_l_ds[2], src_ul_ds[2]])
    tgt_val_ds = ConcatDataset([tgt_l_ds[1], tgt_ul_ds[1]])
    tgt_test_ds = ConcatDataset([tgt_l_ds[1], tgt_ul_ds[1]])
    
    if args.subset_val:
        val_shuffled_ids = torch.randperm(len(src_val_ds))
        val_size = math.floor(config["cotrain"]["instance_pn"] * len(src_val_ds))
        # test_shuffled_ids = torch.randperm(len(src_test_ds))
        # test_size = math.floor(config["cotrain"]["instance_pn"] * len(src_test_ds))
        src_val_ds = Subset(src_val_ds, val_shuffled_ids[:val_size])
        # src_test_ds = Subset(src_test_ds, test_shuffled_ids[:test_size])
        tgt_val_ds = Subset(tgt_val_ds, val_shuffled_ids[:val_size])
        # tgt_test_ds = Subset(tgt_test_ds, test_shuffled_ids[:test_size])

    dl_params = {"batch_size": config["train"]["batch_size"], "shuffle": True, "collate_fn": pad_collate}
    src_val_dl = DataLoader(src_val_ds, **dl_params)
    src_test_dl = DataLoader(src_test_ds, **dl_params)
    tgt_val_dl = DataLoader(tgt_val_ds, **dl_params)
    tgt_test_dl = DataLoader(tgt_test_ds, **dl_params)

    logger.info(f"after dataloaders: {time.time() - start_time}")

    best_src_gen_fp = os.path.join(args.src_model_dir, "best_gen_weights.pth")  # will be updated
    best_tgt_gen_fp = os.path.join(args.tgt_model_dir, "best_gen_weights.pth")
    co_best_src_val_metrics = get_best_val_metrics(read_json(os.path.join(args.src_model_dir, "best_val_results.json")))
    co_best_tgt_val_metrics = get_best_val_metrics(read_json(os.path.join(args.tgt_model_dir, "best_val_results.json")))
    co_es_count = 0
    co_writer = SummaryWriter(args.out_dir)
    shuffled_ids = torch.randperm(len(src_ul_ds[0]))

    for co_t in range(config["cotrain"]["epochs"]):
        logger.info(f"Cotrain Epochs {co_t}\n-------------------------------")
        # updates train datasets with new cotrain masks
        src_gen = instantiate_generator(config, device, best_src_gen_fp)
        tgt_gen = instantiate_generator(config, device, best_tgt_gen_fp)
        # cur_lr = config["train"]["lr"] / (2 ** co_t)
        cur_lr = config["train"]["lr"]
        src_optimizer = get_optimizer([src_gen], cur_lr)
        tgt_optimizer = get_optimizer([tgt_gen], cur_lr)

        # rate = min(2 * config["cotrain"]["rate"] * (1 - 0.5 ** (co_t + 1)), 1)  # sum of GP with a = rate, r = 0.5
        pn = config["cotrain"]["instance_pn"] * (co_t + 1)
        size = math.floor(pn * len(src_ul_ds[0]))
        
        logger.info(f"pn: {pn}")
        logger.info(f"size: {size}")
        sub_idxs = set(shuffled_ids[:size].tolist())
        src_ul_subfeats = [ann for i, ann in enumerate(src_ul_feats[0]) if i in sub_idxs]
        logger.info(f"after iterating")
        src_ul_subds = EraserDataset(src_ul_subfeats, tokenizer, embedding_model, is_labelled=False)

        tgt_ul_subfeats = [ann for i, ann in enumerate(tgt_ul_feats[0]) if i in shuffled_ids[:size]]
        tgt_ul_subds = EraserDataset(tgt_ul_subfeats, tokenizer, embedding_model, is_labelled=False)
        src_ul_train_ds, tgt_ul_train_ds, cotrain_scalar_metrics = cotrain(src_gen, tgt_gen, src_ul_subds, tgt_ul_subds, args, config, device)
        exit(1)
        for tag, val in cotrain_scalar_metrics.items():
            co_writer.add_scalar(tag, val, co_t)
        co_writer.add_scalar("lr", cur_lr, co_t)

        # fine-tuning C_k 
        src_out_dir = os.path.join(args.out_dir, f"src_{co_t}")
        src_writer = SummaryWriter(src_out_dir)
        # src_train_dl = DataLoader(ConcatDataset([src_l_ds[0], src_ul_train_ds]), **dl_params)
        src_train_l_dl = DataLoader(src_l_ds[0], **dl_params)
        src_train_ul_dl = DataLoader(src_ul_train_ds, **dl_params)
        src_gen, _, src_best_val_scalar_metrics = train_loop(src_train_l_dl, src_val_dl, src_gen, None, src_optimizer, 
                    src_out_dir, src_writer, device, config, logger, src_train_ul_dl)
        
        tgt_out_dir = os.path.join(args.out_dir, f"tgt_{co_t}")
        tgt_writer = SummaryWriter(tgt_out_dir)
        # tgt_train_dl = DataLoader(ConcatDataset([tgt_l_ds[0], tgt_ul_train_ds]), **dl_params)
        tgt_train_l_dl = DataLoader(tgt_l_ds[0], **dl_params)
        tgt_train_ul_dl = DataLoader(tgt_ul_train_ds, **dl_params)
        tgt_gen, _, tgt_best_val_scalar_metrics = train_loop(tgt_train_l_dl, tgt_val_dl, tgt_gen, None, tgt_optimizer, 
                    tgt_out_dir, tgt_writer, device, config, logger, tgt_train_ul_dl)

        src_best_val_scalar_metrics = get_best_val_metrics(src_best_val_scalar_metrics)
        tgt_best_val_scalar_metrics = get_best_val_metrics(tgt_best_val_scalar_metrics)

        # co-train early stopping
        logger.info(f"src_best_val_scalar_metrics: {src_best_val_scalar_metrics}")
        logger.info(f"co_best_src_val_metrics: {co_best_src_val_metrics}")
        logger.info(f"tgt_best_val_scalar_metrics: {tgt_best_val_scalar_metrics}")
        logger.info(f"co_best_tgt_val_metrics: {co_best_tgt_val_metrics}")
        if src_best_val_scalar_metrics >= co_best_src_val_metrics or tgt_best_val_scalar_metrics >= co_best_tgt_val_metrics:
            co_es_count = 0
        else:
            co_es_count += 1    
            if co_es_count >= config["cotrain"]["patience"]:
                logger.info("Early stopping co-training!")
                break
        
        # saving best models
        if src_best_val_scalar_metrics > co_best_src_val_metrics: 
            co_best_src_val_metrics = src_best_val_scalar_metrics
            best_src_gen_fp = os.path.join(src_out_dir, "best_gen_weights.pth")
        if tgt_best_val_scalar_metrics > co_best_tgt_val_metrics:
            co_best_tgt_val_metrics = tgt_best_val_scalar_metrics
            best_tgt_gen_fp = os.path.join(tgt_out_dir, "best_gen_weights.pth")
        
        src_writer.close()
        tgt_writer.close()

    src_gen = instantiate_generator(config, device, best_src_gen_fp)
    tgt_gen = instantiate_generator(config, device, best_tgt_gen_fp)
    co_src_test_scalar_metrics = test(src_test_dl, None, src_gen, config, split="src_test")
    co_tgt_test_scalar_metrics = test(tgt_test_dl, None, tgt_gen, config, split="tgt_test")
    co_test_scalar_metrics = {**co_src_test_scalar_metrics, **co_tgt_test_scalar_metrics}
    for tag, val in co_test_scalar_metrics.items(): 
        co_writer.add_scalar(tag, val)
    co_test_scalar_metrics["total_time"] = str(datetime.timedelta(seconds=time.time() - start_time))
    write_json(co_test_scalar_metrics, os.path.join(args.out_dir, "results.json"))

if __name__ == "__main__":
    main()
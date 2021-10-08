import sys; sys.path.insert(0, "..")
import os
import time
import datetime
import logging
import shutil
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
import argparse
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from models.encoder import Encoder
from models.generator import Generator
import random
from functools import partial

from pipeline.main import EraserDataset, train, test, pad_collate, tune_hp
from utils import get_top_k_prob_mask, load_datasets, create_instance, load_documents, load_id_jsonl_as_dict, load_instances, get_num_classes, get_optimizer, dataset_mapping, get_base_dataset_name, parse_alignment, read_json, top_k_idxs_multid, write_json, plot_grad_flow, tracked_named_parameters, score_hard_rationale_predictions, add_offsets

logging.basicConfig(level=logging.INFO, format='%(relativeCreated)6d %(threadName)s %(message)s')
# let's make this more or less deterministic (not resistent to restarts)

logger = logging.getLogger(__name__)
args = None
device = None
writer = None
config = None
max_tokens = 100  # assume all sequences ≤ 100 tokens

def parse_args():
    parser = argparse.ArgumentParser("Cotraining.")
    parser.add_argument("--src_data_dir", required=True, help="Input directory to data.")
    parser.add_argument("--tgt_data_dir", required=True, help="Input directory to data.")
    parser.add_argument("--src_model_dir", required=True, help="Model weights file path.")
    parser.add_argument("--tgt_model_dir", required=True, help="Model weights file path.")
    parser.add_argument("--config", required=True, help="Model config file.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tune_hp", action="store_true")
    parser.add_argument("--seed", required=True, type=int, default=100)

    return parser.parse_args()

class CotrainDataset(EraserDataset):
    """ Augments EraserDataset with cotrain_mask: (N, T) wherein row i corresponds to self labels for annotation i.
    -1 denotes not labelled, 0 denotes false, 1 denotes true. """

    def __init__(self, anns, docs, tokenizer, embedding_model, logger, cotrain_mask):
        super().__init__(anns, docs, tokenizer, embedding_model, logger)
        self.cotrain_mask = cotrain_mask
    
    def __getitem__(self, idx: int):
        t_e_pad, t_e_lens, r_pad, l, ann_id = super().__getitem__(idx)
        return t_e_pad, t_e_lens, r_pad, l, ann_id, self.cotrain_mask[idx]
        

def custom_BCE():
    # NOTE: current top_k_prob_mask has value -1 if unlabelled.
    pass

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

        # print(src_offset)
        # print(tgt_offset)
        # # print(src_doc_h_wa)
        # # print(src_doc_p_wa)
        # print(tgt_doc_h_wa)
        # print(tgt_doc_p_wa)
        # # print(src_ann.alignment)
        # print(tgt_ann.alignment)
    return src_train_anns, tgt_train_anns

def get_algn_mask(anns, documents):
    algn_masks = []
    for ann in anns:
        h = f"{ann.annotation_id}_hypothesis"
        p = f"{ann.annotation_id}_premise"
        total_len = len(documents[h].split()) + len(documents[p].split())
        a_m = torch.zeros(total_len)
        for k in ann.alignment: a_m[k] = 1
        algn_masks.append(a_m)
    algn_masks = pad_sequence(algn_masks)
    return algn_masks
    
def get_top_k_prob_mask():
    pass

def cotrain(gen, train_dataset, algn_mask, k) -> DataLoader:
    """ Augments Eraserdataset with self-labels. """
    gen.eval()
    # shuffle false for idx 0 to correspond to annotation 0.
    cotrain_dataloader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=False, collate_fn=pad_collate)  # pad embedding with 0.5 as we don't want padding to be selected

    with torch.no_grad():
        # forward pass to obtain prob of all tokens
        # use padding strat here instead
        # prob_mask = torch.empty(algn_mask.size()).to(device).float()
        prob_mask = []
        for batch, (t_e_pad, t_e_lens, r_pad, l, _) in enumerate(tqdm(cotrain_dataloader)): 
            mask = gen(t_e_pad, t_e_lens)  # (L, bs)
            prob_mask.append(mask.cpu())  # cpu since not many operations to mask

            # zero out prob_mask here

            # mask_full = torch.full((max_tokens, config["train"]["batch_size"]), 0.5).to(device)  # (max_tokens, bs)
            # mask_full[:mask.size(0), :] = mask  # (max_tokens, bs)

            # ann_idx = batch * config["train"]["batch_size"]
            # prob_mask[:, ann_idx:config["train"]["batch_size"]] = mask
        
        prob_mask = pad_sequence(prob_mask)
        assert prob_mask.size(1) == algn_mask.size(1), "Should have equal number of examples"
        algn_mask_full = torch.zeros(prob_mask.size())
        algn_mask_full[:algn_mask.size(0), :] = algn_mask

        # label top 1% of most confident tokens
        prob_mask[algn_mask_full == 0] = 0.5   # ensure that tokens with no alignment (including padding) are not selected
        top_k_prob_mask = get_top_k_prob_mask(prob_mask, k).nonzero()  # (max_tokens, trg_size)
        
        # co-training
        ## top_k_prob_mask is a set of indexes (i, j): use j to access jth annotation, and i to access ith alignment. 
        ### => alignment needs to be List[Dict[int, List[int]]] => augment annotation at the start?
        ## update both languages' top_k_prob_mask by looping.
        ### remove conflicting tokens
        ## create new prob_mask of all zeros. Use top_k_prob_mask cross product indexing 


    assert len(top_k_prob_mask) == len(train_dataset), "Row i of prob mask corresponds to self labels for ith annotation."
    train_dataset = CotrainDataset(train_dataset.anns, train_dataset.docs, train_dataset.tokenizer, train_dataset.embedding_model, train_dataset.logger, top_k_prob_mask)
    train_dataloader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    return train_dataloader

def main():
    start_time = time.time()
    global args, device, writer, config, prob_mask, algn_mask
    args = parse_args()
    logger.info(args)
    
    if os.path.exists(args.out_dir): shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)
    base_dataset_name = get_base_dataset_name(os.path.basename(args.src_data_dir))
    writer = SummaryWriter(args.out_dir)
    write_json(vars(args), os.path.join(args.out_dir, "exp_args.json"))

    config = read_json(args.config)
    if args.tune_hp:
        config = tune_hp(config)
    write_json(config, os.path.join(args.out_dir, "config.json"))
    config["encoder"]["num_classes"] = get_num_classes(base_dataset_name)

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
    src_train_anns, src_val_anns, src_test_anns = load_datasets(args.src_data_dir)
    tgt_train_anns, tgt_val_anns, tgt_test_anns = load_datasets(args.tgt_data_dir)
    src_train_anns, tgt_train_anns = add_wa_to_anns(src_train_anns, tgt_train_anns, src_was, tgt_was, src_documents, tgt_documents)
    src_algn_mask = get_algn_mask(src_train_anns, src_documents)
    tgt_algn_mask = get_algn_mask(tgt_train_anns, tgt_documents)

    src_val_dataset = EraserDataset(src_val_anns, src_documents, tokenizer, embedding_model, logger)
    src_test_dataset = EraserDataset(src_test_anns, src_documents, tokenizer, embedding_model, logger)
    src_val_dataloader = DataLoader(src_val_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    src_test_dataloader = DataLoader(src_test_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)

    tgt_val_dataset = EraserDataset(tgt_val_anns, tgt_documents, tokenizer, embedding_model, logger)
    tgt_test_dataset = EraserDataset(tgt_test_anns, tgt_documents, tokenizer, embedding_model, logger)
    tgt_val_dataloader = DataLoader(tgt_val_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    tgt_test_dataloader = DataLoader(tgt_test_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    

    # instantiate models
    src_enc = Encoder(config["encoder"]).to(device)
    src_enc.load_state_dict(torch.load(os.path.join(args.src_model_dir, "best_enc_weights.pth")))
    src_gen = Generator(config["generator"]).to(device)
    src_gen.load_state_dict(torch.load(os.path.join(args.src_model_dir, "best_gen_weights.pth")))

    tgt_enc = Encoder(config["encoder"]).to(device)
    tgt_enc.load_state_dict(torch.load(os.path.join(args.tgt_model_dir, "best_enc_weights.pth")))
    tgt_gen = Generator(config["generator"]).to(device)
    tgt_gen.load_state_dict(torch.load(os.path.join(args.tgt_model_dir, "best_gen_weights.pth")))

    # instantiate optimiser
    src_optimizer = get_optimizer([src_gen, src_enc], config["train"]["lr"])
    src_scheduler = ReduceLROnPlateau(src_optimizer, 'max', patience=2)
    tgt_optimizer = get_optimizer([tgt_gen, tgt_enc], config["train"]["lr"])
    tgt_scheduler = ReduceLROnPlateau(tgt_optimizer, 'max', patience=2)

    epochs = config["train"]["num_epochs"]
    best_val_target_metric = 0
    es_count = 0
    for t in range(epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        train_dataloader = cotrain()
        train_scalar_metrics, _ = train(train_dataloader, enc, gen, optimizer)
        val_scalar_metrics = test(val_dataloader, enc, gen)
        overall_scalar_metrics = {**train_scalar_metrics, **val_scalar_metrics}
        val_target_metric = overall_scalar_metrics["val_f1"] + overall_scalar_metrics["val_tok_f1"]
        scheduler.step(val_target_metric)

        # logging metrics
        for tag, val in overall_scalar_metrics.items():
            writer.add_scalar(tag, val, t)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], t)

        # early stopping
        if val_target_metric > best_val_target_metric:
            best_val_target_metric = val_target_metric
            es_count = 0
            torch.save(gen.state_dict(), os.path.join(args.out_dir, "best_gen_weights.pth"))
            torch.save(enc.state_dict(), os.path.join(args.out_dir, "best_enc_weights.pth"))
        else: 
            es_count += 1
            if es_count >= config["train"]["patience"]: 
                logger.info("Early stopping!")
                break
    logger.info("Done training!")
    logger.info("Evaluating best model on test set")
    gen.load_state_dict(torch.load(os.path.join(args.out_dir, "best_gen_weights.pth")))
    enc.load_state_dict(torch.load(os.path.join(args.out_dir, "best_enc_weights.pth")))
    test_scalar_metrics = test(test_dataloader, enc, gen, split="test")
    test_scalar_metrics["total_time"] = str(datetime.timedelta(seconds=time.time() - start_time))
    write_json(test_scalar_metrics, os.path.join(args.out_dir, "results.json"))

if __name__ == "__main__":
    main()
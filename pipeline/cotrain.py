import sys; sys.path.insert(0, "..")
import os
import time
import datetime
import logging
import shutil
import torch
from torch import nn
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
from utils import get_top_k_prob_mask, load_datasets, create_instance, load_documents, load_instances, get_num_classes, get_optimizer, dataset_mapping, get_base_dataset_name, read_json, top_k_idxs_multid, write_json, plot_grad_flow, tracked_named_parameters, score_hard_rationale_predictions

logging.basicConfig(level=logging.INFO, format='%(relativeCreated)6d %(threadName)s %(message)s')
# let's make this more or less deterministic (not resistent to restarts)

logger = logging.getLogger(__name__)
args = None
device = None
base_dataset_name = None
writer = None
config = None

def parse_args():
    parser = argparse.ArgumentParser("Cotraining.")
    parser.add_argument("--data_dir", required=True, help="Input directory to data.")
    parser.add_argument("--model_dir", required=True, help="Model weights file path.")
    parser.add_argument("--config", required=True, help="Model config file.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tune_hp", action="store_true")
    parser.add_argument("--seed", required=True, type=int, default=100)

    return parser.parse_args()

class CotrainDataset(EraserDataset):
    def __init__(self, anns, docs, tokenizer, embedding_model, logger, cotrain_mask):
        super().__init__(anns, docs, tokenizer, embedding_model, logger)
        self.cotrain_mask = cotrain_mask
    
    def __getitem__(self, idx: int):
        t_e_pad, t_e_lens, r_pad, l, ann_id = super().__getitem__(idx)
        return t_e_pad, t_e_lens, r_pad, l, ann_id, self.cotrain_mask[idx]
        

def custom_BCE():
    # NOTE: current top_k_prob_mask has value -1 if unlabelled.
    pass

def cotrain(gen, train_dataset) -> EraserDataset:
    """ Augments Eraserdataset with self-labels. """
    gen.eval()
    # shuffle false for idx 0 to correspond to annotation 0.
    # batch size 1 to ensure padded tokens aren't selected (since there will be no padding)
    cotrain_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=partial(pad_collate, 0.5))  # pad embedding with 0.5 as we don't want padding to be selected

    with torch.no_grad():
        # forward pass to obtain prob of all tokens
        prob_mask = []
        for batch, (t_e_pad, t_e_lens, r_pad, l, _) in enumerate(tqdm(cotrain_dataloader)): 
            mask = gen(t_e_pad, t_e_lens)
            prob_mask.append(mask)

        # label top 1% of most confident tokens
        ##  pad sequence with padding value of 0.5 => confidence == 0 => pading won't be selected
        prob_mask = pad_sequence(prob_mask, padding_value = 0.5)  # (T, B, *)
        prob_mask = torch.unbind(prob_mask, dim=1)  # (T, *), unstack padded tensors
        assert prob_mask.size() == algn_mask.size()
        prob_mask[algn_mask == -1] = 0.5  # ensure that tokens with no alignment are not selected
        top_k_prob_mask = get_top_k_prob_mask(prob_mask, k)

        # TODO: use indexes in alignment mask to map labels from src mask to tgt mask


    assert len(top_k_prob_mask) == len(train_dataset), "Row i of prob mask corresponds to self labels for ith annotation."
    train_dataset = CotrainDataset(train_dataset.anns, train_dataset.docs, train_dataset.tokenizer, train_dataset.embedding_model, train_dataset.logger, top_k_prob_mask)
    train_dataloader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    return train_dataloader

def main():
    start_time = time.time()
    global args, device, base_dataset_name, writer, config
    args = parse_args()
    logger.info(args)
    
    if os.path.exists(args.out_dir): shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)
    base_dataset_name = get_base_dataset_name(os.path.basename(args.data_dir))
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
    documents: Dict[str, str] = load_documents(args.data_dir, docids=None)
    train_anns, val_anns, test_anns = load_datasets(args.data_dir)

    # create datset
    val_dataset = EraserDataset(val_anns, documents, tokenizer, embedding_model, logger)
    test_dataset = EraserDataset(test_anns, documents, tokenizer, embedding_model, logger)

    val_dataloader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)

    # instantiate models
    enc = Encoder(config["encoder"]).to(device)
    enc.load_state_dict(torch.load(os.path.join(args.model_dir, "best_enc_weights.pth")))

    gen = Generator(config["generator"]).to(device)
    gen.load_state_dict(torch.load(os.path.join(args.model_dir, "best_gen_weights.pth")))

    # instantiate optimiser
    optimizer = get_optimizer([gen, enc], config["train"]["lr"])
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2)

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
    test_scalar_metrics = test(test_dataloader, enc, gen)
    test_scalar_metrics["total_time"] = str(datetime.timedelta(seconds=time.time() - start_time))
    write_json(test_scalar_metrics, os.path.join(args.out_dir, "results.json"))

if __name__ == "__main__":
    main()
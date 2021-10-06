import sys; sys.path.insert(0, "..")
import os
from itertools import chain
import json
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

from utils import load_datasets, create_instance, load_documents, load_instances, get_num_classes, get_optimizer, dataset_mapping, get_base_dataset_name, read_json, write_json, plot_grad_flow, tracked_named_parameters, score_hard_rationale_predictions

logging.basicConfig(level=logging.INFO, format='%(relativeCreated)6d %(threadName)s %(message)s')
# let's make this more or less deterministic (not resistent to restarts)

logger = logging.getLogger(__name__)
args = None
device = None
base_dataset_name = None
writer = None
config = None

def parse_args():
    parser = argparse.ArgumentParser("Translates the files in docs.")
    parser.add_argument("--data_dir", required=True, help="Input directory to data.")
    parser.add_argument("--config", required=True, help="Model config file.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tune_hp", action="store_true")
    parser.add_argument("--sup", action="store_true")
    parser.add_argument("--seed", required=True, type=int, default=100)

    return parser.parse_args()

class EraserDataset(Dataset):
    """ ERASER dataset. """

    def __init__(self, anns, docs, tokenizer, embedding_model, logger):
        self.anns = anns
        self.docs = docs
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.logger = logger

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx: int):
        return create_instance(self.anns[idx], self.docs, self.tokenizer, self.embedding_model, self.logger)

def pad_collate(batch):
    # TODO: pad for rationale
    (t_e, r, l, ann_id) = zip(*batch)
    t_e_lens = [t.size()[0] for t in t_e]

    t_e_pad = pad_sequence(t_e)  # (L, N, H_in)
    r_pad = pad_sequence(r).to(device)
    # t_e_packed = pack_padded_sequence(t_e_pad, t_e_lens, enforce_sorted=False)
    l = torch.tensor([dataset_mapping[base_dataset_name][x] for x in l], dtype=torch.long).to(device)

    return t_e_pad, t_e_lens, r_pad, l, ann_id

def tune_hp(config):
    config["generator"]["selection_lambda"] = round(10 ** random.uniform(-4, -2), 5)
    config["generator"]["continuity_lambda"] = round(10 ** random.uniform(-4, -2), 5)
    config["train"]["lr"] = round(10 ** random.uniform(-4, -2), 5)
    
    return config

def train(dataloader, enc, gen, optimizer):
    running_scalar_labels = ["trg_loss", "trg_obj_loss", "trg_mask_loss", "trg_total_f1", "trg_tok_precision", "trg_tok_recall", "trg_tok_f1"]
    running_scalar_metrics = torch.zeros(len(running_scalar_labels))
    # total_params = len(tracked_named_parameters(chain(gen.named_parameters(), enc.named_parameters())))
    # mean_grads = torch.zeros(total_params).to(device)
    # var_grads = torch.zeros(total_params).to(device)

    gen.train()
    enc.train()

    for batch, (t_e_pad, t_e_lens, r_pad, l, _) in enumerate(tqdm(dataloader)):        
        # forward pass
        mask = gen(t_e_pad, t_e_lens)
        logit = enc(t_e_pad, t_e_lens, mask=mask)  # NOTE: to test gen and enc independently, change mask to none
        probs = nn.Softmax(dim=1)(logit.detach().cpu())
        y_pred = torch.argmax(probs, dim=1)

        # compute losses
        selection_cost, continuity_cost = gen.loss(mask)
        # print(f"mask: {mask.type()}")
        # print(f"r_pad: {r_pad.type()}")
        if args.sup: mask_sup_loss = nn.BCELoss()(mask, r_pad)
        else: mask_sup_loss = torch.tensor(0)
        obj_loss = nn.CrossEntropyLoss()(logit, l)
        loss = obj_loss + selection_cost + continuity_cost + mask_sup_loss
        f1 = f1_score(l.detach().cpu(), y_pred, average="macro")
        tok_p, tok_r, tok_f1 = score_hard_rationale_predictions(mask.detach().cpu(), r_pad.detach().cpu(), t_e_lens)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # tracking metrics
        mask_loss = selection_cost.detach() + continuity_cost.detach() + mask_sup_loss.detach()
        running_scalar_metrics += torch.tensor([loss.detach(), obj_loss.detach(), mask_loss, f1, tok_p, tok_r, tok_f1])

        # tracking gradients
        # t_n_p = tracked_named_parameters(chain(gen.named_parameters(), enc.named_parameters()))
        # for i, (_, p) in enumerate(t_n_p):
        #     var, mean = torch.var_mean(p.grad.detach().abs())  # abs to ensure gradient's don't cancel out to wrongly indicate vanishing grad
        #     mean_grads[i] += mean
        #     var_grads[i] += var
            

    total_scalar_metrics = running_scalar_metrics / (batch + 1)
    scalar_metrics = {}
    for i in range(len(running_scalar_labels)): scalar_metrics[running_scalar_labels[i]] = total_scalar_metrics[i]

    # tensor_metrics = {
    #     "mean_grads": (mean_grads / (batch + 1)).cpu(),
    #     "var_grads": (var_grads / (batch + 1)).cpu()
    # }
    return scalar_metrics, _

def test(dataloader, enc, gen):
    running_scalar_labels = ["val_f1", "val_tok_precision", "val_tok_recall", "val_tok_f1"]
    running_scalar_metrics = torch.zeros(len(running_scalar_labels))
    
    gen.eval()
    enc.eval()
    with torch.no_grad():
        for batch, (t_e_pad, t_e_lens, r_pad, l, ann_id) in enumerate(tqdm(dataloader)):        
            mask = gen(t_e_pad, t_e_lens)
            logit = enc(t_e_pad, t_e_lens, mask=mask)  # NOTE: to test gen and enc independently, change mask to none
            probs = nn.Softmax(dim=1)(logit.detach().cpu())
            y_pred = torch.argmax(probs, dim=1)
            f1 = f1_score(l.detach().cpu(), y_pred, average="macro")
            tok_p, tok_r, tok_f1 = score_hard_rationale_predictions(mask.detach().cpu(), r_pad.detach().cpu(), t_e_lens)

            running_scalar_metrics += torch.tensor([f1, tok_p, tok_r, tok_f1])

        total_scalar_metrics = running_scalar_metrics / (batch + 1)
        scalar_metrics = {}
        for i in range(len(running_scalar_labels)): scalar_metrics[running_scalar_labels[i]] = total_scalar_metrics[i].item()
        return scalar_metrics

def main():
    global args, device, base_dataset_name, writer, config
    args = parse_args()
    logger.info(args)
    
    if os.path.exists(args.out_dir): 
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)
    base_dataset_name = get_base_dataset_name(os.path.basename(args.data_dir))
    writer = SummaryWriter(args.out_dir)

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

    # # debug: ensuring create_instance works for entire dataset
    # load_instances(args.data_dir, tokenizer, embedding_model, logger, debug=True)

    # setting up data
    documents: Dict[str, str] = load_documents(args.data_dir, docids=None)
    train_anns, val_anns, test_anns = load_datasets(args.data_dir)

    # create datset
    train_dataset = EraserDataset(train_anns, documents, tokenizer, embedding_model, logger)
    val_dataset = EraserDataset(val_anns, documents, tokenizer, embedding_model, logger)
    test_dataset = EraserDataset(test_anns, documents, tokenizer, embedding_model, logger)

    train_dataloader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)

    # instantiate models
    enc = Encoder(config["encoder"]).to(device)
    gen = Generator(config["generator"]).to(device)

    # Note: no longer used
    # for gradient flow tracking later
    # layer_names = [n for n, _ in tracked_named_parameters(chain(gen.named_parameters(), enc.named_parameters()))]  

    # instantiate optimiser
    optimizer = get_optimizer([gen, enc], config["train"]["lr"])
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2)

    epochs = config["train"]["num_epochs"]
    best_val_target_metric = 0
    es_count = 0
    for t in range(epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        train_scalar_metrics, _ = train(train_dataloader, enc, gen, optimizer)
        val_scalar_metrics = test(val_dataloader, enc, gen)
        overall_scalar_metrics = {**train_scalar_metrics, **val_scalar_metrics}
        val_target_metric = overall_scalar_metrics["val_f1"] + overall_scalar_metrics["val_tok_f1"]
        scheduler.step(val_target_metric)

        # logging metrics
        for tag, val in overall_scalar_metrics.items():
            writer.add_scalar(tag, val, t)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], t)
        # plot_grad_flow(train_tensor_metrics["mean_grads"], train_tensor_metrics["var_grads"], layer_names, os.path.join(args.out_dir, f"model_grad_flow_{t}.png"))
        # del train_tensor_metrics

        # early stopping
        if val_target_metric > best_val_target_metric:
            best_val_target_metric = val_target_metric
            es_count = 0
            torch.save(enc.state_dict(), os.path.join(args.out_dir, "best_encoder_weights.pth"))
        else: 
            es_count += 1
            if es_count >= config["train"]["patience"]: 
                logger.info("Early stopping!")
                break
    logger.info("Done training!")
    logger.info("Evaluating best model on test set")
    enc.load_state_dict(torch.load(os.path.join(args.out_dir, "best_encoder_weights.pth")))
    test_scalar_metrics = test(test_dataloader, enc, gen)

    write_json(test_scalar_metrics, os.path.join(args.out_dir, "results.json"))

if __name__ == "__main__":
    main()
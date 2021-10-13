import sys; sys.path.insert(0, "..")
import os
import time
import datetime
import logging
import shutil
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import argparse
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModel
import random

from pipeline.cotrain_utils import *
from utils import instantiate_models, load_datasets, load_documents, load_instances, get_optimizer, read_json, slice_by_index, write_json, plot_grad_flow, tracked_named_parameters, score_hard_rationale_predictions

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
    parser.add_argument("--model_dir", default=None, help="Model weights file path. If none, trains from scratch.")
    parser.add_argument("--config", required=True, help="Model config file.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tune_hp", action="store_true")
    parser.add_argument("--gen_only", action="store_true", help="Supervised training with gen only")
    parser.add_argument("--seed", required=True, type=int, default=100)

    return parser.parse_args()

def main():
    start_time = time.time()
    global args, device, base_dataset_name, writer, config
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
    if args.gen_only: 
        assert config["train"]["sup_pn"] > 0
        config["generator"]["selection_lambda"] = 0
        config["generator"]["continuity_lambda"] = 0
    write_json(config, os.path.join(args.out_dir, "config.json"))
    config["encoder"]["num_classes"] = len(dataset_mapping)

    tokenizer = AutoTokenizer.from_pretrained(config["embedding_model_name"])
    embedding_model = AutoModel.from_pretrained(config["embedding_model_name"], output_hidden_states=True)
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    embedding_model.to(device)
    embedding_model.eval()  # only extracting pre-trained embeddings

    # # debug: ensuring create_instance works for entire dataset
    # load_instances(args.data_dir, tokenizer, embedding_model, logger, debug=True)

    # setting up data
    documents: Dict[str, str] = load_documents(args.data_dir, docids=None)
    # train_anns, val_anns, test_anns = load_datasets(args.data_dir)
    # NOTE: can toggle --gen_only here if i need full retraining with partial supervision.
    train_anns, val_anns, test_anns = load_datasets(args.data_dir)
    label_size = math.ceil(config["train"]["sup_pn"] * len(train_anns))
    label_idxs = set(random.sample(range(len(train_anns)), label_size))
    train_anns = slice_by_index(train_anns, label_idxs)
    train_feat, val_feat, test_feat = create_datasets_features([train_anns, val_anns, test_anns], documents, device)


    # TODO: change to tensor dataset to avoid expensive zipping operation of pad_collate
    # create datset
    train_dataset = EraserDataset(train_feat, tokenizer, embedding_model)
    val_dataset = EraserDataset(val_feat, tokenizer, embedding_model)
    test_dataset = EraserDataset(test_feat, tokenizer, embedding_model)

    train_dataloader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)

    # instantiate models
    enc, gen = instantiate_models(config, device, args.model_dir)
    if args.gen_only: enc = None

    # instantiate optimiser
    optimizer = get_optimizer([gen, enc], config["train"]["lr"])
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2)

    epochs = config["train"]["num_epochs"]
    best_val_target_metric = 0
    best_val_scalar_metrics = {}
    es_count = 0
    for t in range(epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        train_scalar_metrics, _ = train(train_dataloader, enc, gen, optimizer, args, device, config)
        val_scalar_metrics = test(val_dataloader, enc, gen, device)
        overall_scalar_metrics = {**train_scalar_metrics, **val_scalar_metrics}
        val_target_metric = overall_scalar_metrics["val_f1"] + overall_scalar_metrics["val_tok_f1"]
        scheduler.step(val_target_metric)

        # logging metrics
        for tag, val in overall_scalar_metrics.items():
            writer.add_scalar(tag, val, t)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], t)
        # del train_tensor_metrics

        # early stopping
        if val_target_metric > best_val_target_metric:
            best_val_target_metric = val_target_metric
            best_val_scalar_metrics = val_scalar_metrics
            es_count = 0
            torch.save(gen.state_dict(), os.path.join(args.out_dir, "best_gen_weights.pth"))
            if enc is not None: torch.save(enc.state_dict(), os.path.join(args.out_dir, "best_enc_weights.pth"))
        else: 
            es_count += 1
            if es_count >= config["train"]["patience"]: 
                logger.info("Early stopping!")
                break
    logger.info("Done training!")
    logger.info("Evaluating best model on test set")
    gen.load_state_dict(torch.load(os.path.join(args.out_dir, "best_gen_weights.pth")))
    if enc is not None:  enc.load_state_dict(torch.load(os.path.join(args.out_dir, "best_enc_weights.pth")))
    test_scalar_metrics = test(test_dataloader, enc, gen, device, split="test")
    for tag, val in test_scalar_metrics.items(): 
        writer.add_scalar(tag, val)

    test_scalar_metrics["total_time"] = str(datetime.timedelta(seconds=time.time() - start_time))
    for k in list(best_val_scalar_metrics.keys()): 
        best_val_scalar_metrics[f"best_{k}"] = best_val_scalar_metrics.pop(k)
    overall_scalar_metrics = {**test_scalar_metrics, **best_val_scalar_metrics}
    write_json(overall_scalar_metrics, os.path.join(args.out_dir, "results.json"))
    writer.close()

if __name__ == "__main__":
    main()
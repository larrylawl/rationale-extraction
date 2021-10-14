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
from utils import load_datasets, load_documents, load_instances, get_optimizer, read_json, slice_by_index, write_json

logging.basicConfig(level=logging.INFO, format='%(relativeCreated)6d %(threadName)s %(message)s')
# let's make this more or less deterministic (not resistent to restarts)

logger = logging.getLogger(__name__)
# args = None
# device = None
# base_dataset_name = None
# writer = None
# config = None

def parse_args():
    parser = argparse.ArgumentParser("Translates the files in docs.")
    parser.add_argument("--lab_data_dir", required=True, help="Input directory to data.")
    parser.add_argument("--model_dir", default=None, help="Model weights file path. If none, trains from scratch.")
    parser.add_argument("--config", required=True, help="Model config file.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tune_hp", action="store_true")
    parser.add_argument("--gen_only", action="store_true", help="Supervised training with gen only")
    parser.add_argument("--seed", required=True, type=int, default=100)

    return parser.parse_args()

def main():
    start_time = time.time()
    # global args, device, base_dataset_name, writer, config
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
    if args.gen_only: 
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
    documents: Dict[str, str] = load_documents(args.lab_data_dir, docids=None)
    train_feat, val_feat, test_feat = create_datasets_features(load_datasets(args.lab_data_dir), documents, device)

    # TODO: change to tensor dataset to avoid expensive zipping operation of pad_collate
    # create datset
    train_dataset = EraserDataset(train_feat, tokenizer, embedding_model, is_labelled=True)
    val_dataset = EraserDataset(val_feat, tokenizer, embedding_model, is_labelled=True)
    test_dataset = EraserDataset(test_feat, tokenizer, embedding_model, is_labelled=True)

    train_dataloader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)

    # instantiate models
    gen = instantiate_generator(config, device)
    enc = None if args.gen_only else instantiate_encoder(config, device)

    # instantiate optimiser
    optimizer = get_optimizer([gen, enc], config["train"]["lr"])

    # training
    gen, enc, best_val_scalar_metrics = train_loop(train_dataloader, val_dataloader, gen, enc, optimizer, config["train"]["num_epochs"], \
                                                    args.out_dir, writer, device, config["train"]["patience"], logger)

    logger.info("Evaluating best model on test set")
    test_scalar_metrics = test(test_dataloader, enc, gen, split="test")
    for tag, val in test_scalar_metrics.items(): 
        writer.add_scalar(tag, val)

    test_scalar_metrics["total_time"] = str(datetime.timedelta(seconds=time.time() - start_time))
    overall_scalar_metrics = {**test_scalar_metrics, **best_val_scalar_metrics}
    write_json(overall_scalar_metrics, os.path.join(args.out_dir, "results.json"))
    writer.close()

if __name__ == "__main__":
    main()
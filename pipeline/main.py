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
from tqdm import tqdm
import random

from pipeline.cotrain_utils import *
from utils import instantiate_models, load_datasets, load_documents, load_instances, get_optimizer, dataset_mapping, get_base_dataset_name, read_json, write_json, plot_grad_flow, tracked_named_parameters, score_hard_rationale_predictions

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

def main():
    start_time = time.time()
    global args, device, base_dataset_name, writer, config
    args = parse_args()
    logger.info(args)
    
    if os.path.exists(args.out_dir): shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    write_json(vars(args), os.path.join(args.out_dir, "exp_args.json"))

    config = read_json(args.config)
    if args.tune_hp:
        config = tune_hp(config)
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
    train_anns, val_anns, test_anns = load_datasets(args.data_dir)

    # create datset
    train_dataset = EraserDataset(train_anns, documents, tokenizer, embedding_model, logger)
    val_dataset = EraserDataset(val_anns, documents, tokenizer, embedding_model, logger)
    test_dataset = EraserDataset(test_anns, documents, tokenizer, embedding_model, logger)

    train_dataloader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)

    # instantiate models
    enc, gen = instantiate_models(config, device)

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
        train_scalar_metrics, _ = train(train_dataloader, enc, gen, optimizer, args, device)
        val_scalar_metrics = test(val_dataloader, enc, gen, device)
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
    test_scalar_metrics = test(test_dataloader, enc, gen, device, split="test")
    test_scalar_metrics["total_time"] = str(datetime.timedelta(seconds=time.time() - start_time))
    write_json(test_scalar_metrics, os.path.join(args.out_dir, "results.json"))

if __name__ == "__main__":
    main()
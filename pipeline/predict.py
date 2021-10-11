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
from utils import annotations_from_jsonl, instantiate_models, load_datasets, load_documents, load_instances, get_optimizer, read_json, write_json, plot_grad_flow, tracked_named_parameters, score_hard_rationale_predictions

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
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--config", required=True, help="Model config file.")
    parser.add_argument("--seed", required=True, type=int, default=100)

    return parser.parse_args()

def main():
    start_time = time.time()
    global args, device, base_dataset_name, writer, config
    args = parse_args()
    logger.info(args)
    set_seed(args)
    
    config = read_json(args.config)
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
    test_data = annotations_from_jsonl(os.path.join(args.data_dir, 'test.jsonl'))
    test_feat = create_datasets_features([test_data], documents, device)[0]

    # create datset
    test_dataset = EraserDataset(test_feat, tokenizer, embedding_model, logger)
    test_dataloader = DataLoader(test_dataset, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=pad_collate)

    # instantiate models
    enc, gen = instantiate_models(config, device, os.path.join(args.model_dir, "best_enc_weights.pth"), os.path.join(args.model_dir, "best_gen_weights.pth"))

    logger.info("Evaluating best model on test set")
    test_scalar_metrics = test(test_dataloader, enc, gen, device, split="test")
    test_scalar_metrics["total_time"] = str(datetime.timedelta(seconds=time.time() - start_time))
    print(test_scalar_metrics)

if __name__ == "__main__":
    main()
import sys; sys.path.insert(0, "..")
import os
import math
from functools import reduce
import numpy as np
import random
import torch
import torch.nn as nn
from operator import attrgetter
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple
from utils import PRFScore, create_instance, get_token_embeddings, higher_conf, prob_to_conf, same_label, score_hard_rationale_predictions, Annotation, Evidence, generate_document_evidence_map, get_optimizer, write_json
from models.encoder import Encoder
from models.generator import Generator

### Dataset

class EraserDataset(Dataset):
    """ ERASER dataset. """

    def __init__(self, anns, tokenizer, embedding_model, is_labelled, evd_ratio = 0.1, cotrain_mask=None):
        self.anns: AnnotationFeature = anns
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.cotrain_mask = cotrain_mask  # (max_tokens, N)
        self.evd_ratio = evd_ratio  # 1.6 / 16 for e-snli, according to ERASER paper
        self.is_labelled = is_labelled

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx: int):
        ann_id, t, r, l, algn = attrgetter("annotation_id", "text", "rationale", "label", "alignment")(self.anns[idx])
        t_e = get_token_embeddings(t, self.tokenizer, self.embedding_model)
        assert len(t_e) == len(r)
        # if it's labelled, c_mask to take rationale labels instead.
        # 
        c_mask = self.cotrain_mask[:, idx] if not self.cotrain_mask is None else None
        return t_e, len(t_e), r, l, ann_id, c_mask

@dataclass(eq=True)
class AnnotationFeature:
    annotation_id: str
    text: str 
    rationale: Tensor
    label: Tensor
    alignment: Dict[int, List[int]] = None

def pad_collate(batch):
    # print(len(batch))
    # print(type(batch))
    (t_e, t_e_lens, r, l, ann_ids, c_mask) = zip(*batch)
    # t_e_lens = [t.size()[0] for t in t_e]

    t_e_pad = pad_sequence(t_e)  # (L, bs, H_in)
    r_pad = pad_sequence(r, padding_value=-1.)  # -1 to not clash with 0, which reps not rationale
    # t_e_packed = pack_padded_sequence(t_e_pad, t_e_lens, enforce_sorted=False)
    # l = torch.tensor([dataset_mapping[x] for x in l], dtype=torch.long)
    t_e_lens = torch.tensor(t_e_lens)
    l = torch.stack(l, dim = 0)  # (bs)
    if c_mask[0] != None: c_mask = torch.stack(c_mask, dim = 1)

    return t_e_pad, t_e_lens, r_pad, l, ann_ids, c_mask

def tune_hp(config):
    # config["generator"]["selection_lambda"] = round(10 ** random.uniform(-4, -2), 5)
    # config["generator"]["continuity_lambda"] = round(10 ** random.uniform(-4, -2), 5)
    # config["train"]["lr"] = round(10 ** random.uniform(-4, -2), 5)
    # config["train"]["sup_pn"] = round(random.uniform(0.01, 0.05), 5)
    config["train"]["rat_multp"] = round(random.uniform(2, 10), 5)
    
    return config

def get_best_val_metrics(metrics: Dict[str, float]):
    return metrics["val_f1"] + metrics["val_tok_f1"]

def create_datasets_features(dataset: List[List[Annotation]], docs: Dict[str, str], device) -> List[List[AnnotationFeature]]:
    res = []
    for ds in dataset:
        features = create_features(ds, docs, device)
        res.append(features)
    return res

def create_features(anns: List[Annotation], docs: Dict[str, str], device) -> List[AnnotationFeature]:
    features = []
    for ann in anns:
        annotation_id: str = ann.annotation_id
        evidences: List[List[Evidence]] = ann.evidences
        label: str = str(ann.classification)
        docids: List[str] = [f"{annotation_id}_hypothesis", f"{annotation_id}_premise"] # only for esnli

        document_evidence_map: Dict[str, List[Tuple[int, int]]] = generate_document_evidence_map(evidences)
        assert set(document_evidence_map.keys()).issubset(set(docids)), "Evidence should come from docids!"

        text = []
        rationale = []
        for docid in docids:
            t = docs[docid]
            text.append(t)
            # get rationale
            r = [0.0] * len(t.split())
            if docid in document_evidence_map:
                for s, e in document_evidence_map[docid]: 
                    r[s:e] = [1.0] * (e - s)

            rationale.extend(r)
        rationale = torch.tensor(rationale).to(device)
        mapped_label: Tensor = torch.tensor(dataset_mapping[label]).to(device)
        ann_feature = AnnotationFeature(annotation_id, " ".join(text), rationale, mapped_label)
        features.append(ann_feature)
    return features

dataset_mapping = {
    "contradiction": 0,
    "entailment": 1,
    "neutral": 2
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


### Model

def instantiate_generator(config, device, fp=None):
    gen = Generator(config["generator"]).to(device)
    if fp: gen.load_state_dict(torch.load(fp))
    return gen

def instantiate_encoder(config, device, fp=None):
    enc = Encoder(config["encoder"]).to(device)
    if fp: enc.load_state_dict(torch.load(fp))
    return enc

def train(dataloader, enc, gen, optimizer, config, split="trg", weight_strategy="weighted"):
    running_scalar_labels = ["loss", "obj_loss", "cont_loss", "sel_loss", "mask_sup_loss", "cotrain_sup_loss", "total_f1", "tok_p", "tok_r", "tok_f1"]
    running_scalar_metrics = torch.zeros(len(running_scalar_labels))
    skipped_count = 0

    gen.train()
    if enc is not None: enc.train()
    for batch, (t_e_pad, t_e_lens, r_pad, l, _, c_mask) in enumerate(tqdm(dataloader)):  
        # forward pass
        mask = gen(t_e_pad, t_e_lens)
        # hard yet differentiable by using the same trick as https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html#
        mask_hard = (mask.detach() > 0.5).float() - mask.detach() + mask  
        
        # encoder
        if enc is not None:
            logit = enc(t_e_pad, t_e_lens, mask=mask_hard)  # NOTE: to test gen and enc independently, change mask to none
            probs = nn.Softmax(dim=1)(logit.detach())
            y_pred = torch.argmax(probs, dim=1)

            obj_loss = nn.CrossEntropyLoss()(logit, l)
            _, _, f1 = PRFScore(average="macro")(l.detach(), y_pred)
        else:
            obj_loss = torch.tensor(0)
            f1 = 0

        # NOTE: training metrics won't be accurate as only cotrained idxs are labelled
        tok_p, tok_r, tok_f1 = score_hard_rationale_predictions(mask_hard.detach(), r_pad.detach(), t_e_lens)

        # compute losses
        ## sel and cont loss
        selection_cost, continuity_cost = gen.loss(mask)

        ## sup rationale loss
        if dataloader.dataset.is_labelled:
            weight = r_pad.clone()
            weight[weight == -1] = 0  # padding values
            if weight_strategy == "default":
                weight[weight == 0] = 1
            elif weight_strategy == "weighted":
                weight[weight == 1] = 1 - dataloader.dataset.evd_ratio  # rationales
                weight[weight == 0] = dataloader.dataset.evd_ratio  # non rationales
            mask_sup_loss = nn.BCELoss(weight, reduction="sum")(mask, r_pad) / torch.count_nonzero(weight)  # manual average
            mask_sup_loss = torch.nan_to_num(mask_sup_loss)  # if no labels
            # print(f"loss: {nn.BCELoss(weight, reduction='sum')(mask, r_pad)}")
            # print(torch.count_nonzero(weight))
            # print(f"mask_sup_loss: {mask_sup_loss}")
        else: mask_sup_loss = torch.tensor(0)

        ## cotrain rationale loss
        if not c_mask[0] == None:
            # TODO: try scaling confidence to quadratic x^2
            # NOTE: c_mask naturally won't select supervised labels
            # only apply BCE on nonzero values of cotrain mask
            if config["cotrain"]["perfect"]: 
                assert torch.equal(r_pad[(c_mask+1).nonzero(as_tuple=True)], c_mask[(c_mask+1).nonzero(as_tuple=True)]), f"{r_pad} != {c_mask}"
            mask_pred = mask[(c_mask + 1).nonzero(as_tuple=True)]  # dim == 1
            mask_y_prob = c_mask[c_mask != -1] # dim == 1 
            mask_y_conf = prob_to_conf(mask_y_prob)
            mask_y = (mask_y_prob > 0.5).float()
            weight = mask_y_conf * torch.where(mask_y == 1, 1 - dataloader.dataset.evd_ratio, dataloader.dataset.evd_ratio)
            # total possible rationale: len(r_pad + 1.nonzero())
            lab_pn = len(mask_pred) / len((r_pad+1).nonzero())
            cotrain_sup_loss = nn.BCELoss(weight)(mask_pred, mask_y) * lab_pn
            cotrain_sup_loss = torch.nan_to_num(cotrain_sup_loss)  # if no self-labels
        else: cotrain_sup_loss = torch.tensor(0)

        loss = obj_loss + selection_cost + continuity_cost + mask_sup_loss + cotrain_sup_loss

        if loss.item() == 0: 
            skipped_count += 1
            continue  # not all batches have labels

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # tracking metrics
        running_scalar_metrics += torch.tensor([loss.detach(), obj_loss.detach(), continuity_cost.detach(), selection_cost.detach(), mask_sup_loss.detach(), cotrain_sup_loss.detach(), f1, tok_p, tok_r, tok_f1])            

    total_scalar_metrics = running_scalar_metrics / ((batch + 1) - skipped_count)
    scalar_metrics = {}
    for i in range(len(running_scalar_labels)): scalar_metrics[f"{split}_{running_scalar_labels[i]}"] = total_scalar_metrics[i]

    return enc, gen, scalar_metrics

def test(dataloader, enc, gen, split="val"):
    running_scalar_labels = [f"{split}_f1", f"{split}_tok_precision", f"{split}_tok_recall", f"{split}_tok_f1", f"{split}_mask_sup_loss"]
    running_scalar_metrics = torch.zeros(len(running_scalar_labels))
    
    gen.eval()
    if enc is not None: enc.eval()
    with torch.no_grad():
        for batch, (t_e_pad, t_e_lens, r_pad, l, _, _) in enumerate(tqdm(dataloader)):        
            mask = gen(t_e_pad, t_e_lens)
            mask_hard = (mask > 0.5).float()
            weight = r_pad.clone()
            weight[weight == 1] = 1 - dataloader.dataset.evd_ratio  # rationales
            weight[weight == 0] = dataloader.dataset.evd_ratio  # non rationales
            weight[weight == -1] = 0  # padding values
            mask_sup_loss = nn.BCELoss(weight, reduction="sum")(mask, r_pad) / torch.count_nonzero(weight)  # manual average )
            mask_sup_loss = torch.nan_to_num(mask_sup_loss)  # if no labels
            tok_p, tok_r, tok_f1 = score_hard_rationale_predictions(mask_hard, r_pad, t_e_lens)

            if enc is not None:
                logit = enc(t_e_pad, t_e_lens, mask=mask_hard)  # NOTE: to test gen and enc independently, change mask to none
                probs = nn.Softmax(dim=1)(logit)
                y_pred = torch.argmax(probs, dim=1)
                _, _, f1 = PRFScore(average='macro')(l, y_pred)
            else:
                f1 = torch.tensor(0)

            running_scalar_metrics += torch.tensor([f1, tok_p, tok_r, tok_f1, mask_sup_loss])

        total_scalar_metrics = running_scalar_metrics / (batch + 1)
        scalar_metrics = {}
        for i in range(len(running_scalar_labels)): scalar_metrics[running_scalar_labels[i]] = total_scalar_metrics[i].item()

        return scalar_metrics

def train_loop(train_dl, train_ul_dl, val_dl, gen, enc, optimizer, out_dir, writer, device, config, logger):
    # instantiate optimisers
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2)
    best_val_target_metric = 0
    best_val_scalar_metrics = {}
    es_count = 0

    for t in range(config["train"]["num_epochs"]):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        if train_ul_dl: enc, gen, train_ul_scalar_metrics = train(train_ul_dl, enc, gen, optimizer, config, split="trg_ul")
        else: train_ul_scalar_metrics = {}
        enc, gen, train_scalar_metrics = train(train_dl, enc, gen, optimizer, config)
        val_scalar_metrics = test(val_dl, enc, gen)
        overall_scalar_metrics = {**train_scalar_metrics, **train_ul_scalar_metrics, **val_scalar_metrics}
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
            torch.save(gen.state_dict(), os.path.join(out_dir, "best_gen_weights.pth"))
            if enc is not None: torch.save(enc.state_dict(), os.path.join(out_dir, "best_enc_weights.pth"))
        else: 
            es_count += 1
            if es_count >= config["train"]["patience"]: 
                logger.info("Early stopping!")
                break
    gen.load_state_dict(torch.load(os.path.join(out_dir, "best_gen_weights.pth")))
    if enc is not None: torch.save(enc.state_dict(), os.path.join(out_dir, "best_enc_weights.pth"))
    write_json(best_val_scalar_metrics, os.path.join(out_dir, "best_val_results.json"))
    logger.info("Done training!")
    return gen, enc, best_val_scalar_metrics
    
def label(prob_a: Tensor, prob_b: Tensor, fns):
    res = reduce(lambda res, fn: res and fn(prob_a, prob_b), fns, True)
    return res

def parallelising(ctx, fn, src_args, tgt_args):
    src_done_queue = ctx.SimpleQueue()

    src_p = ctx.Process(target=fn, args=(src_args, src_done_queue))
    src_p.start()
    tgt_done_queue = ctx.SimpleQueue()
    tgt_p = ctx.Process(target=fn, args=(tgt_args, tgt_done_queue))
    tgt_p.start()

    src_p.join()
    tgt_p.join()
    src_results = src_done_queue.get()
    tgt_results = tgt_done_queue.get()

    return src_results, tgt_results


### TESTS
def test_label():
    prob_a = torch.tensor(0.2)
    prob_b = torch.tensor(0.4)
    fns = [same_label]
    assert label(prob_a, prob_b, fns) == True

    prob_a = torch.tensor(0.2)
    prob_b = torch.tensor(0.6)
    fns = [same_label]
    assert label(prob_a, prob_b, fns) == False

    prob_a = torch.tensor(0.2)
    prob_b = torch.tensor(0.9)
    fns = [higher_conf]
    assert label(prob_a, prob_b, fns) == False

    prob_a = torch.tensor(0.4)
    prob_b = torch.tensor(0.45)
    fns = [same_label, higher_conf]
    assert label(prob_a, prob_b, fns) == True


def extractor_worker(x, done_queue):
    done_queue.put(x.numpy())

def test_parallelising():
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    src_res, tgt_res = parallelising(ctx, extractor_worker, torch.tensor(1.0), torch.tensor(2.0))

if __name__ == "__main__":
    print("Running unit tests...")
    # test_label()
    test_parallelising()
    print("Unit tests passed!")
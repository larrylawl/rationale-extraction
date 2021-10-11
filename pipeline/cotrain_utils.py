import sys; sys.path.insert(0, "..")
import math
import numpy as np
import random
import torch
import torch.nn as nn
from operator import attrgetter
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple
from utils import PRFScore, create_instance, get_token_embeddings, score_hard_rationale_predictions, Annotation, Evidence, generate_document_evidence_map

class EraserDataset(Dataset):
    """ ERASER dataset. """

    def __init__(self, anns, tokenizer, embedding_model, logger, cotrain_mask=None):
        self.anns: AnnotationFeature = anns
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.logger = logger
        self.cotrain_mask = cotrain_mask  # (max_tokens, N)

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx: int):
        ann_id, t, r, l, algn = attrgetter("annotation_id", "text", "rationale", "label", "alignment")(self.anns[idx])
        t_e = get_token_embeddings(t, self.tokenizer, self.embedding_model)
        assert len(t_e) == len(r)
        # t_e, r, l, ann_id = create_instance(self.anns[idx], self.docs, self.tokenizer, self.embedding_model, self.logger)
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
    r_pad = pad_sequence(r)
    # t_e_packed = pack_padded_sequence(t_e_pad, t_e_lens, enforce_sorted=False)
    # l = torch.tensor([dataset_mapping[x] for x in l], dtype=torch.long)
    l = torch.stack(l, dim = 0)  # (bs)
    if c_mask[0] != None: c_mask = torch.stack(c_mask, dim = 1)

    return t_e_pad, t_e_lens, r_pad, l, ann_ids, c_mask

def tune_hp(config):
    # config["generator"]["selection_lambda"] = round(10 ** random.uniform(-4, -2), 5)
    # config["generator"]["continuity_lambda"] = round(10 ** random.uniform(-4, -2), 5)
    # config["train"]["lr"] = round(10 ** random.uniform(-4, -2), 5)
    config["train"]["sup_pn"] = round(random.uniform(0.01, 0.05), 5)
    
    return config

def train(dataloader, enc, gen, optimizer, args, device, config):
    running_scalar_labels = ["trg_loss", "trg_obj_loss", "trg_cont_loss", "trg_sel_loss", "trg_mask_sup_loss", "trg_total_f1", "trg_tok_precision", "trg_tok_recall", "trg_tok_f1"]
    running_scalar_metrics = torch.zeros(len(running_scalar_labels))
    # total_params = len(tracked_named_parameters(chain(gen.named_parameters(), enc.named_parameters())))
    # mean_grads = torch.zeros(total_params).to(device)
    # var_grads = torch.zeros(total_params).to(device)

    gen.train()
    enc.train()
    size = config["train"]["sup_pn"] * len(dataloader.dataset)
    labelled_batch: int = math.ceil(size / dataloader.batch_size)
    # labelled_batch_idx = random.sample(range(len(dataloader)), )

    for batch, (t_e_pad, t_e_lens, r_pad, l, _, c_mask) in enumerate(tqdm(dataloader)):  
        # forward pass
        mask = gen(t_e_pad, t_e_lens)
        # hard yet differentiable by using the same trick as https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html#
        mask_hard = (mask.detach() > 0.5).float() - mask.detach() + mask  
        logit = enc(t_e_pad, t_e_lens, mask=mask_hard)  # NOTE: to test gen and enc independently, change mask to none
        probs = nn.Softmax(dim=1)(logit.detach())
        y_pred = torch.argmax(probs, dim=1)

        # compute losses
        selection_cost, continuity_cost = gen.loss(mask)
        if batch < labelled_batch: mask_sup_loss = nn.BCELoss()(mask, r_pad)
        elif not c_mask is None:
            # TODO: check if mask is correct

            # only apply BCE on nonzero values of cotrain mask
            mask_pred = mask[(c_mask + 1).nonzero(as_tuple=True)] # (L, bs)
            mask_y_prob = c_mask[c_mask != -1] # (max_tokens, bs)
            mask_y = (mask_y_prob > 0.5).float()
            mask_sup_loss = nn.BCELoss(mask_y_prob)(mask_pred, mask_y) 
            mask_sup_loss = torch.nan_to_num(mask_sup_loss)  # if no self-labels
            print(f"mask_sup_loss: {mask_sup_loss}")
        else: mask_sup_loss = torch.tensor(0)
        obj_loss = nn.CrossEntropyLoss()(logit, l)
        loss = obj_loss + selection_cost + continuity_cost + mask_sup_loss

        _, _, f1 = PRFScore(average="macro")(l.detach(), y_pred)
        tok_p, tok_r, tok_f1 = score_hard_rationale_predictions(mask_hard.detach(), r_pad.detach(), t_e_lens)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # tracking metrics
        running_scalar_metrics += torch.tensor([loss.detach(), obj_loss.detach(), continuity_cost.detach(), selection_cost.detach(), mask_sup_loss.detach(), f1, tok_p, tok_r, tok_f1])

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

def test(dataloader, enc, gen, device, split="val"):
    running_scalar_labels = [f"{split}_f1", f"{split}_tok_precision", f"{split}_tok_recall", f"{split}_tok_f1"]
    running_scalar_metrics = torch.zeros(len(running_scalar_labels))
    
    gen.eval()
    enc.eval()
    with torch.no_grad():
        for batch, (t_e_pad, t_e_lens, r_pad, l, _, _) in enumerate(tqdm(dataloader)):        
            r_pad = r_pad.to(device)
            l = l.to(device)

            mask = gen(t_e_pad, t_e_lens)
            mask_hard = (mask.detach() > 0.5).float() - mask.detach() + mask  
            logit = enc(t_e_pad, t_e_lens, mask=mask_hard)  # NOTE: to test gen and enc independently, change mask to none
            probs = nn.Softmax(dim=1)(logit.detach())
            y_pred = torch.argmax(probs, dim=1)
            _, _, f1 = PRFScore(average='macro')(l.detach(), y_pred)
            tok_p, tok_r, tok_f1 = score_hard_rationale_predictions(mask_hard.detach(), r_pad.detach(), t_e_lens)

            running_scalar_metrics += torch.tensor([f1, tok_p, tok_r, tok_f1])

        total_scalar_metrics = running_scalar_metrics / (batch + 1)
        scalar_metrics = {}
        for i in range(len(running_scalar_labels)): scalar_metrics[running_scalar_labels[i]] = total_scalar_metrics[i].item()
        return scalar_metrics

def create_datasets_features(dataset: List[Annotation], docs: Dict[str, str], device) -> List[List[AnnotationFeature]]:
    res = []
    for ds in dataset:
        features = create_features(ds, docs, device)
        res.append(features)
    return res

def create_features(anns: Annotation, docs: Dict[str, str], device) -> List[AnnotationFeature]:
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

class MaskBCELoss:
    """ BCE loss that skips computation for value 0. 
    Let x = [1, 1, 0]
    f(x) = x_1 + x_2 = 3
    df(x) / dx = [df(x) / dx_1, ..., df(x) / dx_3] = [1, 1, 0]
    Gradient for skipped value (i.e. x_3) will then be 0.
    """
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        # only compute BCE for 
        # bce_loss = -1 * (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

        # compute loss for only specific elements. 
        # if i compute for everything, it'll detect operation all.

        loss = torch.where(pred == 0, 
                    pred,
                    torch.tensor(1.0))  # BCE

        avg_loss = sum(loss) / torch.count_nonzero(pred)
        return avg_loss

# TESTS
def test_custom_bce_loss():
    mask_bce_loss = MaskBCELoss()
    loss = nn.BCELoss()
    m = nn.Sigmoid()

    # test for normal bce
    input = torch.randn(3, requires_grad=True)
    target = torch.empty(3).random_(2)
    output = mask_bce_loss(m(input), target)
    output.backward()
    print(input.grad)
    expected_output = loss(m(input), target)
    # assert torch.equal(output, expected_output), f"{output} != {expected_output}"

    # test for skip
    # test for normal bce
    input = torch.tensor([0.2, 0, 0.8], requires_grad=True)
    target = torch.tensor([0, 1, 1])
    output = mask_bce_loss(input, target)
    expected_output = -torch.log(torch.tensor(0.8))
    output.backward()
    print(input.grad)
    # expected_output = loss(m(input), target)
    # assert torch.equal(output, expected_output), f"{output} != {expected_output}"


if __name__ == "__main__":
    print("Running unit tests...")
    test_custom_bce_loss()
    print("Unit tests passed!")
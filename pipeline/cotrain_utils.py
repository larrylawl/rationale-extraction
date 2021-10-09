import sys; sys.path.insert(0, "..")
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import tqdm
from sklearn.metrics import f1_score
from utils import dataset_mapping, create_instance, score_hard_rationale_predictions

class EraserDataset(Dataset):
    """ ERASER dataset. """

    def __init__(self, anns, docs, tokenizer, embedding_model, logger, cotrain_mask=None):
        self.anns = anns
        self.docs = docs
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.logger = logger
        self.cotrain_mask = cotrain_mask  # (max_tokens, N)

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx: int):
        t_e, r, l, ann_id = create_instance(self.anns[idx], self.docs, self.tokenizer, self.embedding_model, self.logger)
        c_mask = self.cotrain_mask[:, idx] if not self.cotrain_mask is None else None
        return t_e, r, l, ann_id, c_mask

def pad_collate(batch):
    # print(len(batch))
    # print(type(batch))
    (t_e, r, l, ann_id, c_mask) = zip(*batch)
    t_e_lens = [t.size()[0] for t in t_e]

    t_e_pad = pad_sequence(t_e)  # (L, N, H_in)
    r_pad = pad_sequence(r)
    # t_e_packed = pack_padded_sequence(t_e_pad, t_e_lens, enforce_sorted=False)
    l = torch.tensor([dataset_mapping[x] for x in l], dtype=torch.long)

    return t_e_pad, t_e_lens, r_pad, l, ann_id, c_mask

def tune_hp(config):
    config["generator"]["selection_lambda"] = round(10 ** random.uniform(-4, -2), 5)
    config["generator"]["continuity_lambda"] = round(10 ** random.uniform(-4, -2), 5)
    config["train"]["lr"] = round(10 ** random.uniform(-4, -2), 5)
    
    return config

def train(dataloader, enc, gen, optimizer, args, device):
    running_scalar_labels = ["trg_loss", "trg_obj_loss", "trg_cont_loss", "trg_sel_loss", "trg_mask_sup_loss", "trg_total_f1", "trg_tok_precision", "trg_tok_recall", "trg_tok_f1"]
    running_scalar_metrics = torch.zeros(len(running_scalar_labels))
    # total_params = len(tracked_named_parameters(chain(gen.named_parameters(), enc.named_parameters())))
    # mean_grads = torch.zeros(total_params).to(device)
    # var_grads = torch.zeros(total_params).to(device)

    gen.train()
    enc.train()

    for batch, (t_e_pad, t_e_lens, r_pad, l, _, c_mask) in enumerate(tqdm(dataloader)):  
        # to device
        r_pad = r_pad.to(device)
        l = l.to(device)

        # TODO: co-training
        # Assign sub_cotrain_mask = cotrain_mask[mask.size(0):], batch number same, but likely need to slice rows (sequence length)
        # use same dataloader since share same train method

        # forward pass
        mask = gen(t_e_pad, t_e_lens)
        # hard yet differentiable by using the same trick as https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html#
        mask_hard = (mask.detach() > 0.5).float() - mask.detach() + mask  
        logit = enc(t_e_pad, t_e_lens, mask=mask_hard)  # NOTE: to test gen and enc independently, change mask to none
        probs = nn.Softmax(dim=1)(logit.detach())
        y_pred = torch.argmax(probs, dim=1)

        # compute losses
        selection_cost, continuity_cost = gen.loss(mask)
        if "sup" in args and args.sup: mask_sup_loss = nn.BCELoss()(mask, r_pad)
        else: mask_sup_loss = torch.tensor(0)
        obj_loss = nn.CrossEntropyLoss()(logit, l)
        loss = obj_loss + selection_cost + continuity_cost + mask_sup_loss
        f1 = f1_score(l.detach().cpu(), y_pred.cpu(), average="macro")
        tok_p, tok_r, tok_f1 = score_hard_rationale_predictions(mask_hard.detach().cpu(), r_pad.detach().cpu(), t_e_lens)

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
            f1 = f1_score(l.detach().cpu(), y_pred.cpu(), average="macro")
            tok_p, tok_r, tok_f1 = score_hard_rationale_predictions(mask_hard.detach().cpu(), r_pad.detach().cpu(), t_e_lens)

            running_scalar_metrics += torch.tensor([f1, tok_p, tok_r, tok_f1])

        total_scalar_metrics = running_scalar_metrics / (batch + 1)
        scalar_metrics = {}
        for i in range(len(running_scalar_labels)): scalar_metrics[running_scalar_labels[i]] = total_scalar_metrics[i].item()
        return scalar_metrics
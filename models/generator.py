import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import pdb

'''
    The generator selects a rationale z from a document x that should be sufficient
    for the encoder to make it's prediction.

    Several froms of Generator are supported. Namely CNN with arbitary number of layers, and @taolei's FastKNN
'''
class Generator(nn.Module):

    def __init__(self, config):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(**config["lstm"])
        self.dropout = nn.Dropout(config["dropout"])
        self.fc = nn.Linear(config["linear"]["in_features"], 2)  # rationale or not
        self.log_sm = nn.LogSoftmax(dim=-1)  
        self.tau = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.selection_lambda = config["selection_lambda"]
        self.continuity_lambda = config["continuity_lambda"]

    def forward(self, t_e_pad, t_e_lens,):
        '''
            Given input x_indx of dim (batch, length), return z (batch, length) such that z
            can act as element-wise mask on x
        '''
        t_e_packed = pack_padded_sequence(t_e_pad, t_e_lens, enforce_sorted=False)
        out_packed, _ = self.lstm(t_e_packed)  
        out_padded, out_lens = pad_packed_sequence(out_packed) # (L, N, D*H_out)
        out = self.dropout(out_padded) 
        out = self.fc(out)  # (L, N, 2)
        logits = self.log_sm(out)
        mask_prob = F.gumbel_softmax(logits, tau=self.tau, hard=False)  # (L, N, 2)
        mask = mask_prob[:, :, 1]  # [:, :, 1] := prob of the token being a rationale; # (L, N)
        return mask

    def loss(self, mask):
        '''
            Compute the generator specific costs, i.e selection cost, continuity cost, and global vocab cost
        '''
        # TODO: removing padding from mask
        selection_cost = torch.mean( torch.sum(mask, dim=0) )
        l_padded_mask =  torch.cat( [mask[[0], :], mask] , dim=0)  # (1, N) w (L, N)
        r_padded_mask =  torch.cat( [mask, mask[[-1], :]] , dim=0)
        continuity_cost = torch.mean( torch.sum( torch.abs( l_padded_mask - r_padded_mask ) , dim=0) )
        return self.selection_lambda * selection_cost, self.continuity_lambda * continuity_cost

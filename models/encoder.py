import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import pdb

class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(**config["lstm"])
        self.dropout = nn.Dropout(config["dropout"])
        self.fc = nn.Linear(config["linear"]["in_features"], config["num_classes"])

    def forward(self, t_e_pad, t_e_lens, mask=None):
        '''
            x: PackedSequence
            mask: Mask to apply over embeddings for tao rationales
        '''
        # NOTE: not ideal, but pytorch has no RNN masking layers. suspect that zero vector inputs will lead to zero gradients being backpropped.
        if not mask is None: 
            t_e_pad = (t_e_pad.T * mask.T).T  # ((L, N, D*H_out).T * (L, N).T).T = ((D*H_out, N, L) * (N, L)).T = (L, N, D*H_out)

        t_e_packed = pack_padded_sequence(t_e_pad, t_e_lens, enforce_sorted=False)
        out_packed, _ = self.lstm(t_e_packed)
        out_padded, out_lens = pad_packed_sequence(out_packed)  # (L, N, D*H_out), (N)
        out_fwd_padded, out_bwd_padded = torch.chunk(out_padded, 2, dim=2)  # (L, N, H_out)
        # use last op of normal rnn and first output of reverse rnn as these outputs have seen entire sequence
        # source: https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
        # take last token excluding padding. 
        # source: https://discuss.pytorch.org/t/how-to-select-specific-vector-in-3d-tensor-beautifully/37724
        out_fwd_last = out_fwd_padded[out_lens - 1, torch.arange(out_fwd_padded.size(1))] # (64, H_out)
        out_cat = torch.cat((out_fwd_last, out_bwd_padded[0]), 1)  
        out = self.dropout(out_cat) 
        logit = self.fc(out)
        return logit

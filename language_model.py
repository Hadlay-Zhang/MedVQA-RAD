"""
This code is from Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang's repository.
https://github.com/jnhwkim/ban-vqa
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout, op=''):
        super(WordEmbedding, self).__init__()
        self.op = op
        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
        if 'c' in op:
            self.emb_ = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
            self.emb_.weight.requires_grad = False # fixed
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file, tfidf=None, tfidf_weights=None):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init
        if tfidf is not None:
            if 0 < tfidf_weights.size:
                weight_init = torch.cat([weight_init, torch.from_numpy(tfidf_weights)], 0)
            weight_init = tfidf.matmul(weight_init) # (N x N') x (N', F)
            self.emb_.weight.requires_grad = True
        if 'c' in self.op:
            self.emb_.weight.data[:self.ntoken] = weight_init.clone()

    def forward(self, x):
        emb = self.emb(x)
        if 'c' in self.op:
            emb = torch.cat((emb, self.emb_(x)), 2)
        emb = self.dropout(emb)
        return emb

class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        super(QuestionEmbedding, self).__init__()
        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.bidirect = bidirect
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)
        
        if rnn_type in ['LSTM', 'GRU']:
            rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
            self.rnn = rnn_cls(in_dim, num_hid, nlayers, bidirectional=bidirect, dropout=dropout, batch_first=True)
        elif rnn_type == 'ConvGRU':
            # self.conv2 = nn.Conv1d(in_dim, in_dim // 2, kernel_size=2, padding=0)  # Head padding
            self.conv3 = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1)  # Symmetric padding
            self.rnn = nn.GRU(in_dim, num_hid, nlayers, bidirectional=bidirect, dropout=dropout if nlayers > 1 else 0, batch_first=True)
        else:
            raise ValueError(f"{rnn_type} is not a supported rnn_type.")

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid // self.ndirections)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new_zeros(hid_shape)), Variable(weight.new_zeros(hid_shape)))
        elif self.rnn_type == 'GRU':
            return Variable(weight.new_zeros(hid_shape))
        # No initialization required for ConvGRU since hidden state is automatically initialized in GRU layer

    def forward(self, x):
        return self.forward_all(x)[:, -1]

    def forward_all(self, x):
        batch = x.size(0)

        if self.rnn_type == 'ConvGRU':
            x = x.transpose(1, 2)  # [B, in_dim, seq_len] for Conv1d
            # conv2_out = self.conv2(F.pad(x, (1, 0)))
            conv3_out = self.conv3(x)
            x = conv3_out.transpose(1, 2)
            # x = torch.cat((conv2_out, conv3_out), dim=1)  # Concatenate along the channel dimension
            # x = x.transpose(1, 2)  # Back to [B, seq_len, in_dim] for RNN

        hidden = self.init_hidden(batch) if self.rnn_type in ['LSTM', 'GRU'] else None
        output, hidden = self.rnn(x, hidden) if self.rnn_type in ['LSTM', 'GRU'] else self.rnn(x)

        return output

# class QuestionEmbedding(nn.Module):
#     def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
#         """Module for question embedding
#         """
#         super(QuestionEmbedding, self).__init__()
#         assert rnn_type == 'LSTM' or rnn_type == 'GRU'
#         if rnn_type == 'LSTM':
#             rnn_cls = nn.LSTM
#         elif rnn_type == 'GRU':
#             rnn_cls = nn.GRU
#         else:
#             rnn_cls = None

#         self.rnn = rnn_cls(
#             in_dim, num_hid, nlayers,
#             bidirectional=bidirect,
#             dropout=dropout,
#             batch_first=True)

#         self.in_dim = in_dim
#         self.num_hid = num_hid
#         self.nlayers = nlayers
#         self.rnn_type = rnn_type
#         self.ndirections = 1 + int(bidirect)
#     def init_hidden(self, batch):
#         # just to get the type of tensor
#         weight = next(self.parameters()).data
#         hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid // self.ndirections)
#         if self.rnn_type == 'LSTM':
#             return (Variable(weight.new(*hid_shape).zero_()),
#                     Variable(weight.new(*hid_shape).zero_()))
#         else:
#             return Variable(weight.new(*hid_shape).zero_())

#     def forward(self, x):
#         # x: [batch, sequence, in_dim]
#         batch = x.size(0)
#         hidden = self.init_hidden(batch)
#         output, hidden = self.rnn(x, hidden)

#         if self.ndirections == 1:
#             return output[:, -1]

#         forward_ = output[:, -1, :self.num_hid]
#         backward = output[:, 0, self.num_hid:]
#         return torch.cat((forward_, backward), dim=1)

#     def forward_all(self, x):
#         # x: [batch, sequence, in_dim]
#         batch = x.size(0)
#         hidden = self.init_hidden(batch)
#         output, hidden = self.rnn(x, hidden)
#         return output

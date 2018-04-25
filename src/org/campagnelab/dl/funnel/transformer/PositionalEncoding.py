import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.max_len=max_len
        self.dropout = nn.Dropout(p=dropout)
        self.pe_dict={}
        self.initialize(d_model)


    def initialize(self, d_model):
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.max_len, d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe_dict[d_model] = pe
        self.register_buffer('pe_'+str(d_model), pe)

    def reconfigure(self, layer_manager,N, d_model=-1):
        for layer_index in range(N):
            d_model=layer_manager.get_output_dim(layer_index) if d_model==-1 else d_model
            self.initialize(d_model)

    def forward(self, x):
        d_model=x.size(2)
        if d_model not in self.pe_dict:
            self.initialize(d_model)
        pe = self.pe_dict[d_model]
        x = x + Variable(pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class EmbeddingEncoding(nn.Module):
    "Use instead of positional encoding to learn a positional embedding."

    def __init__(self, d_model, dropout, max_len=5000):
        super(EmbeddingEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding=nn.Embedding(max_len,d_model)
        self.max_len=max_len
        self.d_model=d_model


    def forward(self, x):
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1)
        # we add the position embedding to x, then apply dropout:
        return self.dropout(x + self.embedding(x))
import math
import torch
import torch.nn as nn

from src.org.campagnelab.dl.funnel.transformer.EncoderDecoder import clones
import torch.nn.functional as F

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_model_input = d_model
        self.d_model_output = d_model
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def reconfigure(self, layer_index, layer_manager,layer_dim_in=-1, layer_dim_out=-1):
        d_model_input=layer_manager.get_input_dim(layer_index) if layer_dim_in==-1 else layer_dim_in
        d_model_output=layer_manager.get_output_dim(layer_index) if layer_dim_out==-1 else layer_dim_out
        assert d_model_input % self.h == 0
        self.d_k = d_model_input // self.h
        self.d_model_input=d_model_input
        self.d_model_output=d_model_output
        self.linears = clones(nn.Linear(d_model_input, d_model_input), 4)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        #assert encoding_size==query.size(2)
        #assert encoding_size==key.size(2)
        #assert encoding_size==value.size(2)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        #assert encoding_size==query.size(2)
        #assert encoding_size==key.size(2)
        #assert encoding_size==value.size(2)
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
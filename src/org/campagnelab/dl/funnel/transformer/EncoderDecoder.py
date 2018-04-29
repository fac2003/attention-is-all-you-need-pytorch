import copy

import torch
import torch.nn as nn


clone=copy.deepcopy

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, N):
        super(EncoderDecoder, self).__init__()
        self.encoder_src = clone(encoder)
        #self.encoder_tgt = clone(encoder)
        self.decoder = clone(decoder)
        self.src_embed = clone(src_embed)
        self.tgt_embed = clone(tgt_embed)
        self.generator = clone(generator)
        self.N = N

    def reconfigure(self, layer_manager):
        self.encoder_src.reconfigure(layer_manager)
        #self.encoder_tgt.reconfigure(layer_manager)
        self.decoder.reconfigure(layer_manager)
        self.src_embed[1].reconfigure(layer_manager, self.N)
        encoded_dim=layer_manager.get_output_dim(layer_index=self.N-1)
        self.tgt_embed[1].reconfigure(layer_manager, self.N,d_model=encoded_dim)

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        encoded, src_mask=self.encode(src, src_mask)
        return self.decode(encoded, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder_src(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        target_embeded = self.tgt_embed(tgt)
        # optionally here, if the encoding of the target is low level, we could
        # send the target through an encoder. May be useful when we compress the
        # sequences.

        #target_embeded=self.encoder_tgt(self.tgt_embed(tgt), tgt_mask)
        return self.decoder(target_embeded, memory, src_mask, tgt_mask)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.features_size = features
        self.eps = eps

    def reconfigure(self, layer_index, layer_manager, features_size=-1):
        # layer norm is applied to the output of the layer:
        features_size = layer_manager.get_output_dim(layer_index) if features_size==-1 else features_size
        self.a_2 = nn.Parameter(torch.ones(features_size))
        self.b_2 = nn.Parameter(torch.zeros(features_size))
        self.features_size=features_size

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        #print("x  {} features_size {} ".format(x.size(),self.features_size))
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class FunnelSublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(FunnelSublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.linear = None
        self.d_model_in = size
        self.d_model_out = size
        self.layer_index = -1

    def reconfigure(self, layer_index, layer_manager, layer_dim_in=-1, layer_dim_out=-1):
        d_model_in = layer_manager.get_input_dim(layer_index) if layer_dim_in == -1 else layer_dim_in
        d_model_out = layer_manager.get_output_dim(layer_index) if layer_dim_out == -1 else layer_dim_out
        self.linear = nn.Linear(d_model_in + d_model_in, d_model_out)
        self.d_model_in = d_model_in
        self.d_model_out = d_model_out
        self.layer_index = layer_index
        clone(self.norm).reconfigure(layer_index=layer_index,layer_manager=layer_manager,
                              features_size=d_model_in)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        assert x.size(2) == self.d_model_in, "x.size(2) {} must match self.d_model_in {}".format(x.size(2) ,
                                                                                                 self.d_model_in)
        residual = x
        layer_output_normalized = self.dropout(sublayer(self.norm(x)))
        assert layer_output_normalized.size(2) == self.d_model_in
        if self.linear is None:  # or x.size(2)==layer_output_normalized.size(2):
            return x + layer_output_normalized
        else:
            #print("residual {} layer output {} ".format(residual.size(), layer_output_normalized.size()))
            residual_reshaped = residual.view(-1, self.d_model_in)
            layer_output_reshaped = layer_output_normalized.view(-1, self.d_model_in)
            return self.linear(torch.cat([residual_reshaped, layer_output_reshaped], dim=1)).view(x.size(0), -1,
                                                                                                  self.d_model_out)


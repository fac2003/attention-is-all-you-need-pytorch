
import copy

import torch
import torch.nn as nn

from src.org.campagnelab.dl.funnel.transformer.EncoderDecoder import clones, LayerNorm, FunnelSublayerConnection

clone=copy.deepcopy
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norms = clones(LayerNorm(layer.output_size), N)

    def reconfigure(self, layer_manager):
        for layer_index, layer in enumerate(self.layers):
            encoder_layer_index = layer_index
            layer.reconfigure(encoder_layer_index, layer_manager)

        for layer_index, norm in enumerate(self.norms):
            norm.reconfigure(layer_index, layer_manager)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer_index, layer in enumerate(self.layers):
            x = layer(x, mask)
            x = self.norms[layer_index](x)
        return x

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(FunnelSublayerConnection(size, dropout), 2)
        self.output_size = size
        self.layer_index = -1

    def reconfigure(self, layer_index, layer_manager):
        layer_dim_in = layer_manager.get_input_dim(layer_index)
        layer_dim_out = layer_manager.get_output_dim(layer_index)
        clone(self.self_attn).reconfigure(layer_index, layer_manager)
        clone(self.feed_forward).reconfigure(layer_index, layer_manager)
        # the first one needs to output num_out dim from this layer.
        clone(self.sublayer[0]).reconfigure(layer_index, layer_manager, layer_dim_in, layer_dim_out)
        # the second one already sees layer_dim_out and converts to layer_dim_out again
        clone(self.sublayer[1]).reconfigure(layer_index, layer_manager, layer_dim_out, layer_dim_out)
        self.layer_index = layer_index
        self.output_size = layer_dim_out
        self.layer_dim_in = layer_dim_in
        self.layer_dim_out = layer_dim_out

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x:        self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


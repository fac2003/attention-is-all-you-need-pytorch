import copy

import torch
import torch.nn as nn

from src.org.campagnelab.dl.funnel.transformer.EncoderDecoder import clones, LayerNorm, FunnelSublayerConnection


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.N = N
        self.norms = clones(LayerNorm(layer.size), N)

    def reconfigure(self, layer_manager):
        pass
        # first_decoder_layer=self.N - 1
        # for layer_index, layer in enumerate(self.layers):
        #     decoder_layer_index = self.N - 1 - layer_index
        #     layer.reconfigure( self.N - 1, layer_manager)
        #
        # for layer_index, norm in enumerate(self.norms):
        #     norm.reconfigure(layer_index, layer_manager)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer_index, layer in enumerate(self.layers):
            x = layer(x, memory, src_mask, tgt_mask)
            x = self.norms[layer_index](x)
        return x


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(FunnelSublayerConnection(size, dropout), 3)

    def reconfigure(self, layer_index, layer_manager):
        pass
        # layer_dim_in = layer_manager.get_input_dim(layer_index=layer_index)
        # layer_dim_out = layer_manager.get_output_dim(layer_index=layer_index)
        #
        # self.self_attn.reconfigure(layer_index, layer_manager,layer_dim_in=layer_dim_in, layer_dim_out=layer_dim_in)
        # self.src_attn.reconfigure(layer_index, layer_manager,layer_dim_in=layer_dim_in, layer_dim_out=layer_dim_in)
        # self.feed_forward.reconfigure(layer_index, layer_manager,layer_dim_in=layer_dim_in, layer_dim_out=layer_dim_in)
        # self.sublayers[0].reconfigure(layer_index, layer_manager,layer_dim_in=layer_dim_in, layer_dim_out=layer_dim_in)
        # self.sublayers[1].reconfigure(layer_index, layer_manager,layer_dim_in=layer_dim_in, layer_dim_out=layer_dim_in)
        # self.sublayers[2].reconfigure(layer_index, layer_manager,layer_dim_in=layer_dim_in, layer_dim_out=layer_dim_in)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayers[2](x, self.feed_forward)

import copy

import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, N):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.N = N

    def reconfigure(self, layer_manager):
        self.encoder.reconfigure(layer_manager)
        self.decoder.reconfigure(layer_manager)
        self.src_embed[1].reconfigure(layer_manager, self.N)
        self.tgt_embed[1].reconfigure(layer_manager, self.N)

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def reconfigure(self, layer_index, layer_manager):
        # layer norm is applied to the output of the layer:
        features_size = layer_manager.get_output_dim(layer_index)
        self.a_2 = nn.Parameter(torch.ones(features_size))
        self.b_2 = nn.Parameter(torch.zeros(features_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(FunnelSublayerConnection(size, dropout), 2)
        self.output_size = size
        self.layer_index=-1
    def reconfigure(self, layer_index, layer_manager):
        self.self_attn.reconfigure(layer_index, layer_manager)
        self.feed_forward.reconfigure(layer_index, layer_manager)
        for sublayer in self.sublayer:
            sublayer.reconfigure(layer_index, layer_manager)
        self.layer_index =layer_index
        self.output_size = layer_manager.get_output_dim(layer_index)

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


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
        self.layer_index =-1

    def reconfigure(self, layer_index, layer_manager):
        d_model_in = layer_manager.get_input_dim(layer_index)
        d_model_out = layer_manager.get_output_dim(layer_index)
        self.linear = nn.Linear(d_model_in + d_model_out, d_model_out)
        self.d_model_in = d_model_in
        self.d_model_out = d_model_out
        self.layer_index=layer_index

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        residual = x
        layer_output_normalized = self.dropout(sublayer(self.norm(x)))
        if self.linear is None:  # or x.size(2)==layer_output_normalized.size(2):
            return x + layer_output_normalized
        else:
            return self.linear(torch.cat([residual.view(-1, self.d_model_in),
                                          layer_output_normalized.view(-1, self.d_model_out)], dim=1
                                         ))                               .view(x.size(0), x.size(1), self.d_model_out)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.N = N
        self.norms = clones(LayerNorm(layer.size), N)

    def reconfigure(self, layer_manager):
        for layer_index, layer in enumerate(self.layers):
            decoder_layer_index = self.N - 1 - layer_index
            layer.reconfigure(decoder_layer_index, layer_manager)

        for layer_index, norm in enumerate(self.norms):
            norm.reconfigure(layer_index, layer_manager)

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
        self.self_attn.reconfigure(layer_index, layer_manager)
        self.src_attn.reconfigure(layer_index, layer_manager)
        self.feed_forward.reconfigure(layer_index, layer_manager)
        for sublayer in self.sublayers:
            sublayer.reconfigure(layer_index, layer_manager)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayers[2](x, self.feed_forward)

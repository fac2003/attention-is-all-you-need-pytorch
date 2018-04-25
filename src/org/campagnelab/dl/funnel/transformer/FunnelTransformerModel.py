import copy

from src.org.campagnelab.dl.funnel.transformer.Attention import MultiHeadedAttention
from src.org.campagnelab.dl.funnel.transformer.Decoder import DecoderLayer, Decoder
from src.org.campagnelab.dl.funnel.transformer.Embedding import Embeddings
from src.org.campagnelab.dl.funnel.transformer.Encoder import EncoderLayer, Encoder
from src.org.campagnelab.dl.funnel.transformer.EncoderDecoder import EncoderDecoder
from src.org.campagnelab.dl.funnel.transformer.FeedForward import PositionwiseFeedForward
from src.org.campagnelab.dl.funnel.transformer.Generator import Generator
from src.org.campagnelab.dl.funnel.transformer.PositionalEncoding import PositionalEncoding, EmbeddingEncoding

import torch.nn as nn

from transformer.FunnelModels import DoubleEachLayerManager, ConstantDimLayerManager


def make_funnel_model(src_vocab, tgt_vocab, N=6,
                      d_model=512, d_ff=2048, h=8, dropout=0.1,
                      max_length=-1):
    "Helper: Construct a model from hyperparameters."
    layer_manager = DoubleEachLayerManager(constant_dimension=d_model, increase_factor=2)
    d_model=layer_manager.get_input_dim(layer_index=0)
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    initial_d_model = layer_manager.get_input_dim(layer_index=0)
    encoded_d_model = layer_manager.get_input_dim(layer_index=N)
    initial_d_ff = layer_manager.get_hidden_dim(layer_index=0)
    ff = PositionwiseFeedForward(d_model, initial_d_ff, dropout)
    enc_position = PositionalEncoding(initial_d_model, dropout, max_len=max_length)
    dec_position = PositionalEncoding(d_model, dropout, max_len=max_length)
    model = EncoderDecoder(
        Encoder(EncoderLayer(initial_d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(initial_d_model, src_vocab), c(enc_position)),
        nn.Sequential(Embeddings(initial_d_model, tgt_vocab), c(dec_position)),
        Generator(d_model, tgt_vocab), N=N)

    # layer_manager=ConstantDimLayerManager(d_model)
    model.reconfigure(layer_manager)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

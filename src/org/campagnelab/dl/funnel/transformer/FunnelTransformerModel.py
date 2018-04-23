
import copy

from src.org.campagnelab.dl.funnel.transformer.Attention import MultiHeadedAttention
from src.org.campagnelab.dl.funnel.transformer.Embedding import Embeddings
from src.org.campagnelab.dl.funnel.transformer.EncoderDecoder import EncoderDecoder, EncoderLayer, Encoder, Decoder, \
    DecoderLayer
from src.org.campagnelab.dl.funnel.transformer.FeedForward import PositionwiseFeedForward
from src.org.campagnelab.dl.funnel.transformer.Generator import Generator
from src.org.campagnelab.dl.funnel.transformer.PositionalEncoding import PositionalEncoding, EmbeddingEncoding

import torch.nn as nn

from transformer.FunnelModels import DoubleEachLayerManager, ConstantDimLayerManager


def make_funnel_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1,
               max_length=-1):
    "Helper: Construct a model from hyperparameters."
    layer_manager=DoubleEachLayerManager(constant_dimension=8, increase_factor=2)
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    initial_d_model=layer_manager.get_input_dim(layer_index=0)
    initial_d_ff=layer_manager.get_hidden_dim(layer_index=0)
    ff = PositionwiseFeedForward(initial_d_model, initial_d_ff, dropout)
    position = PositionalEncoding(initial_d_model, dropout,max_len=max_length)
    model = EncoderDecoder(
        Encoder(EncoderLayer(initial_d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(initial_d_model, c(attn), c(attn),                             c(ff), dropout), N),
        nn.Sequential(Embeddings(initial_d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(initial_d_model, tgt_vocab), c(position)),
        Generator(initial_d_model, tgt_vocab),N=N)

    #layer_manager=ConstantDimLayerManager(d_model)
    model.reconfigure(layer_manager)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
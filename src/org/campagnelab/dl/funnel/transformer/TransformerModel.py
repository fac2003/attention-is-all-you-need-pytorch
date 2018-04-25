import copy

from src.org.campagnelab.dl.funnel.transformer.Attention import MultiHeadedAttention
from src.org.campagnelab.dl.funnel.transformer.Decoder import Decoder, DecoderLayer
from src.org.campagnelab.dl.funnel.transformer.Embedding import Embeddings
from src.org.campagnelab.dl.funnel.transformer.Encoder import Encoder, EncoderLayer
from src.org.campagnelab.dl.funnel.transformer.EncoderDecoder import EncoderDecoder
from src.org.campagnelab.dl.funnel.transformer.FeedForward import PositionwiseFeedForward
from src.org.campagnelab.dl.funnel.transformer.Generator import Generator
from src.org.campagnelab.dl.funnel.transformer.PositionalEncoding import PositionalEncoding, EmbeddingEncoding

import torch.nn as nn

def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1,
               max_length=-1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout,max_len=max_length)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),N)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
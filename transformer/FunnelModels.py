''' Define the Funnel Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import transformer.Constants as Constants
from transformer.Modules import BottleLinear as Linear, LayerNormalization
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.PaddingBottleneck import PaddingBottleneck
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward

# Code adapted from Yu-Hsiang Huang Transfomer implementation.
__author__ = "Fabien Campagne"

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    #assert seq_q.dim() == 2 and seq_k.dim() == 2
    if seq_q.dim() == 3:
        # reduce the sequence to the first encoding dimension, and eliminate this dimension
        seq_q=seq_q[:,:,0].squeeze()
    if seq_k.dim()==3:
        seq_k=seq_k[:,:,0].squeeze()
        # then calculate the mask as before:
    mb_size, len_q = seq_q.size(0),seq_q.size(1)
    mb_size, len_k = seq_k.size(0),seq_k.size(1)
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask


class ProbabilisticSkipper(nn.Module):
    """Skip sequence time steps according to a probabilistic plan."""

    def __init__(self):
        super(ProbabilisticSkipper, self).__init__()
        self.noise = torch.Tensor()

    def forward(self, x, skipping_probs):
        batch_size = x.size(0)
        seq_length = x.size(1)
        encoding_dim = x.size(2)

        noise = torch.Tensor()
        noise.resize_as_(skipping_probs.data)
        noise.bernoulli_(skipping_probs.data)
        # following variable contains 1 in timestepts that will be kept, 0 otherwise:
        keeping_timesteps = Variable(1 - noise)
        list_of_example_sequences = x.split(dim=0, split_size=1)
        reduced_seqs = []
        max_length = 0
        for index, seq in enumerate(list_of_example_sequences):
            try:
                index_of_kept_timesteps = keeping_timesteps[index].squeeze().nonzero().squeeze()
                selected_time_steps = seq.index_select(dim=1, index=index_of_kept_timesteps)
                # selected_time_steps has dimension: 1 x seq_length x encoding_dim
                reduced_seqs += [selected_time_steps]
                #print(selected_time_steps)
                max_length = max(max_length, selected_time_steps.size(1))
            except RuntimeError as e:
                print(e)
            # reduced_seqs
        max_length, [seq[0].size() for seq in reduced_seqs]
        padded = []
        for seq in reduced_seqs:
            padded += [
                torch.nn.functional.pad(seq[0], pad=(0, 0, 0, max_length - seq[0].size(0)), mode='constant', value=0)]

        return torch.stack(padded, dim=0)



class MultiProbabilisticSkipper(nn.Module):
    """Skip sequence time steps according to a probabilistic plan."""

    def __init__(self,split_sizes=None):
        super(MultiProbabilisticSkipper, self).__init__()
        self.noise = torch.Tensor()
        self.split_sizes=split_sizes

    def forward(self, list_of_x, skipping_probs):
        batch_size = list_of_x[0].size(0)
        seq_length = list_of_x[0].size(1)
        if self.split_sizes is None:
            self.split_sizes=[1]*len(list_of_x)
        noise = torch.Tensor()
        noise.resize_as_(skipping_probs.data)
        noise.bernoulli_(skipping_probs.data)
        # following variable contains 1 in timestepts that will be kept, 0 otherwise:
        keeping_timesteps = Variable(1 - noise)
        list_of_results=[]
        for tensor_index, x in enumerate(list_of_x):
            list_of_example_sequences = x.split(dim=0, split_size=self.split_sizes[tensor_index])
            reduced_seqs = []
            max_length = 0
            for index, seq in enumerate(list_of_example_sequences):
                index_of_kept_timesteps = keeping_timesteps[index].squeeze().nonzero().squeeze()
                selected_time_steps = seq.index_select(dim=1, index=index_of_kept_timesteps)
                # selected_time_steps has dimension: 1 x seq_length x encoding_dim
                reduced_seqs += [selected_time_steps]
                #print(selected_time_steps)
                max_length = max(max_length, selected_time_steps.size(1))

                # reduced_seqs
            max_length, [seq[0].size() for seq in reduced_seqs]
            padded = []
            for seq in reduced_seqs:
                padded += [
                    torch.nn.functional.pad(seq[0], pad=(0, 0, 0, max_length - seq[0].size(0)), mode='constant', value=0)]

            list_of_results+=[ torch.stack(padded, dim=0)]
        return list_of_results


class SkippingProbability(nn.Module):
    def __init__(self, encoding_dim):
        super(SkippingProbability, self).__init__()
        self.encoding_dim = encoding_dim
        self.linear = nn.Linear(encoding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
        assert x.size(2) == self.encoding_dim, "encoding dimension must match between module and sequence batch."
        reshaped = x.view(-1, self.encoding_dim)
        probabilities = self.sigmoid(self.linear(reshaped))
        shaped_as_seq = probabilities.view(batch_size, seq_length, 1)
        return shaped_as_seq

class LayerManager():
    def get_input_dim(self,layer_index):
        return 0
    def get_output_dim(self,layer_index):
        return 0

class ConstantDimLayerManager():
    def __init__(self,constant_dimension):
        self.constant_dimension=constant_dimension
    def get_input_dim(self,layer_index):
        return self.constant_dimension
    def get_output_dim(self,layer_index):
        return self.constant_dimension


class PositionwiseFeedForwardFunnel(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_input, d_output, d_inner_hid, dropout=0.1):
        super(PositionwiseFeedForwardFunnel, self).__init__()
        self.w_1 = nn.Linear(d_input, d_inner_hid) # position-wise
        self.w_2 = nn.Linear(d_inner_hid, d_output) # position-wise
        self.layer_norm = LayerNormalization(d_output)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        batch_size=x.size(0)
        time_steps=x.size(1)
        encoding_dim=x.size(2)
        output = self.relu(self.w_1(x.view(-1,encoding_dim)))
        output = self.w_2(output)
        output = self.dropout(output)
        output=output.view(batch_size,time_steps,encoding_dim)
        return self.layer_norm(output + residual)


class FunnelEncoderLayer(nn.Module):
    ''' Compose with two layers and add probabilistic token skipping '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1,layer_index=-1,layer_manager=None):
        super(FunnelEncoderLayer, self).__init__()
        assert layer_manager is None or layer_index!=-1, "layer index must be defined"
        if layer_manager is None:
            layer_manager=ConstantDimLayerManager(d_model)
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardFunnel(layer_manager.get_input_dim(layer_index=layer_index),
                                                     layer_manager.get_output_dim(layer_index=layer_index),
                                                     d_inner_hid, dropout=dropout)
        self.prob_estimator=SkippingProbability(d_model)
        self.skipper=ProbabilisticSkipper()

    def forward(self, enc_input, slf_attn_mask=None):
        # estimate skipping probabilities from input, and sample some tokens:
        skip_probs = self.prob_estimator(enc_input)
        enc_input = self.skipper(enc_input, skip_probs)
        # recalculate mask now that sequence lengths have changed:
        slf_attn_mask=get_attn_padding_mask(enc_input, enc_input)
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):

        super(Encoder, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            FunnelEncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout, layer_index=layer_index)
                for layer_index in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):
        # Word embedding look up
        enc_input = self.src_word_emb(src_seq)

        # Position Encoding addition
        enc_input += self.position_enc(src_pos)
        if return_attns:
            enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.layer_stack:
            #print("before, enc_output.size={}".format(enc_output.size()))
            #print("before, enc_slf_attn.size={}".format(enc_slf_attn.size()))
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            #print("after, enc_output.size={}".format(enc_output.size()))
            #print("after, enc_slf_attn.size={}".format(enc_slf_attn.size()))
            if return_attns:
                enc_slf_attns += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):

        super(Decoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(
            n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
        # Word embedding look up
        dec_input = self.tgt_word_emb(tgt_seq)

        # Position Encoding addition
        dec_input += self.position_enc(tgt_pos)

        # Decode
        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        # use encoded sequence to determine length of source:
        dec_enc_attn_pad_mask = get_attn_padding_mask(tgt_seq, enc_output)

        if return_attns:
            dec_slf_attns, dec_enc_attns = [], []

        dec_output = dec_input
        for dec_layer in self.layer_stack:
            #print("dec_output.size: {}".format(dec_output.size()))
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask)

            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attns, dec_enc_attns
        else:
            return dec_output,

class FunnelTransformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism using
    a funnel transformer architecture. '''

    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, d_k=64, d_v=64,
            dropout=0.1, proj_share_weight=True, embs_share_weight=True):

        super(FunnelTransformer, self).__init__()
        self.encoder = Encoder(
            n_src_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout)
        self.decoder = Decoder(
            n_tgt_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout)
        self.tgt_word_proj = Linear(d_model, n_tgt_vocab, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.padding_bottleneck=PaddingBottleneck()
        # We will store the padding tensor here to find it after a call to forward:
        self.padding=None
        self.padding_amount=0
        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module output shall be the same.'

        if proj_share_weight:
            # Share the weight matrix between tgt word embedding/projection
            assert d_model == d_word_vec
            self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight

        if embs_share_weight:
            # Share the weight matrix between src/tgt word embeddings
            # assume the src/tgt word vec size are the same
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
        dec_freezed_param_ids = set(map(id, self.decoder.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, src, tgt):
        src_seq, src_pos = src
        tgt_seq, tgt_pos = tgt

        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_proj(dec_output)
        #max, max_token=torch.max(seq_logit,dim=2)
        #result= (max_token, enc_output)

        result= (seq_logit.view(dec_output.size(0), seq_logit.size(1),-1), enc_output)

        return result

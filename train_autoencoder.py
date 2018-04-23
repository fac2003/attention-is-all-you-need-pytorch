'''
This script handling the training process.
'''

import argparse
import math
import time

import sys
from torch.autograd import Variable
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import transformer.Constants as Constants
from transformer.FunnelModels import FunnelTransformer
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from DataLoader import DataLoader
from transformer.PaddingBottleneck import HardCompressiveBottleneck


def get_performance(crit, pred, gold, smoothing=False, num_class=None):
    ''' Apply label smoothing if needed '''

    # TODO: Add smoothing
    if smoothing:
        assert bool(num_class)
        eps = 0.1
        gold = gold * (1 - eps) + (1 - gold) * eps / num_class
        raise NotImplementedError
    num_words = pred.size(2)
    predicted_seq = pred.view(-1, num_words)
    gold_reshaped = gold.narrow(1, 0, pred.size(1)).contiguous().view(-1)
    loss = crit(predicted_seq, gold_reshaped)
    max_value, pred_tokens = torch.max(pred, dim=2)
    # pred = pred.max(1)[1]

    # gold = gold.contiguous().view(-1)
    n_correct = pred_tokens.data.eq(gold_reshaped.data)
    n_correct = n_correct.masked_select(gold_reshaped.ne(Constants.PAD).data).sum()

    return loss, n_correct


def train_epoch(model, training_data, crit, optimizer, opt):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0
    average_padding_amount = 0
    num_sequences = 0
    average_padding_factor = 0
    padding_indices = Variable(torch.LongTensor([Constants.PAD]), requires_grad=False)
    signal_indices = torch.LongTensor(range(1, opt.d_model))
    if opt.cuda:
        padding_indices = padding_indices.cuda()
        signal_indices = signal_indices.cuda()

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):
        # prepare data
        src, tgt = batch
        target = (Variable(src[0].data, requires_grad=False), Variable(src[1].data, requires_grad=False))

        # forward
        model.zero_grad()
        optimizer.zero_grad()
        # put source in gold:

        (pred, encoded_output) = model(src, target)
        if hasattr(model.padding_amount,"data"):
            average_padding_factor += model.padding_amount.data[0]
        # backward

        loss, n_correct = get_performance(crit, pred, target[0])

        # encourage the encoding with lots of padding (minimize sequence length):
        loss = loss - model.padding_amount * opt.sparsity
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate()

        # note keeping
        n_words = src[0].data.ne(Constants.PAD).sum()
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.data[0]
        num_sequences += 1

    return total_loss / n_total_words, n_total_correct / n_total_words, average_padding_amount / num_sequences, average_padding_factor / num_sequences


def eval_epoch(model, validation_data, crit, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0
    average_padding_factor = 0
    padding_min_logit = sys.maxsize
    padding_max_logit = -sys.maxsize
    num_batches = 0
    for batch in tqdm(
            validation_data, mininterval=2,
            desc='  - (Validation) ', leave=False):
        # prepare data
        src, tgt = batch
        target = (Variable(src[0].data, requires_grad=False), Variable(src[1].data, requires_grad=False))

        # forward
        (pred, encoded_output) = model(src, target)
        loss, n_correct = get_performance(crit, pred, target[0])
        if hasattr(model.padding_amount,"data"):
            padding_min_logit = min(torch.min(model.padding.data[0]), padding_min_logit)
            padding_max_logit = max(torch.max(model.padding.data[0]), padding_max_logit)
            average_padding_factor += model.padding_amount.data[0]
        # note keeping
        n_words = src[0].data.ne(Constants.PAD).sum()
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.data[0]
        num_batches += 1

    return total_loss / n_total_words, n_total_correct / n_total_words, padding_min_logit, padding_max_logit, average_padding_factor / num_batches


def train(model, training_data, validation_data, crit, optimizer, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None
    hard_compressor = HardCompressiveBottleneck(padding_amount=model.padding_amount,
                                                padding_value_threshold=opt.padding_value_threshold)
    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start_training = time.time()
        train_loss, train_accu, average_padding_amount, average_padding_factor = train_epoch(model, training_data, crit,
                                                                                             optimizer, opt)
        end_training = time.time()

        start_validation = time.time()
        valid_loss, valid_accu, padding_min, padding_max, average_padding_factor = eval_epoch(model, validation_data,
                                                                                              crit, opt)
        end_validation = time.time()
        print('\n  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, padding: {padding_amount:3.3f} \
        padding factor: {padding_factor:3.3e} ' \
              'elapsed: {elapse:3.3f} min'.format(
            ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu, padding_amount=average_padding_amount,
            padding_factor=average_padding_factor,
            elapse=(end_training - start_training) / 60))
        print(
            '\n  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, padding min: {padding_min:3.2f} max: {padding_max:3.2f}' \
            ' padding factor: {padding_factor:3.3e} elapsed: {elapse:3.3f} min'.format(
                ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu,
                padding_min=padding_min, padding_max=padding_max, padding_factor=average_padding_factor,
                elapse=(end_validation - start_validation) / 60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu))


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=10000)
    parser.add_argument('-batch_size', type=int, default=64)

    # parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=8)
    parser.add_argument('-d_inner_hid', type=int, default=16)
    parser.add_argument('-d_k', type=int, default=8)
    parser.add_argument('-d_v', type=int, default=8)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-max_size', type=int, default=0)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-lr', type=float, default=1E-3, help="Learning rate.")
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-sparsity', type=float, default=5.0)
    parser.add_argument('-padding_value_threshold', type=float, default=0.0)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-funnel', action='store_true', help="Use FunnelTransformer architecture.")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # ========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len


    # ========= Preparing DataLoader =========#
    max_size_src = opt.max_size if opt.max_size is not 0 else len(data['train']['src'])
    max_size_tgt = opt.max_size if opt.max_size is not 0 else len(data['train']['tgt'])
    print("training: max_size_src={} max_size_tgt={}".format(max_size_src,max_size_tgt))
    training_data = DataLoader(
        data['dict']['src'],
        data['dict']['tgt'],
        src_insts=data['train']['src'][0:max_size_src],
        tgt_insts=data['train']['tgt'][0:max_size_tgt],
        batch_size=opt.batch_size,
        cuda=opt.cuda)
    max_size_src = opt.max_size if opt.max_size is not 0 else len(data['valid']['src'])
    max_size_tgt = opt.max_size if opt.max_size is not 0 else len(data['valid']['tgt'])
    print("validation: max_size_src={} max_size_tgt={}".format(max_size_src, max_size_tgt))
    validation_data = DataLoader(
        data['dict']['src'],
        data['dict']['tgt'],
        src_insts=data['valid']['src'][0:max_size_src],
        tgt_insts=data['valid']['tgt'][0:max_size_tgt],
        batch_size=opt.batch_size,
        shuffle=opt.cuda,
        test=True,
        cuda=opt.cuda)

    opt.src_vocab_size = training_data.src_vocab_size
    opt.tgt_vocab_size = training_data.tgt_vocab_size

    # ========= Preparing Model =========#
    if opt.embs_share_weight and training_data.src_word2idx != training_data.tgt_word2idx:
        print('[Warning]',
              'The src/tgt word2idx table are different but asked to share word embedding.')

    print(opt)

    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        proj_share_weight=opt.proj_share_weight,
        embs_share_weight=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner_hid=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout) if not opt.funnel else FunnelTransformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        proj_share_weight=opt.proj_share_weight,
        embs_share_weight=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner_hid=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout)

    #print(transformer)

    optimizer = ScheduledOptim(
        optim.Adam(
            transformer.parameters(),
            betas=(0.9, 0.98), eps=1e-09, lr=opt.lr),
        opt.d_model, opt.n_warmup_steps)


    def get_criterion(vocab_size):
        ''' With PAD token zero weight '''
        weight = torch.ones(vocab_size)
        weight[Constants.PAD] = 0
        return nn.CrossEntropyLoss(weight, size_average=False)

    crit = get_criterion(training_data.tgt_vocab_size)

    if opt.cuda:
        transformer = transformer.cuda()
        crit = crit.cuda()

    train(transformer, training_data, validation_data, crit, optimizer, opt)


if __name__ == '__main__':
    main()

import unittest

import torch
from torch.autograd import Variable

from src.org.campagnelab.dl.funnel.transformer.Attention import MultiHeadedAttention
from src.org.campagnelab.dl.funnel.transformer.Encoder import EncoderLayer
from src.org.campagnelab.dl.funnel.transformer.FeedForward import PositionwiseFeedForward
from transformer.FunnelModels import ConstantDimLayerManager, DoubleEachLayerManager

dropout=0

class FunnelTransformerComponentsTestCase(unittest.TestCase):
    def test_SingleEncodingLayer0(self):
        layer_manager = ConstantDimLayerManager(constant_dimension=8)
        d_model = 8
        d_inner_hid = 16
        n_head = 8
        size=8

        self_attn=  MultiHeadedAttention(n_head, d_model)
        ff = PositionwiseFeedForward(d_model, d_inner_hid, dropout)


        torch.manual_seed(1212)
        encoder_input = Variable(torch.rand(10, 101, 8))
        attention_mask = Variable(torch.ones(10, 101, 101).type(torch.ByteTensor))

        encoder_layer = EncoderLayer(size,self_attn,feed_forward=ff,dropout=0)
        encoder_layer.reconfigure(0,layer_manager)
        result = encoder_layer.forward(encoder_input,attention_mask)
        self.assertEqual(result.size(2), 8)

    def test_TwoEncodingLayer_0_1(self):
        layer_manager = DoubleEachLayerManager(constant_dimension=8)
        d_model = 8
        d_inner_hid = 16
        n_head = 8
        size = 8

        self_attn = MultiHeadedAttention(n_head, d_model)
        ff = PositionwiseFeedForward(d_model, d_inner_hid, dropout)

        torch.manual_seed(1212)
        encoder_input = Variable(torch.rand(10, 101, 8))
        attention_mask = Variable(torch.ones(10, 101, 101).type(torch.ByteTensor))

        encoder_layer_0 = EncoderLayer(size, self_attn, feed_forward=ff, dropout=0)
        encoder_layer_0.reconfigure(0, layer_manager)

        encoder_layer_1 = EncoderLayer(size, self_attn, feed_forward=ff, dropout=0)
        encoder_layer_1.reconfigure(1, layer_manager)

        result_0 = encoder_layer_0.forward(encoder_input,attention_mask)
        self.assertEqual(result_0.size(2), 8)
        result_1 = encoder_layer_1.forward(result_0,attention_mask)
        self.assertEqual(result_1.size(2), 8)

if __name__ == '__main__':
    unittest.main()

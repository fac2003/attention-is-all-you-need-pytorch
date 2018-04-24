import unittest

import torch
from torch.autograd import Variable

from transformer.FunnelModels import ConstantDimLayerManager, FunnelDecoderLayer, DoubleEachLayerManager, \
    ConstantSequenceLengthManager, ProbabilisticExpander




class FunnelTransformerComponentsTestCase(unittest.TestCase):
    def test_SingleEncodingLayer0(self):
        torch.manual_seed(1212)
        slm = ConstantSequenceLengthManager(constant_length=101)
        expander = ProbabilisticExpander(layer_index=0, sequence_length_manager=slm)

        layer_manager = ConstantDimLayerManager(constant_dimension=8)
        d_model = 8
        d_inner_hid = 16
        n_head = 8
        d_k = 2
        d_v = 2


        encoder_input = Variable(torch.rand(10, 101, 8))
        previous_decoder_output = Variable(torch.rand(10, 101, 8))
        attention_mask = torch.ones(10, 101, 101).type(torch.ByteTensor)
        decoder_layer = FunnelDecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                           dropout=0, layer_index=0,
                                           layer_manager=layer_manager, is_first_decoder_layer=True,
                                           expander=None)

        dec_output, dec_slf_attn, dec_enc_attn = decoder_layer.forward(previous_decoder_output, encoder_input,
                                                                       slf_attn_mask=attention_mask)
        self.assertEqual(dec_output.size(2), 8)

    def test_SingleEncodingLayer1(self):
        layer_manager = ConstantDimLayerManager(constant_dimension=8)
        d_model = 8
        d_inner_hid = 16
        n_head = 8
        d_k = 2
        d_v = 2

        torch.manual_seed(1212)
        encoder_input = Variable(torch.rand(10, 101, 8))
        previous_decoder_output = Variable(torch.rand(10, 101, 8))
        attention_mask = torch.ones(10, 101, 101).type(torch.ByteTensor)
        decoder_layer = FunnelDecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                           dropout=0, layer_index=1,
                                           layer_manager=layer_manager, is_first_decoder_layer=True)

        dec_output, dec_slf_attn, dec_enc_attn = decoder_layer.forward(previous_decoder_output, encoder_input,
                                                                       slf_attn_mask=attention_mask)
        self.assertEqual(dec_output.size(2), 8)

    def test_DoublingSingleEncodingLayer0(self):
        """Increase dimension of encoding inside the layer."""
        layer_manager = DoubleEachLayerManager(constant_dimension=8)
        d_model = 8
        d_inner_hid = 16
        n_head = 8
        d_k = 2
        d_v = 2

        torch.manual_seed(1212)
        encoder_input = Variable(torch.rand(10, 101, 8))
        previous_decoder_output = Variable(torch.rand(10, 101, 8))
        attention_mask = torch.ones(10, 101, 101).type(torch.ByteTensor)

        decoder_layer = FunnelDecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                           dropout=0, layer_index=0,
                                           layer_manager=layer_manager, is_first_decoder_layer=True,
                                           expander=None)

        dec_output, dec_slf_attn, dec_enc_attn = decoder_layer.forward(previous_decoder_output, encoder_input,
                                                                       slf_attn_mask=attention_mask)
        self.assertEqual(dec_output.size(2), 16)

    def test_TwoEncodingLayer_0_1(self):
        layer_manager = ConstantDimLayerManager(constant_dimension=8)
        d_model = 8
        d_inner_hid = 16
        n_head = 8
        d_k = 2
        d_v = 2

        torch.manual_seed(1212)
        encoder_input = Variable(torch.rand(10, 101, 8))
        previous_decoder_output = Variable(torch.rand(10, 101, 8))
        attention_mask = torch.ones(10, 101, 101).type(torch.ByteTensor)

        decoder_layer_0 = FunnelDecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                             dropout=0, layer_index=0,
                                             layer_manager=layer_manager, is_first_decoder_layer=True)

        decoder_layer_1 = FunnelDecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                             dropout=0, layer_index=1,
                                             layer_manager=layer_manager, is_first_decoder_layer=False)

        dec_output, dec_slf_attn, dec_enc_attn = decoder_layer_0.forward(previous_decoder_output, encoder_input,
                                                                         slf_attn_mask=attention_mask,
                                                                         dec_enc_attn_mask=attention_mask)
        self.assertEqual(dec_output.size(2), 8)
        previous_decoder_output = dec_output
        dec_output, dec_slf_attn, dec_enc_attn = decoder_layer_1.forward(previous_decoder_output, encoder_input,
                                                                         slf_attn_mask=attention_mask,
                                                                         dec_enc_attn_mask=attention_mask)
        self.assertEqual(dec_output.size(2), 8)

    def test_DoublingTwoEncodingLayer_0_1(self):
        layer_manager = DoubleEachLayerManager(constant_dimension=8)
        d_model = 8
        d_inner_hid = 16
        n_head = 8
        d_k = 2
        d_v = 2

        torch.manual_seed(1212)
        encoder_input = Variable(torch.rand(10, 101, 8))
        previous_decoder_output = Variable(torch.rand(10, 101, 8))
        attention_mask = torch.ones(10, 101, 101).type(torch.ByteTensor)

        decoder_layer_0 = FunnelDecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                             dropout=0, layer_index=0,
                                             layer_manager=layer_manager, is_first_decoder_layer=True)

        decoder_layer_1 = FunnelDecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                             dropout=0, layer_index=1,
                                             layer_manager=layer_manager, is_first_decoder_layer=False)

        dec_output, dec_slf_attn, dec_enc_attn = decoder_layer_0.forward(previous_decoder_output, encoder_input,
                                                                         slf_attn_mask=attention_mask,
                                                                         dec_enc_attn_mask=attention_mask)
        self.assertEqual(dec_output.size(2), 16)
        previous_decoder_output = dec_output
        dec_output, dec_slf_attn, dec_enc_attn = decoder_layer_1.forward(previous_decoder_output, encoder_input,
                                                                         slf_attn_mask=attention_mask,
                                                                         dec_enc_attn_mask=attention_mask)
        self.assertEqual(dec_output.size(2), 32)


if __name__ == '__main__':
    unittest.main()

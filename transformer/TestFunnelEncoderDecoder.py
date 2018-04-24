import unittest

import torch
from torch.autograd import Variable

from transformer.FunnelModels import ConstantDimLayerManager, FunnelEncoderLayer, FunnelDecoderLayer, \
    DoubleEachLayerManager, DecoderLayerManager, HalfSequenceSkipper


class MyTestCase(unittest.TestCase):
    def test_ConstantDimEncodingDecodingLayers(self):
        layer_manager = ConstantDimLayerManager(constant_dimension=8)
        d_model = 8
        d_inner_hid = 16
        n_head = 8
        d_k = 2
        d_v = 2

        torch.manual_seed(1212)
        encoder_input = Variable(torch.rand(10, 101, 8))
        attention_mask = torch.ones(10, 101, 101).type(torch.ByteTensor)

        encoder_layer = FunnelEncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                           dropout=0, layer_index=1,
                                           layer_manager=layer_manager)

        decoder_layer = FunnelDecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                           dropout=0, layer_index=1,
                                           layer_manager=layer_manager,
                                           is_first_decoder_layer=True)

        result, attention_result = encoder_layer.forward(encoder_input)
        previous_decoder_output=encoder_input
        dec_output, dec_slf_attn, dec_enc_attn = decoder_layer.forward(previous_decoder_output, encoder_input,
                                                                       slf_attn_mask=attention_mask)
        self.assertEqual(dec_output.size(2), 8)
        self.assertEqual(result.size(2), 8)

    def test_VariableDimEncodingDecodingLayers(self):
        layer_manager = DoubleEachLayerManager(constant_dimension=8)
        d_model = 8
        d_inner_hid = 16
        n_head = 8
        d_k = 2
        d_v = 2

        torch.manual_seed(1212)
        encoder_input = Variable(torch.rand(10, 101, 16))
        attention_mask = torch.ones(10, 101, 101).type(torch.ByteTensor)
        encoded_attention_mask = torch.ones(10, 50, 50).type(torch.ByteTensor)

        encoder_layer = FunnelEncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                           dropout=0, layer_index=1,
                                           layer_manager=layer_manager,skipper=HalfSequenceSkipper())
        layer_manager=DecoderLayerManager(layer_manager,1)
        decoder_layer = FunnelDecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                           dropout=0, layer_index=1,
                                           layer_manager=layer_manager,
                                           is_first_decoder_layer=True)

        result, attention_result = encoder_layer.forward(encoder_input)
        self.assertEqual(result.size(2), 32)
        previous_decoder_output=result
        dec_output, dec_slf_attn, dec_enc_attn = decoder_layer.forward(previous_decoder_output, previous_decoder_output,
                                                                       slf_attn_mask=encoded_attention_mask)
        self.assertEqual(dec_output.size(2), 16)


if __name__ == '__main__':
    unittest.main()

import unittest

import torch
from torch.autograd import Variable

from transformer.FunnelModels import ConstantDimLayerManager, FunnelEncoderLayer, DoubleEachLayerManager


class FunnelTransformerComponentsTestCase(unittest.TestCase):
    def test_SingleEncodingLayer0(self):
        layer_manager = ConstantDimLayerManager(constant_dimension=8)
        d_model = 8
        d_inner_hid = 16
        n_head = 8
        d_k = 2
        d_v = 2

        torch.manual_seed(1212)
        encoder_input = Variable(torch.rand(10, 101, 8))

        encoder_layer = FunnelEncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                           dropout=0, layer_index=0,
                                           layer_manager=layer_manager)

        result, attention_result = encoder_layer.forward(encoder_input)
        print(attention_result.size())
        self.assertEqual(result.size(2), 8)

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

        encoder_layer = FunnelEncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                           dropout=0, layer_index=0,
                                           layer_manager=layer_manager)

        result, attention_result = encoder_layer.forward(encoder_input)
        self.assertEqual(result.size(2), 16)

    def test_SingleEncodingLayer1(self):
        layer_manager = ConstantDimLayerManager(constant_dimension=8)
        d_model = 8
        d_inner_hid = 16
        n_head = 8
        d_k = 2
        d_v = 2

        torch.manual_seed(1212)
        encoder_input = Variable(torch.rand(10, 101, 8))

        encoder_layer = FunnelEncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                           dropout=0, layer_index=1,
                                           layer_manager=layer_manager)

        result, attention_result = encoder_layer.forward(encoder_input)
        self.assertEqual(result.size(2), 8)

    def test_TwoEncodingLayer_0_1(self):
        layer_manager = ConstantDimLayerManager(constant_dimension=8)
        d_model = 8
        d_inner_hid = 16
        n_head = 8
        d_k = 2
        d_v = 2

        torch.manual_seed(1212)
        encoder_input = Variable(torch.rand(10, 101, 8))

        encoder_layer_0 = FunnelEncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                           dropout=0, layer_index=0,
                                           layer_manager=layer_manager)


        encoder_layer_1 = FunnelEncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                           dropout=0, layer_index=1,
                                           layer_manager=layer_manager)

        result, attention_result = encoder_layer_0.forward(encoder_input)
        self.assertEqual(result.size(2), 8)
        result, attention_result = encoder_layer_1.forward(result)
        self.assertEqual(result.size(2), 8)

    def test_DoublingTwoEncodingLayer_0_1(self):
        layer_manager = DoubleEachLayerManager(constant_dimension=8)
        d_model = 8
        d_inner_hid = 16
        n_head = 8
        d_k = 2
        d_v = 2

        torch.manual_seed(1212)
        encoder_input = Variable(torch.rand(10, 101, 8))

        encoder_layer_0 = FunnelEncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                           dropout=0, layer_index=0,
                                           layer_manager=layer_manager)


        encoder_layer_1 = FunnelEncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                           dropout=0, layer_index=1,
                                           layer_manager=layer_manager)

        result, attention_result = encoder_layer_0.forward(encoder_input)
        self.assertEqual(result.size(2), 16)
        result, attention_result = encoder_layer_1.forward(result)
        self.assertEqual(result.size(2), 32)


if __name__ == '__main__':
    unittest.main()

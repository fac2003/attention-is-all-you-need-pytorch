import unittest

from transformer.FunnelModels import ConstantDimLayerManager, DoubleEachLayerManager, DecoderLayerManager


class LayerManagerTestCase(unittest.TestCase):
    def test_constant(self):
        layer_manager = ConstantDimLayerManager(constant_dimension=8)

        self.assertEqual(layer_manager.get_input_dim(0), 8)
        self.assertEqual(layer_manager.get_output_dim(0), 8)
        self.assertEqual(layer_manager.get_input_dim(1), 8)
        self.assertEqual(layer_manager.get_output_dim(1), 8)


    def test_doubling(self):
        layer_manager = DoubleEachLayerManager(constant_dimension=8, increase_factor=2)

        self.assertEqual(layer_manager.get_input_dim(0), 8)
        self.assertEqual(layer_manager.get_output_dim(0), 16)
        self.assertEqual(layer_manager.get_input_dim(1), 16)
        self.assertEqual(layer_manager.get_output_dim(1), 32)

    def test_decoding_constant(self):
        layer_manager = ConstantDimLayerManager(constant_dimension=8)
        layer_manager=DecoderLayerManager(layer_manager,2)
        self.assertEqual(layer_manager.get_input_dim(0), 8)
        self.assertEqual(layer_manager.get_output_dim(0), 8)
        self.assertEqual(layer_manager.get_input_dim(1), 8)
        self.assertEqual(layer_manager.get_output_dim(1), 8)


    def test_decoding_double(self):
        layer_manager = DoubleEachLayerManager(constant_dimension=8)
        layer_manager=DecoderLayerManager(layer_manager,2)
        # [0] 8 16, [1] 16 32, [1] 32 16 [0] 16 8
        self.assertEqual(layer_manager.get_input_dim(1), 32)
        self.assertEqual(layer_manager.get_output_dim(1), 16)
        self.assertEqual(layer_manager.get_input_dim(0), 16)
        self.assertEqual(layer_manager.get_output_dim(0), 8)




if __name__ == '__main__':
    unittest.main()

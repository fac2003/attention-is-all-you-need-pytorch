import unittest

from src.org.campagnelab.dl.funnel.transformer.Attention import MultiHeadedAttention
from src.org.campagnelab.dl.funnel.transformer.EncoderDecoder import Encoder, EncoderLayer
from src.org.campagnelab.dl.funnel.transformer.FeedForward import PositionwiseFeedForward
from transformer.FunnelModels import DoubleEachLayerManager, ConstantDimLayerManager

layer_manager=DoubleEachLayerManager(constant_dimension=16)
#layer_manager=ConstantDimLayerManager(constant_dimension=16)
class EncoderReconfigureTestCase(unittest.TestCase):
    def test_encoder_reconfigure(self):
        dropout=0
        size = 8
        n_heads=8
        self_attn= MultiHeadedAttention(n_heads, size)

        initial_d_model = 8
        initial_d_ff = 16
        feed_forward = PositionwiseFeedForward(initial_d_model, initial_d_ff, dropout)
        encoder=Encoder(EncoderLayer(size, self_attn, feed_forward, dropout),1)
        #print(encoder)
        encoder.reconfigure(layer_manager)
        print(encoder)
        # check that the LayerConnections have been reconfigured correctly:
        self.assertEqual(encoder.layers[0].sublayer[0].linear.in_features, 48)
        self.assertEqual(encoder.layers[0].sublayer[0].linear.out_features, 32)
        self.assertEqual(encoder.layers[0].sublayer[1].linear.in_features, 64)
        self.assertEqual(encoder.layers[0].sublayer[1].linear.out_features, 32)


if __name__ == '__main__':
    unittest.main()

import unittest

import torch
from torch.autograd import Variable

from transformer.FunnelModels import ProbabilisticExpander, ConstantSequenceLengthManager


class ExpanderTestCase(unittest.TestCase):
    def test_expand(self):
        slm=ConstantSequenceLengthManager(constant_length=16)
        expander=ProbabilisticExpander(layer_index=0,sequence_length_manager=slm)
        bs=10
        input_length=13
        encoding_dim=3
        expand_probs=Variable(torch.rand(bs,input_length, 1))
        x=Variable(torch.rand(bs,input_length, encoding_dim))
        expanded=expander.forward(x,expanding_probs=expand_probs)
        self.assertNotEqual(expanded.size(1),x.size(1))


if __name__ == '__main__':
    unittest.main()

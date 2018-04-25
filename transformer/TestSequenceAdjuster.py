import unittest

import torch
from torch.autograd import Variable
from torch.nn import MSELoss

from transformer.SequenceAdjusters import SequenceLengthAdjuster


class SequenceAdjusterTestCase(unittest.TestCase):
    def test_adjust(self):
        adjuster=SequenceLengthAdjuster()
        sequences=Variable(torch.rand(10,101,8), requires_grad=True)
        output_length=51
        output_encoding_dim=16
        result=adjuster.adjust(sequences,  output_length, output_encoding_dim)
        self.assertEqual(result.size(2), 16)
        self.assertEqual(result.size(1), 51)
        reconstructed=adjuster.adjust(result,  sequences.size(1), sequences.size(2))
        self.assertEqual(reconstructed.size(2), 8)
        self.assertEqual(reconstructed.size(1), 101)
        mseloss=MSELoss()
        for epoch in range(10):
            result = adjuster.adjust(sequences, output_length, output_encoding_dim)
            reconstructed = adjuster.adjust(result, sequences.size(1), sequences.size(2))
            loss=mseloss(reconstructed,Variable(sequences.data))
            print("mse loss: {} ".format(loss.data[0]))

if __name__ == '__main__':
    unittest.main()

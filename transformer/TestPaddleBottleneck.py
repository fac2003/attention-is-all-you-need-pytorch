import unittest

import torch
from torch.autograd import Variable

from transformer.PaddingBottleneck import PaddingBottleneck, HardCompressiveBottleneck


def build_k():
    k = torch.ones(2, 5, 3)

    k[0, :, 0:1] = torch.Tensor([[-10, -5, -1, -0.5, 0]])
    k[0, :, 1:3] = 1

    k[1, :, 0:1] = torch.Tensor([[0, 5, 10, 20, 30]])
    k[1, :, 1:3] = 2

    print(k)
    return k


class SequenceCompressionTestCase(unittest.TestCase):
    def test_padding_bottleneck(self):
        # 5 timesteps, 3 encoding dimensions:
        k = build_k()

        padder = PaddingBottleneck()
        v = Variable(k, requires_grad=False)
        result = padder(v)
        expected_soft_padding = torch.FloatTensor([
            [
                [-10.0000, 1.0000, 1.0000],
                [- 5.0000, 0.9991, 0.9991],
                [- 1.0000, 0.9526, 0.9526],
                [- 0.5000, 0.9241, 0.9241],
                [0.0000, 0.8808, 0.8808]],
            [
                [0.0000, 1.9640, 1.9640],
                [5.0000, 0.5379, 0.5379],
                [10.0000, 0.0049, 0.0049],
                [20.0000, 0.0000, 0.0000],
                [30.0000, 0.0000, 0.0000]
            ]

        ])
        print("soft padding:" + str(padder(v)))
        self.assertEqual(str(expected_soft_padding), str(result.data))

    def test_hard_padding_bottleneck(self):
        k = build_k()
        padder = HardCompressiveBottleneck(padding_amount=Variable(torch.FloatTensor([0.0])),
                                           padding_value_threshold=11)
        v = Variable(k, requires_grad=False)
        result = padder(v)
        print("hard padding: "+str(result))
        expected_hard_padded = torch.zeros(2, 5, 3)
        expected_hard_padded[0, :, :] = 1
        expected_hard_padded[0, :, 0:1] = torch.FloatTensor([[-10, -5, -1, -0.5, 0]])
        expected_hard_padded[1, :, 0:1] = torch.FloatTensor([[0, 5, 10, 20, 30]])
        expected_hard_padded[1, 0:3, 1:6] = 2
        expected_hard_padded[1, 3:5, 1:6] = 0
        expected_hard_padded[1, 3:5, 0] = 11
        self.assertEqual(str(expected_hard_padded), str(result))


if __name__ == '__main__':
    unittest.main()

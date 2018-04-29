import unittest

import torch
from torch.autograd import Variable

from src.org.campagnelab.dl.CheckGradients import register_hooks
from src.org.campagnelab.dl.funnel.transformer.SequenceCompressors import SequenceCompressor


class SequenceCompressorTestCase(unittest.TestCase):
    def test_gradient_flow1(self):
        x = Variable(torch.FloatTensor([[1, 2], [3, 4]]), requires_grad=True)

        x2=torch.gather(x, 1, Variable(torch.LongTensor([[0, 0], [1, 0]]),requires_grad=False))
        y = Variable(torch.rand(3, x2.size(1), 32), requires_grad=True)
        loss = torch.mean(x2) - torch.mean(y)
        get_dot = register_hooks(loss)
        loss.backward()
        dot = get_dot()
        dot.save('tmp.dot')
        self.assertTrue(x.grad is not None)

    def test_gradient_flow2(self):

        x=Variable(torch.rand(3,10,32),requires_grad=True)
        compressor=SequenceCompressor(encoding_dim=x.size(2))
        #index_of_kept_timesteps=Variable(torch.LongTensor([0,1]*16),requires_grad=False)

        #x2=x.index_select(dim=1, index=index_of_kept_timesteps)

        x2,mask=compressor(x)
        y=Variable(torch.rand(3,x2.size(1),32),requires_grad=True)
        loss=torch.mean(x2)-torch.mean(y)
        get_dot = register_hooks(loss)
        loss.backward()
        dot = get_dot()
        dot.save('tmp.dot')
        self.assertTrue( x.grad is not None)


if __name__ == '__main__':
    unittest.main()

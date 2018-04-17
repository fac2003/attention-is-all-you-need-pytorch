import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter


def separate_signal_from_padding(x, padding_indices, signal_indices):
    padding = torch.index_select(x, 1, padding_indices)
    signal = torch.index_select(x, 1, signal_indices)
    # torch.split(torch.sum(signal,dim=1),dim=0)
    signal_norm = torch.norm(signal, dim=1, p=1)
    # print("summed: {}".format(signal_summed))
    signal_norm_split = signal_norm.split(split_size=1, dim=0)
    padding_split = padding.split(split_size=1, dim=0)
    return signal_norm_split, padding_split, padding, signal


def interleave(signal_norm_split, padding_split,batch_size, time_steps):
    rows = []
    for i in range(0, len(padding_split)):
        row = torch.stack([padding_split[i].contiguous().view(time_steps, 1), signal_norm_split[i].contiguous().view(time_steps, 1)], dim=1)

        rows += [row]
    # rows
    # print("stacked: {} ".format(torch.stack(rows, dim=1)))
    interleaved = torch.stack(rows, dim=0).view(batch_size*time_steps, 2)
    return interleaved


def calculate_softmax_padding_weights(interleaved):
    weights = F.softmax(interleaved, dim=1).narrow(dimension=1, start=0, length=1)
    one_minus_weights = 1.0 - weights
    return weights, one_minus_weights


class PaddingBottleneck(torch.nn.Module):
    """Apply a bottleneck on an encoding tensor. The bottleneck considers the first
    element of encoding as a padding element, which determines how much the other
     elements of encoding are muted. We use a softmax to soft-mute the signal part
     of the encoding when the padding element has a large value."""
    def __init__(self):
        super(PaddingBottleneck, self).__init__()
        self.padding_indices = Variable(torch.LongTensor([0]), requires_grad=True)

        self.signal_indices = None
        self.padding_amount=Parameter(torch.Tensor([0.0]))
        self.padding = None

    def forward(self, x):
        x=x.view(x.size(0),x.size(2),x.size(1))
        batch_size = x.size(0)
        encoding_dimension = x.size(1)
        time_steps = x.size(2)
        self.signal_indices = Variable(torch.LongTensor(range(1, encoding_dimension)), requires_grad=True)
        if x.is_cuda:
                self.signal_indices=self.signal_indices.cuda()
                self.padding_indices=self.padding_indices.cuda()

        signal_norm_split, padding_split, \
        padding, signal = separate_signal_from_padding(x,
                                                       self.padding_indices,
                                                       self.signal_indices)

        # the padding_amount parameter makes it possible to increase, but not decrease the amount of padding.
        padding = padding * (1.0 + torch.abs(self.padding_amount))
        self.padding = padding
        interleaved = interleave(signal_norm_split, padding_split, batch_size=batch_size, time_steps=time_steps)
        weights, one_minus_weights = calculate_softmax_padding_weights(interleaved)
        signal_weighted = one_minus_weights.view(batch_size,1,time_steps).repeat(1,encoding_dimension-1,1)*signal
        result= torch.cat([padding, signal_weighted], dim=1)
        return result.view(result.size(0),result.size(2),result.size(1))


# k1=torch.ones(4,5)
#
#
# k1[0]=(torch.Tensor([-10,-5,-1,-0.5,0]))
# k2=k1.clone()
#
# k2[0]=(torch.Tensor([1,5,10,20,30]))
# k2[1]=torch.ones(1,5)*2
# k2[2]=torch.ones(1,5)*2
# k2[3]=torch.ones(1,5)*2
# k=torch.stack([k1,k2])
# print(k)
#
# padder=PaddingBottleneck()
# v=Variable(k,requires_grad=False)
# print(padder( v))
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter


def separate_signal_from_padding(x, padding_indices, signal_indices,batch_size, encoding_dimension):
    padding = torch.index_select(x, 2, padding_indices)
    signal = torch.index_select(x, 2, signal_indices)
    # torch.split(torch.sum(signal,dim=1),dim=0)
    signal_norm = torch.norm(signal, dim=2, p=1)
    # print("summed: {}".format(signal_summed))
    signal_norm_split = signal_norm.split(split_size=1, dim=0)
    padding_split = padding.split(split_size=1, dim=0)
    return signal_norm_split, padding_split, padding, signal


def interleave(signal_norm_split, padding_split,batch_size, encoding_dimension, time_steps):
    rows = []
    for i in range(0, len(padding_split)):
        padding_to_stack = padding_split[i].contiguous().view(time_steps, 1)
        signal_to_stack = signal_norm_split[i].contiguous().view(time_steps, 1)
        row = torch.stack([padding_to_stack, signal_to_stack], dim=1)

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

        batch_size = x.size(0)
        time_steps = x.size(1)
        encoding_dimension = x.size(2)
        self.signal_indices = Variable(torch.LongTensor(range(1, encoding_dimension)), requires_grad=True)
        if x.is_cuda:
                self.signal_indices=self.signal_indices.cuda()
                self.padding_indices=self.padding_indices.cuda()

        signal_norm_split, padding_split, \
        padding, signal = separate_signal_from_padding(x,batch_size=batch_size, encoding_dimension=encoding_dimension,
                                                       padding_indices=self.padding_indices,
                                                       signal_indices=self.signal_indices)

        # the padding_amount parameter makes it possible to increase, but not decrease the amount of padding.
        padding = padding * (1.0 + torch.abs(self.padding_amount))
        self.padding = padding
        interleaved = interleave(signal_norm_split, padding_split, batch_size=batch_size, encoding_dimension=encoding_dimension, time_steps=time_steps)
        weights, one_minus_weights = calculate_softmax_padding_weights(interleaved)
        signal_weighted = one_minus_weights.view(batch_size,time_steps,1).repeat(1,1,encoding_dimension-1)*signal
        result= torch.cat([padding, signal_weighted], dim=2)
        return result


class HardCompressiveBottleneck(torch.nn.Module):
    """Apply a bottleneck on an encoding tensor. The bottleneck considers the first
    element of encoding as a padding element and removes any time step whose padding value
    is larger than a threshold."""
    def __init__(self, padding_amount,padding_value_threshold=0):
        super(HardCompressiveBottleneck, self).__init__()
        self.padding_indices = Variable(torch.LongTensor([0]), requires_grad=True)

        self.signal_indices = None
        self.padding_amount=padding_amount
        self.padding = None
        self.threshold=padding_value_threshold

    def forward(self, x):
        batch_size=x.size(0)
        time_steps = x.size(1)
        encoding_dimension = x.size(2)
        self.signal_indices = Variable(torch.LongTensor(range(1, encoding_dimension)), requires_grad=True)
        if x.is_cuda:
                self.signal_indices=self.signal_indices.cuda()
                self.padding_indices=self.padding_indices.cuda()

        signal_norm_split, padding_split, \
        padding, signal = separate_signal_from_padding(x,batch_size=batch_size,
                                                       encoding_dimension=encoding_dimension,
                                                       padding_indices=self.padding_indices,
                                                       signal_indices=self.signal_indices)
        # the padding_amount parameter makes it possible to increase, but not decrease the amount of padding.
        padding = padding * (1.0 + torch.abs(self.padding_amount))
        example_lengths=[0]*batch_size

        for example_index in range(0,batch_size):
            for time_step in range(0,time_steps):
                if padding[example_index, time_step,0].data[0]<self.threshold:
                    example_lengths[example_index]+=1

        clipped_length=max(example_lengths)
        result=torch.zeros(batch_size, clipped_length,encoding_dimension)
        position = [0] * batch_size
        result[:, :, 0] = self.threshold
        for example_index in range(0,batch_size):
            for time_step in range(0,clipped_length):
                if padding[example_index, time_step,0].data[0]<self.threshold:
                    result[example_index, position[example_index],0]=padding[example_index, time_step,0].data[0]
                    result[example_index, position[example_index],1:]=signal[example_index, time_step,:].data
                    position[example_index]+=1
                else:
                    if position[example_index]<clipped_length:
                        result[example_index, position[example_index], 0] =self.threshold

        return result


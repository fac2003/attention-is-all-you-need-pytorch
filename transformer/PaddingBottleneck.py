import torch
import torch.nn.functional as F
from torch.autograd import Variable


def separate_signal_from_padding(x, padding_indices, signal_indices):
    padding = torch.index_select(x, 1, padding_indices)
    signal = torch.index_select(x, 1, signal_indices)
    # torch.split(torch.sum(signal,dim=1),dim=0)
    signal_norm = torch.norm(signal, dim=1, p=1)
    # print("summed: {}".format(signal_summed))
    signal_norm_split = signal_norm.split(split_size=1, dim=0)
    padding_split = padding.split(split_size=1, dim=0)
    return signal_norm_split, padding_split, padding, signal


def interleave(signal_norm_split, padding_split):
    rows = []
    for i in range(0, len(padding_split)):
        row = torch.stack([padding_split[i].view(3, 1), signal_norm_split[i].view(3, 1)], dim=1)

        rows += [row]
    # rows
    # print("stacked: {} ".format(torch.stack(rows, dim=1)))
    interleaved = torch.stack(rows, dim=0).view(6, 2)
    return interleaved


def calculate_softmax_padding_weights(interleaved):
    weights = F.softmax(interleaved, dim=1).narrow(dimension=1, start=0, length=1)
    weights
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

    def forward(self, x):
        if self.signal_indices is None:
            self.signal_indices = Variable(torch.LongTensor(range(1, x.size(2) + 1)), requires_grad=True)
        batch_size = x.size(0)
        encoding_dimension = x.size(1)
        time_steps = x.size(2)
        num_softmax_elements = time_steps * batch_size
        signal_norm_split, padding_split, \
        padding, signal = separate_signal_from_padding(x,
                                                       self.padding_indices,
                                                       self.signal_indices)

        interleaved = interleave(signal_norm_split, padding_split)
        weights, one_minus_weights = calculate_softmax_padding_weights(interleaved)
        padding_weighted = (weights * padding.view(num_softmax_elements, -1)).view(batch_size, 1, time_steps)
        signal_weighted = (one_minus_weights * signal.view(num_softmax_elements, -1)).view(batch_size, -1, time_steps)

        return torch.cat([padding_weighted, signal_weighted], dim=1)

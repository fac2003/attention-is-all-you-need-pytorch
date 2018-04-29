import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter


class SkippingProbability(nn.Module):
    """Estimate skip probabilities from encoded sequence elements. """

    def __init__(self, encoding_dim):
        super(SkippingProbability, self).__init__()
        self.encoding_dim = encoding_dim
        self.linear = nn.Linear(encoding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
        assert x.size(2) == self.encoding_dim, "encoding dimension must match between module and sequence batch."
        reshaped = x.view(-1, self.encoding_dim)
        probabilities = self.sigmoid(self.linear(reshaped))
        shaped_as_seq = probabilities.view(batch_size, seq_length, 1)

        return shaped_as_seq


class ProbabilisticSkipper(nn.Module):
    """Skip sequence time steps according to a probabilistic plan."""

    def __init__(self):
        super(ProbabilisticSkipper, self).__init__()
        self.noise = torch.Tensor()

    def forward(self, x, skipping_probs):
        """
        :param x: encoding to skip
        :param skipping_probs: tensor with one probability per time-step. The probability that
        the token at time step will be skipped.
        :return: encoding with reduced length.
        """
        batch_size = x.size(0)
        seq_length = x.size(1)
        encoding_dim = x.size(2)
        input = x
        noise = torch.Tensor()
        # if x.is_cuda:
        #    noise = noise.cuda()

        noise.resize_as_(skipping_probs.data)
        # print("torch.min(skipping_probs).data[0]: {} max: {}".format(torch.min(skipping_probs).data[0],
        #                                                             torch.max(skipping_probs).data[0]      ))
        noise.bernoulli_(skipping_probs.data)
        # following variable contains 1 in timesteps that will be kept, 0 otherwise:
        keeping_timesteps = Variable(1 - noise, requires_grad=False)

        list_of_example_sequences = x.split(dim=0, split_size=1)
        reduced_seqs = []
        reduced_masks = []
        max_length = 0
        for index, seq in enumerate(list_of_example_sequences):
            index_of_kept_timesteps = keeping_timesteps[index].squeeze().nonzero().squeeze()

            if len(index_of_kept_timesteps   )>0:
              selected_time_steps = seq.index_select(dim=1, index=index_of_kept_timesteps)
            else:
                selected_time_steps=seq
            # selected_time_steps has dimension: 1 x seq_length x encoding_dim
            reduced_seqs += [selected_time_steps]
            # print(selected_time_steps)
            max_length = max(max_length, selected_time_steps.size(1))

        max_length, [seq[0].size() for seq in reduced_seqs]
        reduced_masks = [Variable(torch.ones(1, 1, size), requires_grad=True) for size in
                         [seq[0].size(0) for seq in reduced_seqs]]
        padded = []
        padded_masks = []
        for seq in reduced_seqs:
            padded += [
                torch.nn.functional.pad(seq[0], pad=(0, 0, 0, max_length - seq[0].size(0)), mode='constant', value=0)]
        for mask in reduced_masks:
            padded_masks += [
                torch.nn.functional.pad(mask[0], pad=(0, max_length - mask[0].size(1)), mode='constant', value=0)]

        return torch.stack(padded, dim=0), torch.stack(padded_masks, dim=0)


class SequenceCompressor(nn.Module):

    def __init__(self, encoding_dim, param_is_variable=False):
        super().__init__()
        self.probs_estimator = SkippingProbability(encoding_dim)
        self.skipper = ProbabilisticSkipper()
        compression_rate = torch.FloatTensor([0.0])
        if param_is_variable:
            self.register_parameter(name="compression_rate", param=Parameter(compression_rate))
        else:
            self.compression_rate = Variable(compression_rate, requires_grad=True)

        # assert self.compression_rate.data[0] >= 0 and self.compression_rate.data[
        #    0] <= 1, "Compression rate must be between 0 and 1."

    def replace_with(self,x, condition, value):

        if len(condition)>0:

            x[condition] =value

    def forward(self, x):

        skipping_probs = self.probs_estimator(x)

        # adjust probabilities to mean skip goal:
        mean_p = torch.mean(skipping_probs)
        epsilon = 1E-7
        adjustment_factor = torch.abs(self.compression_rate + epsilon) / (mean_p + epsilon)
        skipping_probs = skipping_probs * adjustment_factor


        self.replace_with(skipping_probs,skipping_probs.detach() > 1,1 )
        self.replace_with(skipping_probs, skipping_probs.detach() <0, 0)

        # print("torch.min(skipping_probs).data[0]: {} max: {}".format(torch.min(skipping_probs).data[0],
        #                                                             torch.max(skipping_probs).data[0]))
        # First position is never skipped:
        skipping_probs[:, 0] = Variable(torch.ones(skipping_probs.size(0)), requires_grad=False)

        reduced, mask = self.skipper(x, skipping_probs)

        return (reduced, mask)

from torch import nn, optim
from torch.autograd import Variable
from torch.nn import Sequential, MSELoss
from torch.optim import Adam


class TrainedLinearTransform:
    """ This object holds a linear transformation for sequences, which is trained as an
    autoencoder, and typically stored for reuse. The decoder is discarded after training. """
    def __init__(self, input_length, output_length, input_encoding_dim, output_encoding_dim):
        self.encoder = nn.Sequential(nn.Linear(input_length * input_encoding_dim, output_encoding_dim * output_length),
                                     )
        self.decoder = nn.Linear(output_encoding_dim * output_length,
                                 input_length * input_encoding_dim)
        self.autoencoder = Sequential(self.encoder, self.decoder)
        self.input_length=input_length
        self.output_length=    output_length
        self.input_encoding_dim=input_encoding_dim
        self.output_encoding_dim=output_encoding_dim
        self.is_trained = False
        self.max_epochs = 10
        self.optimizer = optim.Adam(self.autoencoder.parameters(), betas=(0.9, 0.98), eps=1e-09,
                                    lr=1e-4)

    def encode(self,sequences):
        if not self.is_trained:
            self.train(sequences)
        self.encoder.eval()
        return  self.encoder(sequences)

    def train(self, sequences):

        self.autoencoder.train()
        mse = MSELoss()
        #for epoch in range(self.max_epochs):
        self.autoencoder.zero_grad()
        self.optimizer.zero_grad()
        encoded = self.encoder(Variable(sequences.data, requires_grad=True))
        result = self.decoder(encoded)
        loss = mse(result, Variable(sequences.data, requires_grad=False))
        #print("adjuster loss: {}".format(loss.data[0]))
        loss.backward()
        self.optimizer.step()
        #self.is_trained=True



class SequenceLengthAdjuster():
    def __init__(self, ):
        """Sequence adjusters can transform an input of some length into another length,
        while adjusting the number of dimensions of the encoding at the same time."""

        self.linear_pool = {}

    def adjust(self, sequences,  output_length, output_encoding_dim):
        input_length=sequences.size(1)
        input_encoding_dim=sequences.size(2)
        key=(input_length, output_length, input_encoding_dim, output_encoding_dim)
        if key in self.linear_pool:
            linear = self.linear_pool[key]
        else:
            linear=TrainedLinearTransform(*key)
            self.linear_pool[key]=linear

        batch_size = sequences.size(0)
        reshaped_input = sequences.view(batch_size, -1)
        output=linear.encode(reshaped_input)
        adjusted_seqs= output.view(batch_size, output_length, -1)
        return adjusted_seqs


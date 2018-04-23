import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def reconfigure(self, layer_index, layer_manager):
        d_model_input=layer_manager.get_input_dim(layer_index)
        d_ff=layer_manager.get_hidden_dim(layer_index)
        d_model_output=layer_manager.get_output_dim(layer_index)
        self.w_1 = nn.Linear(d_model_input, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model_output)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
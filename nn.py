import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, n_layers, name):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels = 4, out_channels = 8, kernel_size=10), 
            nn.ReLU(), 
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels = 8, out_channels = 8, kernel_size=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(120, 64),
        )

    def forward(self, x):
        output = self.net(x) 
        return output
        output = x
        for i, layer in enumerate(self.net):
            print(i, output.shape)
            output = layer(output)
            print(i, output.shape)
        return output


class CNN_LSTM_MIX_MODEL(nn.Module):
    def __init__(self,
                lstm_input_size,
                lstm_hidden_size,
                lstm_num_layers,
                name,
                n_lstm_layer = 1,
                ):
        super(CNN_LSTM_MIX_MODEL, self).__init__()
        self.lstm = nn.LSTM(
            input_size = lstm_input_size,
            hidden_size = lstm_hidden_size,
            num_layers = lstm_num_layers,
            batch_first = True,
            ) 
        self.cnn = CNN(
                       n_layers=3,
                       name = "{}.cnn".format(name))
        
        self.fc = nn.Linear(64 + lstm_hidden_size, 4)
        

    def forward(self, lstm_input, cnn_input, use_fp16=False):
        lstm_output = self.lstm(lstm_input)[0][:,3,:]
        cnn_output = self.cnn(cnn_input)

        concat_output = torch.cat((lstm_output, cnn_output), 1)
        last_layer = self.fc(concat_output)
        return last_layer


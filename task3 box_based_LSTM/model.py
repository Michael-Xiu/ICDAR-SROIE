# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as PACK
import torch.nn.init as init

# build the network   Morvan Zhou

class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.rnn = nn.LSTM(
            input_size=50,
            hidden_size=200,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        self.fc1 = nn.Sequential(
            nn.Linear(400, 100),  # 400 100
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(100, 50), # 100 50
            nn.ReLU()
        )
        self.predict = nn.Linear(50, 5)  # 50 5

        self.init_lstm()

    def forward(self, x):
        # 1. LSTM
        r_out, (h_out, c_out) = self.rnn(x)
        out = self.fc1(r_out)
        out = self.fc2(out)
        out = self.predict(out)

        return out

    def init_lstm(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():

            if isinstance(c, nn.Linear):
                nn.init.xavier_normal_(c.weight)
                nn.init.constant_(c.bias, 0.)
            elif isinstance(c, nn.LSTM):
                for param in c.parameters():
                    if len(param.shape) >= 2:
                        init.orthogonal_(param.data)
                    else:
                        init.normal_(param.data)


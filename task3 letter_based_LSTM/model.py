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
            input_size=1,
            hidden_size=64,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        self.fc1 = nn.Sequential(
            nn.Linear(12800, 2000),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2000, 100),
            nn.ReLU()
        )
        self.predict = nn.Linear(100, 5)
        # self.loc_fc1 = nn.Sequential(
        #     nn.Linear(4, 50),
        #     nn.ReLU()
        # )
        # self.loc_fc2 = nn.Sequential(
        #     nn.Linear(50, 20),
        #     nn.ReLU()
        # )
        # self.fc3 = nn.Linear(70, 5)

        # self.init_lstm()

    def forward(self, x):
        # 1. LSTM
        r_out, (h_out, c_out) = self.rnn(x)
        r_out_size = r_out.size()
        r_out_view = r_out.contiguous().view(r_out_size[0], r_out_size[1]*r_out_size[2])
        out = self.fc1(r_out_view)
        out = self.fc2(out)
        out = self.predict(out)

        # # 2. Localize
        # loc = x[:, :4, :]
        # loc = torch.squeeze(loc, dim=2)
        # loc_out = self.loc_fc1(loc)
        # loc_out = self.loc_fc2(loc_out)
        #
        # # 3. cat
        # out_cat = torch.cat((out, loc_out), dim=1)
        # total_out = self.fc3(out_cat)

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


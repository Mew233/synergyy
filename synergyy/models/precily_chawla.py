"""
[Chawla al., 2022]Chawla, S., Rockstroh, A., Lehman, M., Ratther, E., Jain, A., Anand, A., Gupta, A., Bhattacharya, N., Poonia, S., Rai, P. and Das, N., 2022. Gene expression
 based inference of cancer drug sensitivity. Nature communications, 13(1), pp.1-15
"""
import torch.nn as nn


class Precily_Chawla(nn.Module):
    def __init__(self,
        channels: int,
        dropout_rate = 0.1,
    ):
        super(Precily_Chawla, self).__init__()

        self.NN = nn.Sequential(
            nn.Linear(channels, channels), #1429,1429
            nn.ReLU(),
            nn.Linear(channels, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            #
            nn.Linear(512, 140),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            #
            nn.Linear(140, 200),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            #
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # if self.SHAP_analysis[0] == True:
        x = inputs
        x = self.NN(x)

        # else:
        #     x = inputs[0]
        #     x = self.NN(x)
           
        return x
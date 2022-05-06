"""
[Preueret al., 2018]Kristina Preuer, Richard PI Lewis, Sepp Hochre-iter, Andreas Bender, Krishna C Bulusu, and G ̈unter Klambauer.
DeepSynergy: Predicting Anti-Cancer Drug Synergy with DeepLearning.Bioinformatics, 34(9):1538–1546, 2018.
"""
import torch.nn as nn


class Deepsynergy_Preuer(nn.Module):
    def __init__(self,
        channels: int,
        dropout_rate = 0.5 
    ):
        super(Deepsynergy_Preuer, self).__init__()

        self.NN = nn.Sequential(
            nn.Linear(channels, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        x = inputs[0]
        x = self.NN(x)
        return x
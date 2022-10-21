"""

"""
import torch
import torch.nn as nn
from utilitis import EncoderLayer, DecoderLayer,OutputFeedForward, Norm
import copy
import numpy as np
from torch.nn import functional as F

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask=None):
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Combonet(nn.Module):
    def __init__(self, d_input, d_model, n_feature_type, N, heads, dropout):
        super().__init__()

        self.reduction = torch.nn.Linear(d_input,d_model,bias = True)

        self.encoder = Encoder(d_model, N, heads, dropout)
        #self.decoder = Decoder(d_input, d_model, N, heads, dropout)

        input_length = d_model*n_feature_type
        self.out = OutputFeedForward(input_length, n_feature_type, d_layers=[64, 1])

        self.dim_reduction = nn.Sequential(
            nn.Linear(1322, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 100),
        )
        self.dim_reduction2 = nn.Sequential(
            nn.Linear(3285, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 100),
        )

        # self.dose_response = nn.Sequential(
        #     nn.Linear(200, 64),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(64, 1),
        #     nn.Sigmoid()
        # )


    def forward(self, src, src2, src3,src_mask=None):
        
        # 1) Default
        # drug1 and cell
        # e_outputs = self.encoder(src, src_mask)
        # flat_e_output = e_outputs.view(-1, e_outputs.size(-2)*e_outputs.size(-1))

        # e_outputs2 = self.encoder(src2, src_mask)
        # flat_e_output2 = e_outputs2.view(-1, e_outputs2.size(-2)*e_outputs2.size(-1))
        # score1 = self.cell(flat_e_output)
        # score2 = self.cell(flat_e_output2)

        #single-layer neural network

        cell = self.dim_reduction(src3)
        src = self.dim_reduction2(src)
        src2 = self.dim_reduction2(src2)
        input = torch.stack((src, src2, cell), dim=1).squeeze()
        e_outputs3 = self.encoder(input, src_mask)
        flat_e_output3 = e_outputs3.view(-1, e_outputs3.size(-2)*e_outputs3.size(-1))

        # e_outputs = e_outputs3[:,0:2,:]
        # e_outputs2 = e_outputs3[:,1:,:]
        # flat_e_output = e_outputs.view(-1, e_outputs.size(-2)*e_outputs.size(-1))
        # flat_e_output2 = e_outputs2.view(-1, e_outputs2.size(-2)*e_outputs2.size(-1))

        # score1 = self.dose_response(flat_e_output)
        # score2 = self.dose_response(flat_e_output2)

        score1, score2 = None, None
        x_3 = self.out(flat_e_output3)

        return x_3, score1, score2


"""
[Liu Q, Xie L (2021)]TranSynergy: mechanism-driven interpretable deep neural network for the 
synergistic prediction and pathway deconvolution of drug combinations. PLoS Comput Biol 17(2):653
"""
import torch.nn as nn
from utilitis import EncoderLayer, DecoderLayer, Norm, OutputFeedForward
import copy
from torch import flatten

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_input, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_input, d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask=None):
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_input, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_input, d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask=None, trg_mask=None):
        x = trg
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transynergy_Liu(nn.Module):
    def __init__(self, d_input, d_model, n_feature_type, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(d_input, d_model, N, heads, dropout)
        self.decoder = Decoder(d_input, d_model, N, heads, dropout)
        self.out = OutputFeedForward(d_model, n_feature_type, d_layers=[512, 1])

        # self.regression_classify = nn.Sequential(
        #     nn.Linear(600, 512),
        #     nn.ELU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(512, 512),
        #     nn.ELU(),
        #     nn.Dropout(p=0.2),  
        #     nn.Linear(512, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, src, trg=None, src_mask=None, trg_mask=None):
        e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        #d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        #flat_d_output = d_output.view(-1, d_output.size(-2)*d_output.size(-1))
        #output = self.out(flat_d_output)
        flat_e_output = e_outputs.view(-1, e_outputs.size(-2)*e_outputs.size(-1))
        e_outputs = self.out(flat_e_output)
        output = e_outputs
        return output



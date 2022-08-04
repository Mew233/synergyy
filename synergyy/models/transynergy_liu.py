"""
[Liu Q, Xie L (2021)]TranSynergy: mechanism-driven interpretable deep neural network for the 
synergistic prediction and pathway deconvolution of drug combinations. PLoS Comput Biol 17(2):653
"""
import torch.nn as nn
from utilitis import EncoderLayer, DecoderLayer, Norm, OutputFeedForward, dice
import copy
from torch import flatten
import torch.nn.functional as F
from torch import cat
import numpy as np

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


class Transformer_drug(nn.Module):
    def __init__(self, max_drug_sm_len=244,
                 num_comp_char=60):
        super().__init__()
        #compound ID
        self.embed_comp = nn.Embedding(num_comp_char, num_comp_char, padding_idx=0)#padding's idx=0
        #encoding compound
        self.encoderlayer = nn.TransformerEncoderLayer(d_model=num_comp_char, nhead=4)
        self.encoder = nn.TransformerEncoder(self.encoderlayer, num_layers=1)
        #depthwise for compound encoding
        self.conv = nn.Conv2d(1, 1, (1, num_comp_char), groups=1)

    def forward(self, sm):

        sm = self.embed_comp(sm) #bsz*max_drug_sm_len*num_comp_char(embedding size)
        sm = self.encoder(sm)
        sm = self.conv(sm.unsqueeze(1)).squeeze()

        return sm


class Transynergy_Liu(nn.Module):
    def __init__(self, d_input, d_model, n_feature_type, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(d_input, d_model, N, heads, dropout)
        self.decoder = Decoder(d_input, d_model, N, heads, dropout)

        input_length = 768+244*2 #256/
        self.out = OutputFeedForward(input_length, n_feature_type, d_layers=[512, 1])

        # combined drug
        self.combined = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        self.transformer = Transformer_drug()
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, fp=None, sm1=None, sm2=None, trg=None, src_mask=None, trg_mask=None):
        
        # similarity_score = 1
        # similarity_score = 1 - dice(fp[0],fp[1])

        # 1) Default
        # e_outputs = self.encoder(src, src_mask)
        # flat_e_output = e_outputs.view(-1, e_outputs.size(-2)*e_outputs.size(-1))
        # output = self.out(flat_e_output)
        
        # 2) multimodal learning + FC
        # e_outputs = self.encoder(src, src_mask)
        # flat_e_output = e_outputs.view(-1, e_outputs.size(-2)*e_outputs.size(-1))

        # fp = fp.view(-1, fp.size(-2)*fp.size(-1))
        # fp = self.combined(fp)

        # cat_output = cat((flat_e_output, fp), dim=1)

        # output = self.out(cat_output)

        # 3) 
        d1_sm = self.transformer(sm1)
        d2_sm = self.transformer(sm2)
        
        e_outputs = self.encoder(src, src_mask)
        flat_e_output = e_outputs.view(-1, e_outputs.size(-2)*e_outputs.size(-1))

        cat_output = cat((flat_e_output, d1_sm, d2_sm), dim=1)

        output = self.out(cat_output)

        # return self.sigmoid(output-similarity_score)
        return output


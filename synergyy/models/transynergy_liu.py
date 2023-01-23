"""
[Liu Q, Xie L (2021)]TranSynergy: mechanism-driven interpretable deep neural network for the 
synergistic prediction and pathway deconvolution of drug combinations. PLoS Comput Biol 17(2):653
"""
from pickle import FALSE
import torch.nn as nn
from utilitis import EncoderLayer, DecoderLayer, Norm, OutputFeedForward, dice
import copy
from torch import flatten
import torch.nn.functional as F
from torch import cat, stack
import numpy as np
from models.TGSynergy import GNN_drug
import torch


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

class AE(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list = None, \
        dop: float = 0.1, noise_flag: bool = False, **kwargs) -> None:
        super(AE, self).__init__()
        self.latent_dim = latent_dim
        self.noise_flag = noise_flag
        self.dop = dop

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # build encoder
        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0], bias=True),
                #nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(self.dop)
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    #nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(self.dop)
                )
            )
        modules.append(nn.Dropout(self.dop))
        modules.append(nn.Linear(hidden_dims[-1], latent_dim, bias=True))

        self.encoder = nn.Sequential(*modules)

        # build decoder
        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dims[-1], bias=True),
                #nn.BatchNorm1d(hidden_dims[-1]),
                nn.ReLU(),
                nn.Dropout(self.dop)
            )
        )

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    #nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(self.dop)
                )
            )
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, input):
        encoded_input = self.encoder(input)
        # if self.normalize_flag:
        # encoded_input = nn.functional.normalize(encoded_input, p=2, dim=1)
        # output = self.decoder(encoded_input)

        return encoded_input


class Transynergy_Liu(nn.Module):
    def __init__(self, d_input, d_model, n_feature_type, N, heads, dropout):
        super().__init__()
        self.reduction = torch.nn.Linear(d_input,d_model,bias = True)
        self.reduction2 = torch.nn.Linear(3285,d_model,bias = True)

        self.encoder = Encoder(d_model, N, heads, dropout)
        self.encoder2 = Encoder(d_model, N, heads, dropout)
        # self.encoder = Encoder(d_input, d_model, N, heads, dropout)
        # self.decoder = Decoder(d_input, d_model, N, heads, dropout)

        input_length = 2187 #1280 #768+244*2 #256/
        self.out = OutputFeedForward(input_length, n_feature_type, d_layers=[128,64,1])
                
        # self.transformer = Transformer_drug()
        # self.transformer_mg = Encoder(d_input=256, d_model=d_model,N=N, heads=heads, dropout=dropout)
        # self.transformer_sm = Encoder(d_input=256, d_model=d_model,N=N, heads=heads, dropout=dropout)

        self.sigmoid = nn.Softmax()

        ## graph part from tgsynergy
        self.GNN_drug = GNN_drug(3, 128)
        self.drug_emb = nn.Sequential(            
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.reduction3 = torch.nn.Linear(1322,d_model,bias = True)
        self.combined = nn.Sequential(
            nn.Linear(3285*2, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 64)
        )

        self.ae = AE(3285,128)

    def forward(self, src, fp=None, sm1=None, sm2=None, sm1g=None, sm2g=None, \
        trg=None, src_mask=None, trg_mask=None):
        
        # similarity_score = 1
        # similarity_score = 1 - dice(fp[0],fp[1])

        # 1) Default
        # _src = self.reduction(src)
        # e_outputs = self.encoder(_src, src_mask)
        # flat_e_output = e_outputs.view(-1, e_outputs.size(-2)*e_outputs.size(-1))
        # output = self.out(flat_e_output)
        
        # 2) 5层transformer
        _src = self.reduction(src)
        _fp = self.reduction2(fp)
        _cell = sm1
        # _cell = torch.unsqueeze(_cell, dim=1)
        cat_input = cat((_src,_fp), dim=1)

        e_outputs = self.encoder(cat_input, src_mask)
        flat_e_output = e_outputs.view(-1, e_outputs.size(-2)*e_outputs.size(-1))

        cat_output = cat((flat_e_output, _cell), dim=1)
        output = self.out(cat_output)

        #3) 两个transformer
        # _src = self.reduction(src)
        # e_outputs = self.encoder(_src, src_mask)
        # flat_e_output = e_outputs.view(-1, e_outputs.size(-2)*e_outputs.size(-1))

        # #
        # _gsva = self.reduction2(sm1)
        # e_outputs2 = self.encoder2(_gsva, src_mask)
        # flat_e_output2 = e_outputs2.view(-1, e_outputs2.size(-2)*e_outputs2.size(-1))

        # cat_output = cat((flat_e_output, flat_e_output2), dim=1)

        # output = self.out(cat_output)

        return output


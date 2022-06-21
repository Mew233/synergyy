from models.baselines import *
from models.deepsynergy_preuer import *
from models.multitaskdnn_kim import *
from models.matchmaker_brahim import *
from models.deepdds_wang import *
from models.TGSynergy import *
from models.transynergy_liu import *
import argparse

def get_model(model_name,*args):

    base_model = baseline()
    if model_name in ['LR','XGBOOST','RF','ERT']:

        return getattr(base_model, model_name)()

    if model_name is "deepsynergy_preuer":

        return Deepsynergy_Preuer(channels=args[0],dropout_rate = 0.5)
    
    if model_name is "transynergy_liu":

        return Transynergy_Liu(d_input=2750, d_model=256, n_feature_type=3, N=1, heads=8, dropout=0.2)
        #return Transynergy_Liu(setting.d_input, setting.d_model, setting.n_feature_type, setting.n_layers, setting.attention_heads, setting.attention_dropout)
    if model_name is "multitaskdnn_kim":

        return Multitaskdnn_Kim(cell_channels=args[0],\
            drug_fp_channels=args[1],drug_tg_channels=args[2],dropout_rate = 0.5)
    
    if model_name is "matchmaker_brahim":

        return MatchMaker_Brahim(cell_channels=args[0],drug_channels=args[1],dropout_rate = 0.5)
    
    if model_name is "deepdds_wang":
        return DeepDDS_Wang()

    if model_name is "TGSynergy":

        return TGSynergy(cluster_predefine=args[0])

    # model, encoders =  autoencoder_NN(), autoencoder()
    # if model_name is "autodencoders":
    #     return model, encoders

if __name__ == "__main__":
    get_model('deepsynergy_preuer')
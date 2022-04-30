from models.baselines import *
from models.deepsynergy_preuer import *
from models.multitaskdnn_kim import *
from models.matchmaker_brahim import *
from models.deepdds_wang import *
from models.TGSynergy import *
import argparse

def get_model(model_name,*args):

    base_model = baseline()
    if model_name in ['LR','XGBOOST','RF','ERT']:

        return getattr(base_model, model_name)()

    if model_name is "deepsynergy_preuer":

        return Deepsynergy_Preuer(channels=args[0],dropout_rate = 0.5)

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
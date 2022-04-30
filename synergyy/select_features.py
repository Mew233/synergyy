import numpy as np 
import pandas as pd 
import os
from utilitis import *

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

def get_drug(original_list):
    targets = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'drug_data','all.csv'))
    drug_targets = explode_dpi(targets)

    drug_list_with_targets = drug_targets['Drug IDs'].unique().tolist()
    selected_drugs = list(set(original_list) & set(drug_list_with_targets))

    return selected_drugs

def get_GNNCell(cellFeatures_dicts, cellset):
    # targets = np.load(os.path.join(ROOT_DIR, 'data', 'cell_line_data','CCLE','cell_feature_cn_std.npy'),\
    #     allow_pickle=True).item()
    # selected_cells = list(targets.keys())

    temp = cellFeatures_dicts['exp']
    var_df = temp.var(axis=1)
    selected_genes = list(var_df.sort_values(ascending=False).iloc[:1000].index)

    selected_cells = list(set(cellset) & set(list(temp.columns)))

    return selected_cells


def get_cell(cellFeature_dicts, synergy_cellset, cell_omics, cell_filtered_by, matrix=False):
    
    def filter_by_variance():
        if len(cell_omics) > 1:
            # if mut/cnv/exp, use exp
            temp = cellFeature_dicts['exp']
        else:
            temp = cellFeature_dicts[cell_omics[0]]
        var_df = temp.var(axis=1)
        selected_genes = list(var_df.sort_values(ascending=False).iloc[:1000].index)
        
        return selected_genes

    # select genes based on criterion (variance or STRING)
    function_mapping = {'variance':'filter_by_variance', 'STRING':'filter_by_706_genes'}
    selected_genes = locals()[function_mapping[cell_filtered_by]]()


    CCLE_dicts = {}
    # Iterate over different CCLE type, for example exp or cn or mu
    for ccle_type in cell_omics:
        type_df = cellFeature_dicts[ccle_type]
        ## use cells in both synergyt_df and (CCLE_*)
        selected_cols = list(set(synergy_cellset) & set(list(type_df.columns)))
        ## use selected genes in both CCLE_exp and (CCLE_*)
        selected_rows = list(set(selected_genes) & set(list(type_df.index)))

        trimmed_type_df = type_df.loc[selected_rows, selected_cols]
        trimmed_type_df.dropna(axis=0, how='any',inplace=True)
        CCLE_dicts[ccle_type] = trimmed_type_df
                
        # if integrate is True, then the return value is a dataframe
        # otherwise, the return value is a dictionary of dataframe
        if matrix == True:
            feats = pd.concat(list(CCLE_dicts.values()))
        else:
            feats = CCLE_dicts

    return feats, selected_cols



def get_drug_feats_dim(drug_data_dicts, drug_feats):
    if len(drug_feats) == 1:
        dims = len(list(drug_data_dicts[drug_feats[0]].values())[0])

    else:
        dims = 0
        for feat_type in drug_feats:
            dims += len(list(drug_data_dicts[feat_type].values())[0])
    
    return dims

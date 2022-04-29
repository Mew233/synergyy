"""
    A collection of full training and evaluation pipelines.
"""
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn

from tqdm import tqdm
import joblib
from joblib import dump, load
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_validate

from prepare_data import *
from select_features import *
from get_model import get_model
from dataloader import dataloader, dataloader_graph, k_fold_trainer, k_fold_trainer_graph, evaluator, evaluator_graph
import torch_geometric.data


def prepare_data(args):
    _configs_ = configuration_from_json()
    config = _configs_[args.model]

    print("loading synergy dataset ...")
    synergy_df = load_synergy(config['synergy_df'])
    print("loading drug features ...")
    drugFeature_dicts = load_drug_features()
    print("loading cell line features ...")
    cellFeatures_dicts = load_cellline_features(config['cell_df'])


    # get full drug set and cell line set
    drugset = synergy_df['drug1'].unique().tolist() + synergy_df['drug2'].unique().tolist()
    cellset = synergy_df['cell'].unique().tolist()

    # Cleaning synergy data. Drugs not in drug–target interaction(DTI) are removed
    selected_drugs = get_drug(drugset)
    print("\ndrug features contructed")

    # Cleaning synergy data. Cells not having top variance genes are removed
    ## Cell feats: multi-omics dataset / get_cell select top genes by variance or kegg pathway

    cell_feats, selected_cells = get_cell(cellFeatures_dicts, cellset, config['cell_omics'], \
        config['cell_filtered_by'], config['get_cellfeature_concated'])

    print("cell line features constructed")
    synergy_df = synergy_df[(synergy_df['drug1'].isin(selected_drugs))\
        &(synergy_df['drug2'].isin(selected_drugs))&(synergy_df['cell'].isin(selected_cells))]


    print("\nSynergy triplets are: ")
    print("\t{} drugs:".format(len(selected_drugs)))
    print("\t{} cells:".format(len(selected_cells)))
    print("\t{} rows:".format(synergy_df.shape[0]))

    print("\ngenerating cell line features...")
    if config['get_cellfeature_concated'] == True:
        # in this case, cell_fets stores a dataframe containing features
        X_cell = np.zeros((synergy_df.shape[0], cell_feats.shape[0]))
        for i in tqdm(range(synergy_df.shape[0])):
            row = synergy_df.iloc[i]
            X_cell[i,:] = cell_feats[row['cell']].values
    else:
        X_cell = {}
        for feat_type in config['cell_omics']:
            print(feat_type, cell_feats[feat_type].shape[0])
            temp_cell = np.zeros((synergy_df.shape[0], cell_feats[feat_type].shape[0]))
            for i in tqdm(range(synergy_df.shape[0])):
                row = synergy_df.iloc[i]
                temp_cell[i,:] = cell_feats[feat_type][row['cell']].values
            X_cell[feat_type] = temp_cell
    

    if config['get_cellfeature_concated'] == True:
        print("cell features: ", X_cell.shape)
    else:
        print("cell features:", list(X_cell.keys()))


    # generate matrices for drug features
    # first generate individual data matrices for drug1 and drug2 and different feat types
    print("\ngenerating drug features...")
    drug_mat_dict = {}
    for feat_type in config['drug_omics']:
### 需要修改, append到一个list
        if feat_type=='smiles2graph':
            temp_X_drug1, temp_X_drug2 = [], []
            for i in tqdm(range(synergy_df.shape[0])):
                row = synergy_df.iloc[i]
                ## This is graph. append graph object to list
                temp_X_drug1.append(drugFeature_dicts[feat_type][int(row['drug1'])])
                temp_X_drug2.append(drugFeature_dicts[feat_type][int(row['drug2'])])

        ## this is valid for tabular features
        else:
            dim = drugFeature_dicts[feat_type].shape[0]
            temp_X_drug1 = np.zeros((synergy_df.shape[0], dim))
            temp_X_drug2 = np.zeros((synergy_df.shape[0], dim))
            for i in tqdm(range(synergy_df.shape[0])):
                row = synergy_df.iloc[i]
                temp_X_drug1[i,:] = drugFeature_dicts[feat_type][int(row['drug1'])]
                temp_X_drug2[i,:] = drugFeature_dicts[feat_type][int(row['drug2'])]

        drug_mat_dict[feat_type+"_1"] = temp_X_drug1
        drug_mat_dict[feat_type+"_2"] = temp_X_drug2

    # now aggregate drug features based on whether they should be summed (drug1+drug2)
    X_drug_temp = {}
    if config['get_drugs_summed'] == True:
        for feat_type in config['drug_omics']:
            temp_X = drug_mat_dict[feat_type+"_1"] + drug_mat_dict[feat_type+"_2"]
            X_drug_temp[feat_type] = temp_X
    else:
        X_drug_temp = drug_mat_dict
    
    # now aggregate drug features based on whether they should be concatenatd
    if config['get_drugfeature_concated'] == False:
        X_drug = X_drug_temp
    else:
        # in this case, drug feature is a numpy array instead of dict of arrays
        X_drug = np.concatenate(list(X_drug_temp.values()), axis=1)
    

    if config['get_drugfeature_concated'] == True:
        print("drug features: ", X_drug.shape)
    else:
        print("drug features")
        print(list(X_drug.keys()))
        for key, value in X_drug.items():
            print(key, len(value))

    
    Y_score = (synergy_df['score']>args.synergy_thres).astype(int).values


    return X_cell, X_drug, Y_score

def training_baselines(X_cell, X_drug, Y, args):
    # --------------- baseline  --------------- #
    if args.model in ['LR','XGBOOST','RF','ERT']:
        X = np.concatenate([X_cell,X_drug], axis=1)
        X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # init model
        model = get_model(args.model)
        # returned for evaluation
        test_loader = {}
        test_loader['X_test'] = X_test
        test_loader['actuals'] = Y_test

        # prepare the cross-validation procedure
        kfold = KFold(n_splits=5, random_state=42, shuffle=True)
        
        # load the best model
        if args.train_test_mode == 'test':
            rfc_fit = 'best_model_%s.pth' % args.model
            scores = 0
        elif args.train_test_mode == 'train':
            # evaluate model
            cv_results = cross_validate(model, X_trainval, Y_trainval, cv=kfold, scoring='roc_auc', return_estimator=True)
            scores = cv_results['test_score']
            rfc_fit = cv_results['estimator']
            # select the best
            rfc_fit = rfc_fit[np.argmax(scores)]
            # save it
            save_path = os.path.join(ROOT_DIR, 'best_model_%s.pth' % args.model)
            joblib.dump(rfc_fit,save_path)
            # should return address at the end
            rfc_fit = 'best_model_%s.pth' % args.model

    return rfc_fit, scores, test_loader


def training(X_cell, X_drug, Y, args):

# --------------- multitask dnn --------------- #
    if args.model == 'multitaskdnn_kim':
        
        X_cell_trainval, X_cell_test, \
        X_fp_drug1_trainval, X_fp_drug1_test,\
        X_fp_drug2_trainval, X_fp_drug2_test,\
        X_tg_drug1_trainval, X_tg_drug1_test,\
        X_tg_drug2_trainval, X_tg_drug2_test,\
        Y_trainval, Y_test \
        = train_test_split(X_cell, X_drug['morgan_fingerprint_1'],X_drug['morgan_fingerprint_2'], \
                                X_drug['drug_target_1'],X_drug['drug_target_2'],Y, \
                                test_size=0.2, random_state=42)

        cell_channels = X_cell_trainval.shape[1]
        drug_fp_channels = X_fp_drug1_trainval.shape[1]
        drug_tg_channels = X_tg_drug1_trainval.shape[1]

        # init model, order is important 
        model = get_model(args.model,cell_channels,drug_fp_channels,drug_tg_channels)

        # train_val set for k-fold, test set for testing
        # should be compatible with fp_drug, tg_drug, fp_drug2, tg_drug2, cell
        train_val_dataset, test_loader = dataloader(\
            X_fp_drug1_trainval=X_fp_drug1_trainval, X_fp_drug1_test=X_fp_drug1_test,\
            X_tg_drug1_trainval=X_tg_drug1_trainval, X_tg_drug1_test=X_tg_drug1_test,\
            X_fp_drug2_trainval=X_fp_drug2_trainval, X_fp_drug2_test=X_fp_drug2_test,\
            X_tg_drug2_trainval=X_tg_drug2_trainval, X_tg_drug2_test=X_tg_drug2_test,\
            X_cell_trainval=X_cell_trainval, X_cell_test=X_cell_test,\
            Y_trainval=Y_trainval, Y_test=Y_test
                )

        # load the best model
        if args.train_test_mode == 'test':
            net_weights = 'best_model_%s.pth' % args.model
        elif args.train_test_mode == 'train':
            net_weights = k_fold_trainer(train_val_dataset,model,args)
   
# --------------- deep synergy --------------- #
    elif args.model == 'deepsynergy_preuer':
        
        X = np.concatenate([X_cell,X_drug], axis=1)
        #X_{}_trainval, X_{}_test, Y_{}_trainval, Y_{}_test
        X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        channels = X_trainval.shape[1]

        # init model
        model = get_model(args.model,channels)
        
        # train_val set for k-fold, test set for testing
        train_val_dataset, test_loader = dataloader(X_trainval=X_trainval, X_test=X_test, Y_trainval=Y_trainval, Y_test=Y_test)

        # load the best model
        if args.train_test_mode == 'test':
            net_weights = 'best_model_%s.pth' % args.model
        elif args.train_test_mode == 'train':
            net_weights = k_fold_trainer(train_val_dataset,model,args)
   
# --------------- matchmaker --------------- #
    elif args.model == 'matchmaker_brahim':
        
        X_cell_trainval, X_cell_test, \
        X_fp_drug1_trainval, X_fp_drug1_test,\
        X_fp_drug2_trainval, X_fp_drug2_test,\
        Y_trainval, Y_test \
        = train_test_split(X_cell, X_drug['morgan_fingerprint_1'],X_drug['morgan_fingerprint_2'], Y, \
                                test_size=0.2, random_state=42)

        cell_channels = X_cell_trainval.shape[1]
        drug_channels = X_fp_drug1_trainval.shape[1]

        # init model
        model = get_model(args.model,cell_channels,drug_channels)
    
        # should be compatible with fp_drug, fp_drug2, cell
        train_val_dataset, test_loader = dataloader(\
            X_fp_drug1_trainval=X_fp_drug1_trainval, X_fp_drug1_test=X_fp_drug1_test,\
            X_fp_drug2_trainval=X_fp_drug2_trainval, X_fp_drug2_test=X_fp_drug2_test,\
            X_cell_trainval=X_cell_trainval, X_cell_test=X_cell_test,\
            Y_trainval=Y_trainval, Y_test=Y_test
                )

        # load the best model
        if args.train_test_mode == 'test':
            net_weights = 'best_model_%s.pth' % args.model
        elif args.train_test_mode == 'train':
            net_weights = k_fold_trainer(train_val_dataset,model,args)

# --------------- deepdds --------------- #
    elif args.model == 'deepdds_wang':
        X_cell_trainval, X_cell_test, \
        X_deepdds_sm_drug1_trainval, X_deepdds_sm_drug1_test,\
        X_deepdds_sm_drug2_trainval, X_deepdds_sm_drug2_test,\
        Y_trainval, Y_test \
        = train_test_split(X_cell, X_drug['smiles2graph_1'],X_drug['smiles2graph_2'], Y, \
                                test_size=0.2, random_state=42)


    
        # should be compatible with fp_drug, fp_drug2, cell
        train_val_dataset,test_loader = dataloader_graph(\
            X_deepdds_sm_drug1_trainval=X_deepdds_sm_drug1_trainval, X_deepdds_sm_drug1_test=X_deepdds_sm_drug1_test,\
            X_deepdds_sm_drug2_trainval=X_deepdds_sm_drug2_trainval, X_deepdds_sm_drug2_test=X_deepdds_sm_drug2_test,\
            X_cell_trainval=X_cell_trainval, X_cell_test=X_cell_test,\
            Y_trainval=Y_trainval, Y_test=Y_test
                )
        
        # init model
        model = get_model(args.model)


        # load the best model
        if args.train_test_mode == 'test':
            net_weights = 'best_model_%s.pth' % args.model
        elif args.train_test_mode == 'train':
            net_weights = k_fold_trainer_graph(train_val_dataset,model,args)

    
    return model, net_weights, test_loader




def evaluate(model, model_weights, test_loader, args):

    if args.model in ['LR','XGBOOST','RF','ERT']:
    
        model = load(model)
        actuals, predictions = test_loader['actuals'], model.predict_proba(test_loader['X_test'])[:,1]

    elif args.model == 'deepdds_wang':

        actuals, predictions = evaluator_graph(model, model_weights,test_loader)
        
    else:

        actuals, predictions = evaluator(model, model_weights,test_loader)

    auc = roc_auc_score(y_true=actuals, y_score=predictions)
    ap = average_precision_score(y_true=actuals, y_score=predictions)
    val_results = {'AUC':auc, 'AUPR':ap}

    return val_results



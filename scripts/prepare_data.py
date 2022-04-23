import numpy as np 
import pandas as pd 
import os
from rdkit.Chem import AllChem
import rdkit

from utilitis import *

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

def load_synergy(dataset):
    '''
    Load synergy datasets. 
    Load multi-omics dataset and revise into the specified format.

    param:
        dataset: str
    '''

    function_mapping = {'NCI_ALMANAC':'process_almanac', 'DrugComb':'process_drugcomb'}

    def process_drugcomb():
        data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'synergy_data','DrugComb','drugcomb_trueset.csv'))
        data = data[['drug_row', 'drug_col','cell_line_name','study_name','tissue_name',\
          'synergy_zip','synergy_loewe','synergy_hsa','synergy_bliss','DepMap_ID_x','RRID_x',\
          'pubchemID_x','compound0_x','pubchemID_y','compound0_y']]
        data_trim = data[['compound0_x','compound0_y','DepMap_ID_x','study_name','synergy_loewe']]

        ## clean the scores
        data_trim = data_trim[data_trim['synergy_loewe'] != '\\N']
        data_trim['synergy_loewe'] = data_trim['synergy_loewe'].astype(float)

        data_trim['compound0_x'] = data_trim['compound0_x'].apply(lambda x: split_it(x))
        data_trim['compound0_y'] = data_trim['compound0_y'].apply(lambda x: split_it(x))


        # # summarize 3*3 or 5*3 data into one by calculating the mean score
        summary_data = data_trim.groupby(['compound0_x','compound0_y','DepMap_ID_x']).agg({\
            "synergy_loewe":'mean',"study_name":'count'}).reset_index().rename(columns={\
                'synergy_loewe':'MEAN_SCORE','study_name':'count'}).astype({'compound0_x':'int32','compound0_y':'int32'})

        # # some experiments may fail and get NA values, drop these experiments
        summary_data = summary_data.dropna()

        summary_data = summary_data[['compound0_x','compound0_y','DepMap_ID_x','MEAN_SCORE']].rename(columns={\
            'compound0_x':'drug1','compound0_y':'drug2','DepMap_ID_x':'cell','MEAN_SCORE':'score'})

        return summary_data

    
    def process_almanac():
        pass

    # use locals() to run the specified function
    data = locals()[function_mapping[dataset]]()
    return data



def load_cellline_features(dataset):
    '''
    Load cell line features. load all data in the specified dataset and revise into the same format.
    Store all kinds of cell lines features in a dictionary.

    param:
        dataset: str
    '''

    function_mapping = {'CCLE':'process_CCLE'}

    def process_CCLE():

        def load_file(postfix):
            df = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'cell_line_data','CCLE','CCLE_%s.csv' % postfix),sep=',')


            ## need to transform mut into one-hot dataframe
            if postfix == 'mut':
                # remove entre_id which is 0
                mu = df.loc[df['Entrez_Gene_Id'].values !=0]

                # transform data to one_hot format
                clean_cells_ALL = list(set(list(mu['DepMap_ID']))) 
                clean_genes_ALL = list(set(list(mu['Entrez_Gene_Id']))) 
                CCLE_mu = pd.DataFrame(one_hot(mu,clean_cells_ALL,clean_genes_ALL), columns=['Entrez gene id']+clean_genes_ALL)

   
            # use entrez gene id as index
            if postfix != "mut": 
                df.columns = ['Entrez gene id']+[split_it_cell(_) for _ in list(df.columns)[1:]]
                df_transpose = df.T
                # set first row as column
                df_transpose.columns = df_transpose.iloc[0]
                processed_data = df_transpose.drop(df_transpose.index[0])

            else:
                ## same as exp
                mu_transpose = CCLE_mu.T
                # set first row as column
                mu_transpose.columns = mu_transpose.iloc[0]
                processed_data = mu_transpose.drop(mu_transpose.index[0])
            return processed_data
        
        
        # load all cell line features
        save_path = os.path.join(ROOT_DIR, 'data', 'cell_line_data','CCLE')
        save_path = os.path.join(save_path, 'input_cellline_data.npy')
        if not os.path.exists(save_path):
            data_dicts = {}
            for file_type in ['exp', 'cn', 'mut']:
                data_dicts[ file_type ] = load_file(file_type)
            np.save(save_path, data_dicts)
        else:
            data_dicts = np.load(save_path,allow_pickle=True).item()

        return data_dicts


    data = locals()[function_mapping[dataset]]()
    return data



def load_drug_features():
    
    def process_fingerprint():
        # load fingerprint data
        ## column is drug, index is morgan bits
        # Read SDF File
        supplier = rdkit.Chem.SDMolSupplier(os.path.join(ROOT_DIR, 'data', 'drug_data','structures.sdf'))
        molecules = [mol for mol in supplier if mol is not None]

        fingerprints = dict()
        for mol in molecules:
            drugbank_id = mol.GetProp('DATABASE_ID')
            ## or use MACCS: (MACCSkeys.GenMACCSKeys(molecules[0])
            ## Here is morgan
            bitvect = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=256).ToBitString()
            fingerprint = [int(i) for i in bitvect]
            fingerprints[split_it(drugbank_id)] = fingerprint

        fingerprints = pd.DataFrame(fingerprints)
        fingerprints.columns = fingerprints.columns.astype(int)
        return fingerprints

    def process_dpi():
        # load drug target dataset
        targets = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'drug_data','all.csv'))
        drug_targets = explode_dpi(targets)

        drug_mapping = dict(zip(drug_targets['Drug IDs'].unique().tolist(), range(len(drug_targets['Drug IDs'].unique()))))
        gene_mapping = dict(zip(drug_targets['NCBI_ID'].unique().tolist(), range(len(drug_targets['NCBI_ID'].unique()))))
        encoding = np.zeros((len(drug_targets['Drug IDs'].unique()), len(drug_targets['NCBI_ID'].unique())))
        for _, row in drug_targets.iterrows():
            encoding[drug_mapping[row['Drug IDs']], gene_mapping[row['NCBI_ID']]] = 1
        target_feats = dict()
        for drug, row_id in drug_mapping.items():
            target_feats[int(drug)] = encoding[row_id].tolist()
        return pd.DataFrame(target_feats)

    def process_smiles():
        pass
        
    
    save_path = os.path.join(ROOT_DIR, 'data', 'drug_data')
    save_path = os.path.join(save_path, 'input_drug_data.npy')
    if not os.path.exists(save_path):
        data_dicts = {}
        data_dicts['morgan_fingerprint'] = process_fingerprint()
        data_dicts['drug_target'] = process_dpi()
        np.save(save_path, data_dicts)
    else:
        data_dicts = np.load(save_path,allow_pickle=True).item()

    return data_dicts

import numpy as np 
import pandas as pd
import collections
import os
from rdkit.Chem import AllChem
import rdkit

from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem.EState import Fingerprinter

from utilitis import *


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

def load_synergy(dataset,args):
    '''
    Load synergy datasets. 
    Load multi-omics dataset and revise into the specified format.

    param:
        dataset: str
    '''

    function_mapping = {'DrugComb':'process_drugcomb', 'Sanger2022':'process_sanger2022',}

    def process_drugcomb():
        data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'synergy_data','DrugComb','drugcomb_trueset_NoDup.csv'))
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
        # summary_data = data_trim.groupby(['compound0_x','compound0_y','DepMap_ID_x']).agg({\
        #     "synergy_loewe":'mean',"study_name":'count'}).reset_index().rename(columns={\
        #         'synergy_loewe':'MEAN_SCORE','study_name':'count'}).astype({'compound0_x':'int32','compound0_y':'int32'})

        # # some experiments may fail and get NA values, drop these experiments
        summary_data = data_trim.dropna()

        summary_data = summary_data[['compound0_x','compound0_y','DepMap_ID_x','synergy_loewe']].rename(columns={\
            'compound0_x':'drug1','compound0_y':'drug2','DepMap_ID_x':'cell','synergy_loewe':'score'})

        return summary_data
    
    def process_sanger2022():
        data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'synergy_data','Sanger2022','drug_combinations_TGSA_Jaaks.csv'))
        data['compound0_x'] = data['compound0_x'].apply(lambda x: split_it(x))
        data['compound0_y'] = data['compound0_y'].apply(lambda x: split_it(x))
        data['synergy_loewe'] = data['synergy_loewe'].apply(lambda x: x*1)

        summary_data = data[['compound0_x','compound0_y','DepMap_ID','synergy_loewe']].rename(columns={\
            'compound0_x':'drug1','compound0_y':'drug2','DepMap_ID':'cell','synergy_loewe':'score'})
        
        return summary_data

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
        
        # def load_cpi_network():
        #     df = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'cell_line_data','CCLE','CCLE_mut.csv'),sep=',')
        #     mu = df.loc[df['Entrez_Gene_Id'].values !=0]
        #     mu = mu[['DepMap_ID','Entrez_Gene_Id']]

        #     #deplete proteins in dpi, which not in ppi
        #     ppi = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'cell_line_data','PPI','protein-protein_network.csv'))
        #     selected_proteins = ppi['protein_a'].unique().to_list() + ppi['protein_b'].unique().to_list()
        #     mu_new = mu[(mu['Entrez_Gene_Id'].isin(selected_proteins))]
        #     return mu_new  



        # load all cell line features
        save_path = os.path.join(ROOT_DIR, 'data', 'cell_line_data','CCLE')
        save_path = os.path.join(save_path, 'input_cellline_data.npy')
        if not os.path.exists(save_path):
            data_dicts = {}
            for file_type in ['exp', 'cn', 'mut']:
                data_dicts[ file_type ] = load_file(file_type)
            # load GNN_cell
            gnn_path = os.path.join(ROOT_DIR, 'data', 'cell_line_data','CCLE')
            gnn_path = os.path.join(gnn_path, 'cell_feature_cn_std.npy')
            cell_dict = np.load(gnn_path,allow_pickle=True).item()
            data_dicts['GNN_cell'] = cell_dict

            # # load cpi_network
            # data_dicts['cpi_network'] = load_cpi_network()

            np.save(save_path, data_dicts)
        else:
            data_dicts = np.load(save_path,allow_pickle=True).item()
            # load GNN_cell
            gnn_path = os.path.join(ROOT_DIR, 'data', 'cell_line_data','CCLE')
            gnn_path = os.path.join(gnn_path, 'cell_feature_cn_std.npy')
            cell_dict = np.load(gnn_path,allow_pickle=True).item()
            data_dicts['GNN_cell'] = cell_dict

            # # load cpi_network
            # data_dicts['cpi_network'] = load_cpi_network()
            
        edge_path = os.path.join(ROOT_DIR, 'data', 'cell_line_data','CCLE')
        edge_path = os.path.join(edge_path, 'edge_index_PPI_0.95.npy')
        edge_index = np.load(edge_path)
        #for key, value in a_dict.items()
        for key, value in data_dicts['GNN_cell'].items():
            #value should be a data object
            value.edge_index = torch.tensor(edge_index, dtype=torch.long)

        return data_dicts

    data = locals()[function_mapping[dataset]]()
    return data


    

def load_drug_features():
    
    supplier = rdkit.Chem.SDMolSupplier(os.path.join(ROOT_DIR, 'data', 'drug_data','structures.sdf'))
    molecules = [mol for mol in supplier if mol is not None]

    def process_fingerprint():
        # load fingerprint data
        ## column is drug, index is morgan bits
        # Read SDF File
        # supplier = rdkit.Chem.SDMolSupplier(os.path.join(ROOT_DIR, 'data', 'drug_data','structures.sdf'))
        

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
    
    def process_MACCS():
        # MACCS fingerprint
        pass
    
    
    def process_ChemicalDescrpitor():
        # Chemical descriptors
        def get_fps(mol):
            calc=MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
            ds = np.asarray(calc.CalcDescriptors(mol))
            arr=Fingerprinter.FingerprintMol(mol)[0]
            return np.append(arr,ds)
        
        
        fingerprints = dict()
        for mol in molecules:
            drugbank_id = mol.GetProp('DATABASE_ID')
            fingerprints[split_it(drugbank_id)] = get_fps(mol)

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
        
        proessed_dpi = pd.DataFrame(target_feats)
        proessed_dpi.index = drug_targets['NCBI_ID'].unique()
        proessed_dpi.to_csv(os.path.join(ROOT_DIR, 'results','proessed_dpi.csv'))

        return proessed_dpi

    ##RWR algorithm for drug-target from transynergy
    def process_dpi_RWR():
        
        network = nx.read_edgelist(os.path.join(ROOT_DIR, 'data','cell_line_data','PPI','string_network'), delimiter='\t', nodetype=int,
                           data=(('weight', float),))
        
        data_dicts = np.load(os.path.join(ROOT_DIR, 'data', 'cell_line_data','CCLE','input_cellline_data.npy'),allow_pickle=True).item()
        ccle = data_dicts['exp']

        #column is drugbank id, row is entrez id
        drug_target = pd.read_csv(os.path.join(ROOT_DIR, 'results','proessed_dpi.csv'), index_col=0)
        drug_target = drug_target.loc[drug_target.index.isin(list(network.nodes)), :]
        drug_target = drug_target.loc[drug_target.index.isin(list(ccle.index)), :]

        drug_target.fillna(0.00001, inplace = True)

        # generate I matrix
        subnetwork = network.subgraph(list(drug_target.index.values))
        A = (subnetwork.subgraph(c) for c in nx.connected_components(subnetwork))
        subgraphs = list(A)
        subgraph = subgraphs[0]
        subgraph_nodes = list(subgraph.nodes)
        I = pd.DataFrame(np.identity(len(subgraph_nodes)), index=subgraph_nodes, columns=subgraph_nodes)
        print("Preparing network propagation kernel")

        drug_target = drug_target.loc[drug_target.index.isin(list(I.index.values)), :]
        kernel = network_propagation(subgraph, I, alpha=0.5, symmetric_norm=False, verbose=True)
        print("Got network propagation kernel. Start propagate ...")

        genes = I.index.values
        propagated_drug_target = network_kernel_propagation(network=subgraph, network_kernel=kernel,
                                                     binary_matrix=drug_target.T)
        propagated_drug_target = propagated_drug_target.loc[:, list(genes)]
        print("Propagation finished")
        propagated_drug_target = standarize_dataframe(propagated_drug_target)
        
        return propagated_drug_target.T


    # ## for graphsynergy =============unfinished===============
    # def process_dpi_network():
    #     targets = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'drug_data','all.csv'))
    #     drug_targets = explode_dpi(targets)
    #     #deplete proteins in dpi, which not in ppi
    #     ppi = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'cell_line_data','PPI','protein-protein_network.csv'))

    #     selected_proteins = list(set(ppi['protein_a'])) + list(set(ppi['protein_b']))
        
    #     drug_targets_new = drug_targets[(drug_targets['NCBI_ID'].isin(selected_proteins))]
        
        
    #     def get_target_dict(dpi_df):
    #         dp_dict = collections.defaultdict(list)
    #         drug_list = list(set(dpi_df['drug']))
    #         for drug in drug_list:
    #             drug_df = dpi_df[dpi_df['drug']==drug]
    #             target = list(set(drug_df['protein']))
    #             dp_dict[drug] = target
    #         return dp_dict


    #     return drug_targets_new



    def process_smiles2graph():
        # r
        # smiles = pd.read_csv('./data/Drugs/DPIALL_smiles.csv')
        smilesgraph_dict = dict()
        for mol in molecules:
            drugbank_id = mol.GetProp('DATABASE_ID')
            smiles = mol.GetProp('SMILES')
            try:
                smilesgraph_dict[split_it(drugbank_id)] = smile_to_graph(smiles)
            except AttributeError:
                pass
        # np.save('./data/Drugs/drug_feature_graph.npy', drug_dict)
        return smilesgraph_dict

    #----------------For TGSynergy-------------------------------   
    def process_smiles2graph_TGSynergy():
        # r
        # smiles = pd.read_csv('./data/Drugs/DPIALL_smiles.csv')
        smilesgraph_dict = dict()
        for mol in molecules:
            drugbank_id = mol.GetProp('DATABASE_ID')
            smiles = mol.GetProp('SMILES')
            try:
                smilesgraph_dict[split_it(drugbank_id)] = smiles2graph(smiles)
            except AttributeError:
                pass
        # np.save('./data/Drugs/drug_feature_graph.npy', drug_dict)
        return smilesgraph_dict
        
    save_path = os.path.join(ROOT_DIR, 'data', 'drug_data')
    save_path = os.path.join(save_path, 'input_drug_data.npy')
    if not os.path.exists(save_path):
        data_dicts = {}
        data_dicts['morgan_fingerprint'] = process_fingerprint()
        data_dicts['chemical_descriptor'] = process_ChemicalDescrpitor()
        data_dicts['drug_target'] = process_dpi()
        data_dicts['drug_target_rwr'] = process_dpi_RWR()
        data_dicts['drug_target_rwr'].columns = data_dicts['drug_target_rwr'].columns.values.astype(int)
        data_dicts['smiles2graph'] = process_smiles2graph()
        data_dicts['smiles2graph_TGSynergy'] = process_smiles2graph_TGSynergy()
        #data_dicts['dpi_network'] = process_dpi_network()
        np.save(save_path, data_dicts)
    else:
        data_dicts = np.load(save_path,allow_pickle=True).item()
        #data_dicts['drug_target_rwr'] = process_dpi_RWR()
        #data_dicts['drug_target_rwr'].columns = data_dicts['drug_target_rwr'].columns.values.astype(int)

        selected_genes = data_dicts['drug_target_rwr'].index
        a = data_dicts['drug_target']
        a = a.loc[a.index.isin(list(selected_genes)), :]
        data_dicts['drug_target'] = a
       
    return data_dicts

if __name__ == "__main__":
    load_drug_features()
    # process_GNNCell()
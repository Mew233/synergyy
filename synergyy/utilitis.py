import re
import os
import pandas as pd
import argparse
import networkx as nx


from rdkit import Chem
import numpy as np
import pandas as pd
import torch
import torch_geometric
import torch_geometric.data
from dgllife.utils import *

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

# split DB -- unique identifier
def split_it(compound):
    return int(re.split('\d*\D+',compound)[1])

# split PubChemID
def split_it_cell(compound):
    return int(re.search(r'\((.*?)\)', compound).group(1))

def one_hot(mu,clean_cells_ALL,clean_genes_ALL):
    onehot_rows = list()
    for cell in clean_cells_ALL:
        onehot_row = list()
        temp_cell = mu[mu['DepMap_ID']==cell]
        temp_gene_list = list(temp_cell['Entrez_Gene_Id'])
        onehot_row.append(cell)
        for gene in clean_genes_ALL:
            if gene in temp_gene_list:
                onehot_row.append(1)
            else:
                onehot_row.append(0)
        onehot_rows.append(onehot_row)
    return onehot_rows


# Explode Drug-Target Interaction
def explode_dpi(targets):
    targets = targets[targets['Species'] == 'Humans']
    targets['Drug IDs'] = targets['Drug IDs'].str.split('; ').fillna(targets['Drug IDs'])
    targets = targets.explode('Drug IDs')

    targets = targets[['HGNC ID','Name','Gene Name','Drug IDs']]
    ## convert HGNC ID to NCBI ID
    entrez_IDs_df = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'drug_data','hgnc_ncbi.txt'), sep="\t", index_col=False)
    entrez_to_genename = dict(entrez_IDs_df[['HGNC ID', 'NCBI Gene ID']].values)
    targets = targets.dropna(subset=['HGNC ID'])
    targets['NCBI_ID'] = targets['HGNC ID'].apply(lambda x: entrez_to_genename[x])
    ## remove NCBI_ID where is nan
    targets = targets.dropna(subset=['NCBI_ID'])
    targets['NCBI_ID'] = targets['NCBI_ID'].apply(lambda x: int(x))
    targets['Drug IDs'] = targets['Drug IDs'].apply(lambda x: split_it(x))
    drug_targets = targets

    return drug_targets

# read the method_config file
import json
def configuration_from_json(args):
    with open(os.path.join(ROOT_DIR,'%s%s%s' % ('config_',args.model,'.json')), "r") as jsonfile:
        config = json.load(jsonfile)
    return config


## write json from argparse
def write_config(args):
    with open(os.path.join(ROOT_DIR,'%s%s%s' % ('config_',args.model,'.json')), "w") as f:
        json.dump(
            {
                args.model:{
                "synergy_df": args.synergy_df,
                "drug_omics": args.drug_omics,
                "cell_df": args.cell_df,
                "cell_omics": args.cell_omics,
                "cell_filtered_by": args.cell_filtered_by,
                "model_name": args.model,
                "get_cellfeature_concated": args.get_cellfeature_concated,
                "get_drugfeature_concated": args.get_drugfeature_concated,
                "get_drugs_summed": args.get_drugs_summed,
                }
             },
                f
            )
        

## SMILES2Graph for prepare_data for TGSynergy

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    8 features are canonical, 2 features are from OGB
    """
    featurizer_funcs = ConcatFeaturizer([atom_type_one_hot,
                                         atom_degree_one_hot,
                                         atom_implicit_valence_one_hot,
                                         atom_formal_charge,
                                         atom_num_radical_electrons,
                                         atom_hybridization_one_hot,
                                         atom_is_aromatic,
                                         atom_total_num_H_one_hot,
                                         atom_is_in_ring,
                                         atom_chirality_type_one_hot,
                                         ])
    atom_feature = featurizer_funcs(atom)
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    featurizer_funcs = ConcatFeaturizer([bond_type_one_hot,
                                         # bond_is_conjugated,
                                         # bond_is_in_ring,
                                         # bond_stereo_one_hot,
                                         ])
    bond_feature = featurizer_funcs(bond)

    return bond_feature


def smiles2graph(mol):
    """
    Converts SMILES string or rdkit's mol object to graph Data object without remove salt
    :input: SMILES string (str)
    :return: graph object
    """

    if isinstance(mol, Chem.rdchem.Mol):
        pass
    else:
        mol = Chem.MolFromSmiles(mol)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = torch_geometric.data.Data(x=torch.tensor(x, dtype=torch.float),
                 edge_index=torch.tensor(edge_index, dtype=torch.long),
                 edge_attr=torch.tensor(edge_attr), dtype=torch.float)

    return graph


# -------------------------------------------
## SMILES2Graph for prepare_data for DeepDDS

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    
    graph = None
    try:
        graph = torch_geometric.data.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0))
        graph.__setitem__('c_size', torch.LongTensor([c_size]))

    except Exception:
        pass
    
    return graph



if __name__ == "__main__":
    # configuration_from_json()
    test = smile_to_graph('CC[C@H](C)[C@H](NC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)[C@H](CC(O)=O)NC(=O)CNC(=O)[C@H](CC(N)=O)NC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(N)=N)NC(=O)[C@@H]1CCCN1C(=O)[C@H](N)CC1=CC=CC=C1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(O)=O)C(=O)N[C@@H](CCC(O)=O)C(=O)N[C@@H](CC1=CC=C(O)C=C1)C(=O)N[C@@H](CC(C)C)C(O)=O')

    #print(test)
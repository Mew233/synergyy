import re
import os
import pandas as pd
import argparse

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
def configuration_from_json():
    with open(os.path.join(ROOT_DIR,'config.json'), "r") as jsonfile:
        config = json.load(jsonfile)
    return config


## write json from argparse
def write_config(args):
    with open(os.path.join(ROOT_DIR,'config.json'), "w") as f:
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
        

if __name__ == "__main__":
    configuration_from_json()
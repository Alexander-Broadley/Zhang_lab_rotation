#prevent pandas warning due to workaround for deprecation of logical operators
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


import torch
import pandas as pd
import numpy as np
from torch import nn

from filter_dataset import filter_datasets

DATA_ROOT = './data'
MODEL_ROOT = './models'

#load the merged results to get target genes for which there is a model as much faster than loading expression file 
merged_results = pd.read_csv(f'{DATA_ROOT}/merged_results.csv', index_col= 0, header = 0)

#define root directory
DATA_ROOT = './data'

#filter datasets as in model construction - ensures retrieve the same set of TFs used for model construction
TF_expressions, net = filter_datasets(DATA_ROOT, load_genes=False)

#define function to subset transcription factors to only those that directly regulate the target gene
def TF_subset(net, target_gene):
    #simpler version as per discussion w/ Cheng
    #returns all the TFs in the network that directly connect to the target gene
    return(list(net['TF'][net['Gene'] == target_gene])) 

#repeat through all genes for which there is a model
for gene in merged_results.index:
    gene = 'MBOAT7'
    print(f'Processing model for {gene}')
    
    #run same target gene specific pre-processing (retrieves list of genes that will correspond to the input features)
    TF_expression_subset = list(TF_expressions[TF_subset(net, gene)].columns)

    #intialise a df to store model parameters
    param_df = pd.DataFrame(index = TF_expression_subset, columns = ['in_weight', 'in_bias', 'out_weight', 'out_bias', 'reg_direction'])

    #load TPM model for given target gene
    model = torch.load(f'{MODEL_ROOT}/TPM_models/{gene}_TPM_model.pth', weights_only = False)

    #add model parameters to df
    param_df['in_weight'] = np.asarray(model.linear_in.weight.data.cpu())
    param_df['in_bias'] = np.asarray(model.linear_in.bias.data.cpu())
    param_df['out_weight'] = np.asarray(model.linear_out.weight.data.cpu().flatten())
    #for linear-out layer there is only 1 bias so repeat for length of dataframe
    param_df['out_bias'] = (list(np.asarray(model.linear_out.bias.data.cpu())) * len(TF_expression_subset))

    for TF in TF_expression_subset:
        param_df.loc[TF, 'reg_direction'] = net[net['TF'] == TF][net['Gene'] == gene]['Interaction'].values

    #save the dataframe
    param_df.to_csv(f'{DATA_ROOT}/TPM_model_parameters/{gene}_TPM_MPs.csv')

#prevent pandas warning due to workaround for deprecation of logical operators
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


import torch
import pandas as pd
import numpy as np
from torch import nn
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from model_building.filter_dataset import filter_datasets

DATA_ROOT = '../../../data'
MODEL_ROOT = '../../models'

#Load network
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
#Load ARCHS4 gene expressions
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/ARCHS4_healthy_log.tsv"), sep='\t', header=0)


#filter datasets as in model construction - ensures retrieve the same set of TFs used for model construction
TF_expressions, gene_expressions = filter_datasets(net, gene_expressions)

#repeat through all genes for which there is a model
for gene in gene_expressions.columns:
    print(f'Processing model for {gene}')

    if gene in TF_expressions.columns:
        TF_expression_subset = TF_expressions.drop(gene, axis = 1)
    else:
        TF_expression_subset = TF_expressions

    #intialise a df to store model parameters
    param_df = pd.DataFrame(index = TF_expression_subset.columns, columns = ['in_weight', 'in_bias', 'out_weight', 'out_bias'])

    #load TPM model for given target gene
    model = torch.load(f'{MODEL_ROOT}/HEALTHY_models/{gene}_model.pth', weights_only = False)

    #add model parameters to df
    param_df['in_weight'] = np.asarray(model.module.linear_in.weight.data.cpu())
    param_df['in_bias'] = np.asarray(model.module.linear_in.bias.data.cpu())
    param_df['out_weight'] = np.asarray(model.module.linear_out.weight.data.cpu().flatten())
    #for linear-out layer there is only 1 bias so repeat for length of dataframe
    param_df['out_bias'] = (list(np.asarray(model.module.linear_out.bias.data.cpu())) * TF_expression_subset.shape[1])

    #save the dataframe
    param_df.to_csv(f'{DATA_ROOT}/model_parameters/{gene}_MPs.csv')
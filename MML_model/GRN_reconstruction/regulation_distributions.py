#this script gets the contributions of each TF to each target gene in each sample (male and female external samples only). Produces one dataframe per TG where columns are TFs and rows are samples

import os
import sys
import torch
from torch.utils.data import DataLoader
from torch import tensor

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#set seed for reproducibility 
torch.manual_seed(1475460913)

from model_building.filter_dataset import filter_datasets
#import dataset object
from model_building.customTFGE_dataset import CustomTFGE

from model_building.activation_functions import activation_function_map


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#define root directory
DATA_ROOT = '../../data'
MODEL_ROOT = '../models'

print('Loading Datasets')
#Load network
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
#Load target gene expressions

#try new filtering function
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/ARCHS4_healthy_TPM_stricter.tsv"), sep='\t', header=0, index_col = 0).T
#male_meta =  pd.read_csv(f"{DATA_ROOT}/Full data files/ARCHS4_female_healthy_meta.csv", index_col = 0)
#print(male_meta.head())
#gene_expressions = gene_expressions.loc[male_meta.index]



TF_expressions, gene_expressions = filter_datasets(net, GE_df=gene_expressions)

#create array to store dataframes of each target genes regulation for concatenation
inf_networks_list = []

#batch size = 1 as want to run 1 sample at a time
batch_size = 1

activation_function = activation_function_map['MML']

print(gene_expressions.head())
for target_gene in gene_expressions.columns:
    
    print(f'Processing {target_gene} model')

    #target gene must be removed from it's own prediction on a per-target basis
    if target_gene in TF_expressions.columns:
        print('Target gene is a TF, removing from TF dataset')
        TF_expression_subset = TF_expressions.drop(target_gene, axis = 1)
    else:
        TF_expression_subset = TF_expressions

    #create dataframe to store contributions
    contributions_df = pd.DataFrame(columns = gene_expressions.columns)

    


    dataset = CustomTFGE(device, TF_expressions=TF_expression_subset, gene_expressions=gene_expressions, network = net, target_gene = target_gene)

    model = torch.load(f"{MODEL_ROOT}/HEALTHY_models/{target_gene}_model.pth", weights_only = False)
    eval_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    model.eval()

    '''
    TF_in_model = list(TF_expression_subset.columns)

    #intialise a df to store model parameters
    param_df = pd.DataFrame(index = TF_expression_subset.columns, columns = ['in_weight', 'in_bias', 'out_weight', 'out_bias'])
    param_df['TF'] = param_df.index
    param_df = param_df.reset_index()

    #add model parameters to df
    param_df['in_weight'] = np.asarray(model.module.linear_in.weight.data.cpu())
    param_df['in_bias'] = np.asarray(model.module.linear_in.bias.data.cpu())
    param_df['out_weight'] = np.asarray(model.module.linear_out.weight.data.cpu().flatten())
    #for linear-out layer there is only 1 bias so repeat for length of dataframe
    param_df['out_bias'] = (list(np.asarray(model.module.linear_out.bias.data.cpu())) * len(TF_in_model))
    '''

    with torch.no_grad():
        #this gets the indexes of all TFs post-activation that have a value < 0.5 and adds one to the corresponding index in threshold tracker 
        for batch, (X, y) in enumerate(eval_dataloader):
            #contributions = (np.asarray(X.cpu() * np.asarray(param_df['in_weight']) + np.asarray(param_df['in_bias'])))
            contributions = (activation_function((X * model.linear_in.weight) + model.linear_in.bias, leak = 0.01) * model.linear_out.weight) + model.linear_out.bias.data
            contributions_df[batch] = contributions
    
    contributions_df.to_csv(f'./data/TF_contributions/{target_gene}_reg_cont_distributions.csv')
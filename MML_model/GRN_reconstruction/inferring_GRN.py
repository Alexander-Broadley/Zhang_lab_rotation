#converting notebook into python file for easier running
import os
import sys
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import tensor
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#set seed for reproducibility 
torch.manual_seed(1475460913)

from model_building.filter_dataset import filter_datasets
#import dataset object
from model_building.customTFGE_dataset import CustomTFGE

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
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/ARCHS4_healthy_log.tsv"), sep='\t', header=0, index_col = 0)
print(gene_expressions.head())
male_meta =  pd.read_csv(f"{DATA_ROOT}/Full data files/ARCHS4_female_healthy_meta.csv", index_col = 0)
print(male_meta.head())
gene_expressions = gene_expressions.loc[male_meta.index]


print(gene_expressions.head())
TF_expressions, gene_expressions = filter_datasets(net, GE_df=gene_expressions)

#create array to store dataframes of each target genes regulation for concatenation
inf_networks_list = []

#batch size = 1 as want to run 1 sample at a time
batch_size = 1

for target_gene in gene_expressions.columns:
    print(f'Processing {target_gene} model')

    #target gene must be removed from it's own prediction on a per-target basis
    if target_gene in TF_expressions.columns:
        print('Target gene is a TF, removing from TF dataset')
        TF_expression_subset = TF_expressions.drop(target_gene, axis = 1)
    else:
        TF_expression_subset = TF_expressions


    dataset = CustomTFGE(device, TF_expressions=TF_expression_subset, gene_expressions=gene_expressions, network = net, target_gene = target_gene)

    model = torch.load(f"{MODEL_ROOT}/HEALTHY_models/{target_gene}_model.pth", weights_only = False)
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

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


    with torch.no_grad():
        #create an empty numpy array that keeps track of how many times each input TF has a post-activation value past a threshold
        threshold_tracker = np.zeros(shape = len(TF_in_model))
        for batch, (X, y) in enumerate(eval_dataloader):  
            if target_gene == 'ADNP':
                print(X.shape)
                print(len(np.asarray(param_df['in_weight'])))
            #this gets the indexes of all TFs post-activation that have a value < 0.5 and adds one to the corresponding index in threshold tracker   
            threshold_index = np.where(np.asarray((model(X).cpu() * np.asarray(param_df['in_weight']) + np.asarray(param_df['in_bias']))) >= 1)
            threshold_tracker[threshold_index] += 1

        regulating_index = np.where(threshold_tracker == len(dataset))

        past_thresh_tfs = param_df.loc[regulating_index]['TF']
        past_thresh_reg = param_df.loc[regulating_index]['out_weight']

        #create a dataframe that will store the entire network
        inferred_net = pd.DataFrame(columns = ['TF','Gene', 'Reg'])
        inferred_net['TF'] = past_thresh_tfs
        inferred_net['Gene'] = target_gene
        inferred_net['Reg'] = past_thresh_reg
        inf_networks_list.append(inferred_net)

inferred_GRN = pd.concat(inf_networks_list)

#inferred_GRN['Reg'] = np.sign(inferred_GRN['Reg'])
inferred_GRN.to_csv('./data/100per_act_inferred_GRN_FEMALE.csv')
print('Finished and saved female GRNs')
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

#define the sex of the samples getting reg contributions for
#sex = 'male'
sex = 'female'

#try new filtering function
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/ARCHS4_{sex}_external_expressions.tsv"), sep='\t', header=0, index_col = 0)
#male_meta =  pd.read_csv(f"{DATA_ROOT}/Full data files/ARCHS4_male_healthy_meta.csv", index_col = 0)
#print(male_meta.head())
#gene_expressions = gene_expressions.loc[male_meta.index]



TF_expressions, gene_expressions = filter_datasets(net, GE_df=gene_expressions)

#create array to store dataframes of each target genes regulation for concatenation
inf_networks_list = []

#batch size = 1 as want to run 1 sample at a time
batch_size = 1

activation_function = activation_function_map['MML']['activation']

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
    contributions_df = pd.DataFrame(columns = TF_expression_subset.columns)

    dataset = CustomTFGE(device, TF_expressions=TF_expression_subset, gene_expressions=gene_expressions, network = net, target_gene = target_gene)

    model = torch.load(f"{MODEL_ROOT}/MF_external_models/{target_gene}_model.pth", weights_only = False)
    eval_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()

    with torch.no_grad():
        #this gets the indexes of all TFs post-activation that have a value < 0.5 and adds one to the corresponding index in threshold tracker 
        for batch, (X, y) in enumerate(eval_dataloader):
            #contributions = (np.asarray(X.cpu() * np.asarray(param_df['in_weight']) + np.asarray(param_df['in_bias'])))
            #contributions = np.asarray(((activation_function((X * model.module.linear_in.weight) + model.module.linear_in.bias, leak = 0.01) * model.module.linear_out.weight) + model.module.linear_out.bias.data).cpu().flatten())
            contributions = np.asarray((activation_function((X * model.module.linear_in.weight) + model.module.linear_in.bias, leak = 0.01) * model.module.linear_out.weight).cpu().flatten())
            contributions_df.loc[batch] = contributions
    #print(contributions_df.head())
    contributions_df.to_csv(f'./data/{sex}_contribution_dists/{target_gene}_reg_cont_distributions.csv')
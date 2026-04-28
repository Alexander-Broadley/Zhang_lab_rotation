import pandas as pd
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch import nn

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from Zhang_lab.MML_model.model_building.batch_loss import gene_MSE_all_samples
from model_building.customTFGE_dataset import CustomTFGE
from model_building.simpleMMLModel import SimpleMMLModel
from model_building.activation_functions import activation_function_map

#define device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#define function to subset transcription factors to only those that directly regulate the target gene
def TF_subset(net, target_gene):
    #simpler version as per discussion w/ Cheng
    #returns all the TFs in the network that directly connect to the target gene
    return(list(net['TF'][net['Gene'] == target_gene])) 

#---------------------------------------------------------------
#Load Datasets
#---------------------------------------------------------------

MODEL_ROOT = '../../models'
DATA_ROOT = '../../data'

print('Loading orig Datasets')
#Load network
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
#Load target gene expressions
gene_expressions_orig = pd.read_csv((f"{DATA_ROOT}/Full data files/Geneexpression (full).tsv"), sep='\t', header=0)
#load TF expressions
TF_expressions_orig = pd.read_csv((f"{DATA_ROOT}/Full data files/TF(full).tsv"), sep='\t', header=0)

#filter network to only include TFs that are in the dataset
net = net[net['TF'].isin(TF_expressions_orig.columns)]

print('Filtering genes in datasets')
#filter genes to nodes in network
network_tfs = set(net['TF'].unique())      # TFs
network_genes = set(net['Gene'].unique())  # target genes
network_nodes = network_tfs | network_genes

TF_expressions_orig = TF_expressions_orig[[gene for gene in TF_expressions_orig.columns if gene in list(network_tfs)]]
gene_expressions_orig = gene_expressions_orig[[gene for gene in gene_expressions_orig.columns if gene in list(network_genes)]]

print('Loading Datasets')

#Load network
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
print('Loaded Network')
#Load gene expressions - these have been pre-processed already
gene_expressions = pd.read_csv(f"{DATA_ROOT}/GEP_Liver_bulk/GEP_Normal_gene_expressions.csv", index_col = 0,header=0)
print('Loaded Gene Expressions')
#load GEP TF expressions - these have been pre-processed already
TF_expressions = pd.read_csv(f"{DATA_ROOT}/GEP_Liver_bulk/GEP_Normal_TF_expressions.csv", index_col = 0, header=0)
print('Loaded TF expressions')

TF_expressions = TF_expressions.drop('TF', axis = 1)
print(TF_expressions_orig.shape)
print(TF_expressions.shape)

#TF_expressions = TF_expressions[TF_expressions_orig.columns]


for x in TF_expressions.columns:
     if x in TF_expressions_orig.columns:
          pass
     else:
        print(x)
#---------------------------------------------------------------
#load models per target gene
#---------------------------------------------------------------

#batch_size = TF_expressions.shape[0]

loss_fn = nn.MSELoss()

#initialise eval results df
#results_df = pd.DataFrame(index = gene_expressions.index, columns = gene_expressions.columns)

external_predicted = pd.DataFrame(columns= gene_expressions.columns)

print('Starting Model Scoring')

for target_gene in gene_expressions.columns:
    print(f'Generating scores for {target_gene} model')
    #print(f'Creating model for {target_gene}')
    TF_expression_subset = TF_expressions#[TF_subset(net, target_gene)]

    dataset = CustomTFGE(device, TF_expressions=TF_expression_subset, gene_expressions=gene_expressions, network = net, target_gene = target_gene)
    eval_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    model = torch.load(f"{MODEL_ROOT}/PEARSONS_200_randn_all/{target_gene}_model.pth", weights_only = False).to(device)

    #put model in eval mode
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(eval_dataloader):     
                external_predicted[target_gene] = model(X).cpu()

        
external_predicted.to_csv('../data/GEP_results/GEP_Normal_results_P2RA.csv')

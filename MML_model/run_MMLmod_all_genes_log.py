######################################################################
# Import Packages
######################################################################

import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import tensor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

######################################################################
# Import and define functions and objects
######################################################################

#import early stopping class
from early_stopper import EarlyStopping
#import LEMBAS activation functions
from activation_functions import activation_function_map
#import model
from simpleMMLModel import SimpleMMLModel
#import MSE per batch calculator
from gene_MSE_all_samples import gene_MSE_all_samples
#import function to train a single epoch
from train_one_epoch import train_one_epoch
#import dataset object
from customTFGE_dataset import CustomTFGE

#define function to subset transcription factors to only those that directly regulate the target gene
def TF_subset(net, target_gene):
    #simpler version as per discussion w/ Cheng
    #returns all the TFs in the network that directly connect to the target gene
    return(list(net['TF'][net['Gene'] == target_gene]))

######################################################################
# Initialisations and data loading
######################################################################

#set seed for reproducibility 
torch.manual_seed(1475460913)

#define device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#define root directory
DATA_ROOT = '/Users/alexanderbroadley/Documents/PhD/Zhang Lab/Learning_PyTorch/data'

print('Loading Datasets')

#Load network
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
#Load target gene expressions
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/Geneexpression (full).tsv"), sep='\t', header=0)
#load TF expressions
TF_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/TF(full).tsv"), sep='\t', header=0)

#filter network to only include TFs that are in the dataset
net = net[net['TF'].isin(TF_expressions.columns)]

print('Filtering genes in datasets')
#filter genes to nodes in network
network_tfs = set(net['TF'].unique())      # TFs
network_genes = set(net['Gene'].unique())  # target genes
network_nodes = network_tfs | network_genes

#filter TF and gene expressions to only those in network
TF_expressions = TF_expressions[[gene for gene in TF_expressions.columns if gene in list(network_nodes)]]
gene_expressions = gene_expressions[[gene for gene in gene_expressions.columns if gene in list(network_nodes)]] 

######################################################################
# Transform data
######################################################################

#add 1 then log transform TF and gene expression values
TF_expressions = TF_expressions + 1
TF_expressions = TF_expressions.apply(np.log10)

#same to gene expressions
gene_expressions = gene_expressions + 1
gene_expressions = gene_expressions.apply(np.log10)

######################################################################
# Intialise hyperparameters
######################################################################

#define training parameters
learning_rate = 1e-3
#run 1 sample at a time, but run through each sample per training epoch
batch_size = TF_expressions.shape[0]
#max 100 epochs
epochs = 100
#initialize MSE loss function - same as LEMBAS
loss_fn = nn.MSELoss()

#create a results dataframe
results_df = pd.DataFrame(index = gene_expressions.columns, columns = ['train_score', 'test_score'])

######################################################################
# Create model per target gene
######################################################################

#for the remaining code need to execute per target gene (per gene in gene_expressions)
for target_gene in gene_expressions.columns:
    #initialise an early stopper to end training if loss on test data does not fall by at least 0.01 MSE for 3 eopochs in a row
    early_stopping = EarlyStopping(patience=3, delta=0.01, verbose=True)
    
    #print(f'Creating model for {target_gene}')
    TF_expression_subset = TF_expressions[TF_subset(net, target_gene)]

    dataset = CustomTFGE(device, TF_expressions=TF_expression_subset, gene_expressions=gene_expressions, network = net, target_gene = target_gene)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

    model = nn.DataParallel(SimpleMMLModel(activation_function_map['MML'], TF_expression_subset.shape[1])).to(device)

    #initialise same optimiser as LEMBAS
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for t in range(epochs):
        #print(f"Epoch {t+1}\n-------------------------------")
        train_one_epoch(train_dataloader, model, loss_fn, optimiser)

        test_loss = gene_MSE_all_samples(test_dataloader, model, loss_fn, target_gene)
        #print(f'Test MSE this epoch is: {test_loss}')

        early_stopping.check_early_stop(test_loss)
    
        if early_stopping.stop_training:
            print(f"Early stopping at epoch {t+1}")
            break
    
    results_df.loc[target_gene, 'train_score'] = gene_MSE_all_samples(test_dataloader, model, loss_fn, target_gene, rev_log = True)
    results_df.loc[target_gene, 'test_score'] = gene_MSE_all_samples(train_dataloader, model, loss_fn, target_gene, rev_log = True)

    torch.save(model, f'/Users/alexanderbroadley/Documents/PhD/Zhang Lab/Learning_PyTorch/models/logTPM_models/{target_gene}_logTPM_model.pth')

results_df.to_csv('data/MSE_results_logTPM.csv')
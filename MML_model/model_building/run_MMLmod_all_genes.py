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

#import early stopping class
from model_building.early_stopper import EarlyStopping

#import LEMBAS activation functions
from model_building.activation_functions import activation_function_map
#import model
from model_building.simpleMMLModel import SimpleMMLModel
#import MSE per batch calculator
from model_building.gene_MSE_all_samples import gene_MSE_all_samples
#import function to train a single epoch
from model_building.train_one_epoch import train_one_epoch

#import dataset object
from model_building.customTFGE_dataset import CustomTFGE

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#define root directory
DATA_ROOT = '../data'

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

TF_expressions = TF_expressions[[gene for gene in TF_expressions.columns if gene in list(network_nodes)]]
gene_expressions = gene_expressions[[gene for gene in gene_expressions.columns if gene in list(network_nodes)]] 

#define function to subset transcription factors to only those that directly regulate the target gene
def TF_subset(net, target_gene):
    #simpler version as per discussion w/ Cheng
    #returns all the TFs in the network that directly connect to the target gene
    return(list(net['TF'][net['Gene'] == target_gene]))


#define training parameters
learning_rate = 1e-3
#run 1 sample at a time, but run through each sample per training epoch
batch_size = TF_expressions.shape[0]
#max 100 epochs
epochs =  100 #12747 * 5
#initialize MSE loss function - same as LEMBAS
#loss_fn = nn.MSELoss()

#trying with new loss_fn
from model_building.pearsons_loss import PearsonLoss
#loss_fn = PearsonLoss()
loss_fn = nn.MSELoss()

#create a results dataframe
results_df = pd.DataFrame(index = gene_expressions.columns, columns = ['train_loss', 'test_loss', 'stopped_early', 'in_features'])

#create dataframes to track the predicted and actual expressions for train and test datasets - recreating datasets with pytorch is unreliable and cannot store 161000 datasets

#intialise training dataset predicted values df
train_predicted = pd.DataFrame(columns=gene_expressions.columns)
test_predicted = pd.DataFrame(columns=gene_expressions.columns)

#initialise training dataset actual values df - easier to do it this way as can get values after torch train-test split
train_actual = pd.DataFrame(columns=gene_expressions.columns)
test_actual = pd.DataFrame(columns=gene_expressions.columns)


#for the remaining code need to execute per target gene (per gene in gene_expressions)
for target_gene in gene_expressions.columns:
    print(target_gene)
    #initialise an early stopper to end training if loss on test data does not fall by at least 0.01 MSE for 3 eopochs in a row
    early_stopping = EarlyStopping(patience=3, delta=0.01, verbose=True)
    
    #print(f'Creating model for {target_gene}')
    TF_expression_subset = TF_expressions[TF_subset(net, target_gene)]

    dataset = CustomTFGE(device, TF_expressions=TF_expression_subset, gene_expressions=gene_expressions, network = net, target_gene = target_gene)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

    model = SimpleMMLModel(activation_function_map['MML'], TF_expression_subset.shape[1])

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    #initialise same optimiser as LEMBAS
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for t in range(epochs):
        
        train_loss = train_one_epoch(train_dataloader, model, loss_fn, optimiser)

        test_loss = gene_MSE_all_samples(test_dataloader, model, loss_fn, target_gene)
        #print(f'Test MSE this epoch is: {test_loss}')

        if (t+1) % 5 == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            print(test_loss)

        early_stopping.check_early_stop(test_loss)
    
        if early_stopping.stop_training:
            print(f"Early stopping at epoch {t+1}")  
            results_df.loc[target_gene, 'stopped_early'] = 1
            break

     #put model in eval mode
    model.eval()
    with torch.no_grad():
        #for each dataset then load the data (using same seed as training to get same train test split)
        #get the predicted and actual score this way as easiest way to get values from the torch train test split
        for batch, (X, y) in enumerate(train_dataloader):
            train_predicted[target_gene] = model(X).cpu()
            train_actual[target_gene] = y.cpu()

        for batch, (X, y) in enumerate(test_dataloader):
            test_predicted[target_gene] = model(X).cpu()
            test_actual[target_gene] = y.cpu()

     #put model in eval mode
    model.eval()
    with torch.no_grad():
        #for each dataset then load the data (using same seed as training to get same train test split)
        #get the predicted and actual score this way as easiest way to get values from the torch train test split
        for batch, (X, y) in enumerate(train_dataloader):
            train_predicted[target_gene] = model(X).cpu()
            train_actual[target_gene] = y.cpu()

        for batch, (X, y) in enumerate(test_dataloader):
            test_predicted[target_gene] = model(X).cpu()
            test_actual[target_gene] = y.cpu()

    results_df.loc[target_gene, 'train_loss'] = train_loss
    results_df.loc[target_gene, 'test_loss'] = test_loss
    results_df.loc[target_gene, 'in_features'] = len(TF_expression_subset.columns)
    torch.save(model, f'../models/pearsons_randn_allTFs_200/{target_gene}_model.pth')



train_actual.to_csv(f'{DATA_ROOT}/Train_dataset_actual_expressions.csv')
train_predicted.to_csv(f'{DATA_ROOT}/Train_dataset_predicted_expressions_PEARSONS_all.csv')

test_actual.to_csv(f'{DATA_ROOT}/Test_dataset_actual_expressions.csv')
test_predicted.to_csv(f'{DATA_ROOT}/Test_dataset_predicted_expressions_PEARSONS_all.csv')


results_df.to_csv('../data/PEARSONS_results.csv')


#converting notebook into python file for easier running

import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import tensor
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

#import LEMBAS activation functions
from activation_functions import activation_function_map

#define device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#define root directory
DATA_ROOT = '/home/alexanderb/LEMBAS-RNN-benchmark'

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

#defining the network
class SimpleMMLModel(nn.Module):
    def __init__(self, activation_function, n_tfs):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.leak = 0.01
        self.linear_in = nn.Linear(n_tfs, n_tfs)
        self.linear_out = nn.Linear(n_tfs, 1)

        # activation function
        self.activation = activation_function['activation']
        self.delta = activation_function['delta']
        self.onestepdelta_activation_factor = activation_function['onestepdelta']
    
    def forward(self, x):
        #project an input for each TF to activation with a linear layer
        expressions = self.linear_in(x)
        expressions = self.activation(expressions, self.leak)
        expressions = self.linear_out(x)
        expressions = expressions.flatten()
        return(expressions)

#define pytorch dataset object
class CustomTFGE(Dataset):
    def __init__(self, device, TF_expressions, gene_expressions, target_gene, network, transform=None, target_transform=None):
        '''
        Custom dataset for loading expression data across all TFs and samples and one target gene expression across all samples
        For a given index will return all TF expressions in one sample and target gene expression across all samples

        Parameters
        --------------
        device : torch device
            torch device to put dataset on, must be the same device as the model
        TF_expressions : Pandas dataframe
            pandas dataframe of TF expressions where columns are gene names and rows are samples. Expressions should be TPM normalised and log10 transformed.
        gene_expressions : Pandas dataframe
            pandas dataframe of target gene expressions, columns are gene names and rows are samples. Expressions should be TPM normalised and log10 transformed.
        target_gene : String
            String containing name of target gene for given model.
        network : networkx network
            Undirected network of the singalling pathway of interest. Used to filter TFs used when predicting target gene expression.
        '''
        #no transforms needed so set to none
        self.transform = transform
        self.target_transform = target_transform

        #load network
        self.network = network
        self.TF_expressions = TF_expressions
        self.gene_expressions = gene_expressions

        #subset to just target gene of interest
        self.target_gene = target_gene
        self.gene_expressions = self.gene_expressions[self.target_gene]

        #convert to torch tensors
        self.TF_expressions = torch.tensor(np.asarray(self.TF_expressions).T, dtype = torch.float32, device = device)
        self.gene_expressions = torch.tensor(np.asarray(self.gene_expressions), dtype = torch.float32, device = device)

    def __len__(self):
        #length of the dataset is the number of samples (not TFs in the dataset) - 15935
        return self.TF_expressions.shape[1]

    def __getitem__(self, idx):
        #get all TFs from sample correspinding to index
        TFs_exp = self.TF_expressions[:, idx]
        #get the target gene expression for the target model
        Gene_exp = self.gene_expressions[idx]
        #returns TFs for sample idx and target gene for sample idx
        return TFs_exp, Gene_exp

#define the training loop
def train_loop(dataloader, model, loss_fn, optimiser):
    losses = []
    size = len(dataloader.dataset)
    
    #put model in train mode
    model.train()
    for batch, (X, y) in enumerate(dataloader):     
        #zero the gradient for each batch
        optimiser.zero_grad()

        #create a prediction
        pred = model(X)
        
        #calculate prediction loss
        loss = loss_fn(pred, y)

        #backpropogate for given loss
        loss.backward()
        optimiser.step()

        
        loss = loss.item()
        losses.append(loss)

        if batch +1 == size:
            print(f'Epoch Finished at batch {batch}\n')

        #print loss every 500 batches - this is effectively every 500th sample
        if batch % 1000 == 0:
            print(f"loss: {round(loss, 10)}")
        
    return(losses)


def gene_MSE_all_samples(dataloader, model, loss_fn, target_gene):
    #put model in eval mode
    model.eval()
    for batch, (X, y) in enumerate(dataloader):     

        #create a prediction - this will be 1 value for given sample and target gene
        pred = model(X)
        
        #calculate prediction loss - this will be one loss value in a distribution across samples per target gene
        loss = loss_fn(pred, y)
        #get loss value
        loss = loss.item()

    return(loss)

#define training parameters
learning_rate = 1e-3
#run 1 sample at a time, but run through each sample per training epoch
batch_size = TF_expressions.shape[0]
#train on each sample 5 times
epochs =  10 #12747 * 5
#initialize MSE loss function - same as LEMBAS
loss_fn = nn.MSELoss()


#create a results dataframe
results_df = pd.DataFrame(index = gene_expressions.columns, columns = ['train_score', 'test_score'])

#for the remaining code need to execute per target gene (per gene in gene_expressions)
for target_gene in gene_expressions.columns:
    print(f'Creating model for {target_gene}')
    TF_expression_subset = TF_expressions[TF_subset(net, target_gene)]

    dataset = CustomTFGE(device, TF_expressions=TF_expression_subset, gene_expressions=gene_expressions, network = net, target_gene = target_gene)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    model = SimpleMMLModel(activation_function_map['MML'], TF_expression_subset.shape[1]).to(device)

    #initialise same optimiser as LEMBAS
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        losses = train_loop(train_dataloader, model, loss_fn, optimiser)

    results_df.loc[target_gene, 'train_score'] = gene_MSE_all_samples(test_dataloader, model, loss_fn, target_gene)
    results_df.loc[target_gene, 'test_score'] = gene_MSE_all_samples(train_dataloader, model, loss_fn, target_gene)

results_df.to_csv('data/MSE_results_TPM.csv')
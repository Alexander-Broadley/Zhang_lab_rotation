import torch
import pandas as pd
import numpy as np
import os
import sys

from torch.utils.data import DataLoader
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_building.customTFGE_dataset import CustomTFGE

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

MODEL_ROOT = '../models'
DATA_ROOT = '../data'

external_expressions = pd.read_csv(f'{DATA_ROOT}/Full data files/Liver_bulk_external.tsv', index_col = 0, sep = '\t')

#Load network
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
#Load target gene expressions
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/Geneexpression (full).tsv"), sep='\t', header=0)
#load TF expressions
TF_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/TF(full).tsv"), sep='\t', header=0)

#only TFs that are not in external dataset are SHOX and ZBED1, for now inserted these as zeros as Christian did
external_expressions['SHOX'] = 0
external_expressions['ZBED1'] = 0

#---------------------------------------------------------------
#filtering genes as in model creation pipeline
#---------------------------------------------------------------

#filter network to only include TFs that are in the dataset
net = net[net['TF'].isin(TF_expressions.columns)]

#filter genes to nodes in network
network_tfs = set(net['TF'].unique())      # TFs
network_genes = set(net['Gene'].unique())  # target genes
network_nodes = network_tfs | network_genes

#filter TF and gene expressions to only those in network
TF_expressions = TF_expressions[[gene for gene in TF_expressions.columns if gene in list(network_nodes)]]
gene_expressions = gene_expressions[[gene for gene in gene_expressions.columns if gene in list(network_nodes)]] 

orig_dataset_TFs = list(TF_expressions.columns)
orig_dataset_GEs = list(gene_expressions.columns)

#---------------------------------------------------------------
#get external dataset TF and gene expressions
#---------------------------------------------------------------

#subset to just TFs used when building model
external_TF = external_expressions[orig_dataset_TFs]
#only include target genes for which there exists a model
external_genes = external_expressions[[gene for gene in list(external_expressions.columns) if gene in orig_dataset_GEs]]

#refilter to only those in network now using external dataset
external_TF = external_TF[[gene for gene in external_TF.columns if gene in list(network_nodes)]]
external_genes = external_genes[[gene for gene in external_genes.columns if gene in list(network_nodes)]] 

#intialise training dataset predicted values df  - easier to do it this way as can get values after torch train-test split
train_predicted = pd.DataFrame(columns=gene_expressions.columns)
test_predicted = pd.DataFrame(columns=gene_expressions.columns)

#initialise training dataset actual values df
train_actual = pd.DataFrame(columns=gene_expressions.columns)
test_actual = pd.DataFrame(columns=gene_expressions.columns)

#---------------------------------------------------------------
#load models per target gene
#---------------------------------------------------------------

batch_size = TF_expressions.shape[0]

loss_fn = nn.MSELoss()

#initialise external dataset predicted values df
external_predicted = pd.DataFrame(index = external_expressions.index, columns= external_expressions.columns)

#intialise training dataset predicted values df
#train_predicted = pd.DataFrame(columns=gene_expressions.columns)
#test_predicted = pd.DataFrame(columns=gene_expressions.columns)

#initialise training dataset actual values df - easier to do it this way as can get values after torch train-test split
#train_actual = pd.DataFrame(columns=gene_expressions.columns)
#test_actual = pd.DataFrame(columns=gene_expressions.columns)

missing_models = []

def reverse_log_transorm(tensor_to_transform):
    #use torch method to reverse the Log(TPM+1) transform
    new_tensor = tensor_to_transform.expm1()
    return(new_tensor)

#set to true if getting predicted expressions for a log model
log_model = False

for target_gene in gene_expressions.columns:
    print(f'Generating scores for {target_gene} model')

    external_TFs = external_TF[TF_subset(net, target_gene)]
    TF_expression_subset = TF_expressions[TF_subset(net, target_gene)]

    #only run external scoring if the target gene is in the external dataset
    if target_gene in external_expressions.columns:
        external_dataset = CustomTFGE(device, TF_expressions=external_TFs, gene_expressions=external_expressions, network = net, target_gene = target_gene)
        eval_dataloader = DataLoader(external_dataset, batch_size=len(external_dataset), shuffle=False)

    #load desired models
    model = torch.load(f"{MODEL_ROOT}/pearsons_models/{target_gene}_model.pth", weights_only = False)
    
    #put model in eval mode
    model.eval()
    with torch.no_grad():
        #for each dataset then load the data (using same seed as training to get same train test split)
        #get the predicted and actual score this way as easiest way to get values from the torch train test split

        #only run if targt gene in external dataset
        if target_gene in external_expressions.columns:
            for batch, (X, y) in enumerate(eval_dataloader):     
                #only if running log model reverse the transformation to make error metrics comparable
                if log_model == True:
                    #create a prediction - this will be 1 value for given sample and target gene
                    external_predicted[target_gene] = reverse_log_transorm(model(X)).cpu()
                else:
                    external_predicted[target_gene] = model(X).cpu()

    

external_predicted.to_csv(f'{DATA_ROOT}/external_dataset_predicted_expressions_PEARSONS.csv')



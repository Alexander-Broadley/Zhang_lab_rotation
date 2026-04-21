import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch import nn

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_building.gene_MSE_all_samples import gene_MSE_all_samples
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
#update TF_expressions and gene expressions to be external dataset
#---------------------------------------------------------------

#subset to just TFs used when building model
TF_expressions = external_expressions[[TF for TF in list(TF_expressions.columns) if TF in orig_dataset_TFs]]
#only include target genes for which there exists a model
gene_expressions = external_expressions[[gene for gene in list(external_expressions.columns) if gene in orig_dataset_GEs]]

#refilter to only those in network now using external dataset
TF_expressions = TF_expressions[[gene for gene in TF_expressions.columns if gene in list(network_tfs)]]
gene_expressions = gene_expressions[[gene for gene in gene_expressions.columns if gene in list(network_genes)]] 

#---------------------------------------------------------------
#load models per target gene
#---------------------------------------------------------------

batch_size = TF_expressions.shape[0]

from model_building.pearsons_loss import PearsonLoss
loss_fn = PearsonLoss()

#initialise eval results df
results_df = pd.DataFrame(index = gene_expressions.columns, columns = ['external_set_score'])

missing_models = []

for target_gene in gene_expressions.columns:
    print(f'Generating scores for {target_gene} model')
    #print(f'Creating model for {target_gene}')
    TF_expression_subset = TF_expressions#[TF_subset(net, target_gene)]

    model = SimpleMMLModel(activation_function_map['MML'], TF_expression_subset.shape[1])
    model.to(device)

    dataset = CustomTFGE(device, TF_expressions=TF_expression_subset, gene_expressions=gene_expressions, network = net, target_gene = target_gene)
    #shouldn't need this anymore - fixed model saving issue
    try:
        model = torch.load(f"{MODEL_ROOT}/PEARSONS_200_randn_all/{target_gene}_model.pth", weights_only = False)
        #model = torch.load(f"{MODEL_ROOT}/pearsons_models/{target_gene}_TPM_model.pth", weights_only = False)
        eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        results_df.loc[target_gene, 'external_set_score'] = gene_MSE_all_samples(eval_dataloader, model, loss_fn, target_gene, rev_log = False)
    except:
        missing_models.append(target_gene)
        pass

print(f'{len(missing_models)} are missing')

results_df.to_csv('data/external_PEARSONS_200_randn_all.csv')

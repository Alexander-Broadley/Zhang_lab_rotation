import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch import nn

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from model_building.batch_loss import batch_loss
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
DATA_ROOT = '../../data'

print('Loading and filtering datasets')

external_expressions = pd.read_csv(f'{DATA_ROOT}/Full data files/Liver_bulk_external.tsv', index_col = 0, sep = '\t')

#imput missing genes as 0 - discuss w/ Cheng at next opportunity
#external_expressions['SHOX'] = 0
#external_expressions['ZBED1'] = 0
#external_expressions['ARNTL'] = 0
#external_expressions['HOMEZ'] = 0

#Load network
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
#Load ARCHS4 gene expressions
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/ARCHS4_healthy_log.tsv"), sep='\t', header=0)
#filter as in experimental models (using only genes in both datasets)
gene_expressions = gene_expressions[[gene for gene in gene_expressions.columns if gene in external_expressions.columns]]
external_expressions = external_expressions[gene_expressions.columns]

from model_building.filter_dataset import filter_datasets
orig_GE, orig_TF = filter_datasets(net, gene_expressions)
orig_GEs = list(orig_GE.columns)
orig_TFs = list(orig_TF.columns)

TF_expressions, gene_expressions = filter_datasets(net, external_expressions)

print(f'there are {len(orig_TFs)} TFs and {len(orig_GEs)} in the original dataset')
print(f'before TF has {TF_expressions.shape}')

'''
#external expression dataset is smaller than the original - for now add all as zeros but may need to retrain on less genes
for gene in orig_GEs:
    if gene not in gene_expressions.columns:
        gene_expressions[gene] = 0

for TF in orig_TFs:
    if TF not in TF_expressions.columns:
        TF_expressions[gene] = 0

TF_expressions = external_expressions[[tf for tf in TF_expressions.columns if tf in orig_TFs]]
gene_expressions = external_expressions[[gene for gene in gene_expressions.columns if gene in orig_GEs]]


print(f'now TF has {TF_expressions.shape}')

print(TF_expressions.shape)
'''
print(gene_expressions.shape)


#---------------------------------------------------------------
#load models per target gene
#---------------------------------------------------------------

print('Loading and scoring models on external dataset')

batch_size = TF_expressions.shape[0]

from model_building.pearsons_loss import PearsonLoss
loss_fn = PearsonLoss()

#initialise eval results df
results_df = pd.DataFrame(index = gene_expressions.columns, columns = ['external_loss'])

#initialise external dataset predicted values df
external_predicted = pd.DataFrame(index = external_expressions.index, columns= external_expressions.columns)

missing_models = []


for target_gene in gene_expressions.columns:
    print(f'Generating scores for {target_gene} model')

    if target_gene in TF_expressions.columns:
        TF_expression_subset = TF_expressions.drop(target_gene, axis = 1)
    else:
        TF_expression_subset = TF_expressions

    #model = SimpleMMLModel(activation_function_map['MML'], TF_expression_subset.shape[1])
    

    dataset = CustomTFGE(device, TF_expressions=TF_expression_subset, gene_expressions= gene_expressions, network = net, target_gene = target_gene)
    #shouldn't need this anymore - fixed model saving issue

    model = torch.load(f"{MODEL_ROOT}/experimental_models/{target_gene}_model.pth", weights_only = False)
    model.to(device)
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    results_df.loc[target_gene, 'external_loss'] = batch_loss(eval_dataloader, model, loss_fn)
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(eval_dataloader):     
            external_predicted[target_gene] = model(X).cpu()
    #except:
    #    print('model failed')
    #    missing_models.append(target_gene)
    #    pass

print(f'{len(missing_models)} are missing')

print('Writing results')
results_df.to_csv('data/external_HEALTHY_PCC.csv')
external_predicted.to_csv(f'{DATA_ROOT}/external_predicted_expressions_EXPERIMENTAL.csv')

import torch
import pandas as pd
import numpy as np
import os
import sys

from torch.utils.data import DataLoader
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from model_building.customTFGE_dataset import CustomTFGE
from model_building.batch_loss import batch_loss

#define device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#---------------------------------------------------------------
#Load Datasets
#---------------------------------------------------------------

MODEL_ROOT = '../../models'
DATA_ROOT = '../../../data'

print('Loading Datasets')
#Load network
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
#Load target gene expressions


#---------------------------------------------------------------
#filtering genes
#---------------------------------------------------------------

from model_building.filter_dataset import filter_datasets

#try new filtering function
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/ARCHS4_healthy_log.tsv"), sep='\t', header=0, index_col=0)

#load male and female metadata
female_sample_IDs = pd.read_csv(f'{DATA_ROOT}/Full data files/ARCHS4_female_external_meta.csv', index_col=0).index
male_sample_IDs = pd.read_csv(f'{DATA_ROOT}/Full data files/ARCHS4_male_external_meta.csv', index_col=0).index

#split the datasets into male and female before indexes are reset
female_expressions = gene_expressions.loc[female_sample_IDs]
male_expressions = gene_expressions.loc[male_sample_IDs]

female_TF_expressions, female_gene_expressions = filter_datasets(net, GE_df=female_expressions)
male_TF_expressions, male_gene_expressions = filter_datasets(net, GE_df=male_expressions)

assert list(female_TF_expressions.columns) == list(male_TF_expressions.columns)
assert list(female_gene_expressions.columns) == list(male_gene_expressions.columns)

#filter out male and female samples
print(female_TF_expressions.shape, female_gene_expressions.shape)
print(male_TF_expressions.shape, male_gene_expressions.shape)
#---------------------------------------------------------------
#load models per target gene
#---------------------------------------------------------------

fem_batch_size = female_TF_expressions.shape[0]
male_batch_size = male_TF_expressions.shape[0]

#load pearsons loss function for batch error scoring
from model_building.pearsons_loss import PearsonLoss
loss_fn = PearsonLoss()

#initialise external dataset predicted values df
results_df = pd.DataFrame(columns = ['female_PCC', 'male_PCC'], index = female_gene_expressions.columns)

#also record raw and predicted expressions for each sample
female_actual = pd.DataFrame(index = female_TF_expressions.index, columns= female_TF_expressions.columns)
female_predicted = pd.DataFrame(index = female_TF_expressions.index, columns= female_TF_expressions.columns)

male_actual = pd.DataFrame(index = male_TF_expressions.index, columns= male_TF_expressions.columns)
male_predicted = pd.DataFrame(index = male_TF_expressions.index, columns= male_TF_expressions.columns)


#just choose one of the target gene datasets - should both have the same columns
for target_gene in female_gene_expressions.columns:
    print(f'Generating scores for {target_gene} model')
    #if target gene is a TF for female will be a TF for male too
    if target_gene in female_TF_expressions.columns:
        #drop just for this target gene, not others
        female_TFs_to_use = female_TF_expressions.drop(target_gene, axis = 1)
        male_TFs_to_use = male_TF_expressions.drop(target_gene, axis = 1)

        #create dataset objects
        female_dataset = CustomTFGE(device, TF_expressions=female_TFs_to_use, gene_expressions=female_gene_expressions, network = net, target_gene = target_gene)
        male_dataset = CustomTFGE(device, TF_expressions=male_TFs_to_use, gene_expressions=male_gene_expressions, network = net, target_gene = target_gene)
    else:
        #if target gene not a TF then don't need to remove anything
        female_dataset = CustomTFGE(device, TF_expressions=female_TF_expressions, gene_expressions=female_gene_expressions, network = net, target_gene = target_gene)
        male_dataset = CustomTFGE(device, TF_expressions=male_TF_expressions, gene_expressions=male_gene_expressions, network = net, target_gene = target_gene)


    #wrap dataset objects in DataLoader iterable
    female_dataloader = DataLoader(female_dataset, batch_size=len(female_dataset), shuffle=False)
    male_dataloader = DataLoader(male_dataset, batch_size=len(male_dataset), shuffle=False)

    #load desired models
    model = torch.load(f"{MODEL_ROOT}/MF_external_models/{target_gene}_model.pth", weights_only = False)
    model.to(device)
    
    #print(f'Getting PCC for {target_gene} model')

    #for both dataloaders load all samples (batch size is the same as the length of the dataset), get the batch loss for each
    results_df.loc[target_gene, 'female_PCC'] = 1 - batch_loss(female_dataloader, model, loss_fn)   
    results_df.loc[target_gene, 'male_PCC'] = 1 - batch_loss(male_dataloader, model, loss_fn)

    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(female_dataloader):  
            female_actual[target_gene] = np.asarray(y.cpu())
            female_predicted[target_gene] = np.asarray(model(X).cpu())
        for batch, (X, y) in enumerate(male_dataloader):  
            male_actual[target_gene] = np.asarray(y.cpu())
            male_predicted[target_gene] = np.asarray(model(X).cpu())


male_predicted.to_csv(f'{DATA_ROOT}/male_predicted_gene_expressions.csv')
female_predicted.to_csv(f'{DATA_ROOT}/female_predicted_gene_expressions.csv')
male_actual.to_csv(f'{DATA_ROOT}/male_actual_gene_expressions.csv')
female_actual.to_csv(f'{DATA_ROOT}/female_actual_gene_expressions.csv')

results_df.to_csv(f'{DATA_ROOT}/female_male_PCC.csv')
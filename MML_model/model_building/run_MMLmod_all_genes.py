#converting notebook into python file for easier running
import os
import sys
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import tensor
import time
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#set seed for reproducibility 
torch.manual_seed(1475460913)

#import early stopping class
from model_building.early_stopper import EarlyStopping

#import LEMBAS activation functions
from model_building.activation_functions import activation_function_map
#import model
from model_building.simpleMMLModel import SimpleMMLModel
#import function to calculate loss per batch (in this case 1 batch is all samples)
from model_building.batch_loss import batch_loss
#import function to train a single epoch
from model_building.train_one_epoch import train_one_epoch
#import function to filter genes in expression dataset
from model_building.filter_dataset import filter_datasets
#import dataset object
from model_building.customTFGE_dataset import CustomTFGE

#assign a torch device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#define root directory
DATA_ROOT = '../../data'

print('Loading Datasets')
#Load network
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
#Load target gene expressions
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/ARCHS4_healthy_log_noFMExternal_norm.tsv"), sep='\t', header=0)

#filter expression dataset and split into TFs and TGs
TF_expressions, gene_expressions = filter_datasets(net, GE_df=gene_expressions)

female_expressions = pd.read_csv(f'{DATA_ROOT}/Full data files/ARCHS4_female_external_expressions_norm.tsv', index_col = 0, sep ='\t')
male_expressions = pd.read_csv(f'{DATA_ROOT}/Full data files/ARCHS4_male_external_expressions_norm.tsv', index_col = 0, sep ='\t')

print(female_expressions.head(n = 2))
print(male_expressions.head(n = 2))

female_TF_expressions, female_gene_expressions = filter_datasets(net, GE_df=female_expressions)
male_TF_expressions, male_gene_expressions = filter_datasets(net, GE_df=male_expressions)

#define training parameters
learning_rate = 0.0001
#set batch size to number of samples (dataset is small enough to allow this)
batch_size = TF_expressions.shape[0]
#max 200 epochs
epochs =  200

#import and assign custom pearsons loss function
from model_building.pearsons_loss import PearsonLoss
loss_fn = PearsonLoss()

#create a results dataframe - means don't have to re-calculate later
results_df = pd.DataFrame(index = gene_expressions.columns, columns = ['train_loss', 'test_loss', 'stopped_early', 'stopped_epoch','in_features', 'train_PCC', 'test_PCC', 'train_SRCC', 'test_SRCC', 'male_PCC', 'female_PCC', 'male_SRCC', 'female_SRCC'])

#create dataframes to track the predicted and actual expressions for train and test datasets - recreating datasets with pytorch is unreliable and cannot store 161000 datasets
#intialise training dataset predicted values df
train_predicted = pd.DataFrame(columns=gene_expressions.columns)
test_predicted = pd.DataFrame(columns=gene_expressions.columns)
male_predicted = pd.DataFrame(columns=gene_expressions.columns)
female_predicted = pd.DataFrame(columns=gene_expressions.columns)

#initialise training dataset actual values df - easier to do it this way as can get values after torch train-test split
train_actual = pd.DataFrame(columns=gene_expressions.columns)
test_actual = pd.DataFrame(columns=gene_expressions.columns)
male_actual = pd.DataFrame(columns=gene_expressions.columns)
female_actual = pd.DataFrame(columns=gene_expressions.columns)

print('TF expressions shape:', TF_expressions.shape)
print('Target gene expressions shape:', gene_expressions.shape)
print(f'There are {len(set(TF_expressions.columns) & set(gene_expressions.columns))} genes that are TFs and TGs')
print(f'There are {len(set(TF_expressions.columns))} TFs')
print(f'There are {len(set(gene_expressions.columns))}  TGs')

#then create a model per target gene
start_time = time.perf_counter()
for target_gene in gene_expressions.columns:
    print(f'Creating model for {target_gene}')
    
    #if target gene is also a TF remove it from the TF dataset to stop data leakage
    if target_gene in TF_expressions.columns:
        print('Target gene is a TF, removing from TF dataset')
        TF_expression_subset = TF_expressions.drop(target_gene, axis = 1)
        female_TF_expression_subset = female_TF_expressions.drop(target_gene, axis = 1)
        male_TF_expression_subset = male_TF_expressions.drop(target_gene, axis = 1)
    else:
         TF_expression_subset = TF_expressions
         female_TF_expression_subset = female_TF_expressions
         male_TF_expression_subset = male_TF_expressions

    #initialise an early stopper to end training if loss on test data does not fall by at least 0.01 for 3 eopochs in a row
    early_stopping = EarlyStopping(patience=3, delta=0.005, verbose=True)
    
    #initailise dataset
    dataset = CustomTFGE(device, TF_expressions=TF_expression_subset, gene_expressions=gene_expressions, network = net, target_gene = target_gene)
    #split into 80% train 20% test
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    female_dataset = CustomTFGE(device, TF_expressions=female_TF_expressions, gene_expressions=female_gene_expressions, network = net, target_gene = target_gene)
    male_dataset = CustomTFGE(device, TF_expressions=male_TF_expressions, gene_expressions=male_gene_expressions, network = net, target_gene = target_gene)

    #intialise model
    model = SimpleMMLModel(activation_function_map['MML'], TF_expression_subset.shape[1])

    #parallelise model if hardware allows
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    #put model on same device as dataset
    model.to(device)

    #initialise same optimiser as LEMBAS
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.00001)

    #create train and test dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    female_dataloader = DataLoader(female_dataset, batch_size=len(female_dataset), shuffle=False)
    male_dataloader = DataLoader(male_dataset, batch_size=len(male_dataset), shuffle=False)
    

    for t in range(epochs):
        #train one epoch, store train and test loss
        train_loss = train_one_epoch(train_dataloader, model, loss_fn, optimiser)
        test_loss = batch_loss(test_dataloader, model, loss_fn)

        if (t+1) % 5 == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            print(test_loss)

        #check if met eearly stopping criteria and stop if so
        early_stopping.check_early_stop(test_loss)
    
        if early_stopping.stop_training:
            print(f"Early stopping at epoch {t+1}")  
            results_df.loc[target_gene, 'stopped_early'] = 1
            results_df.loc[target_gene, 'stopped_epoch'] = t+1
            break

    #put model in eval mode
    model.eval()
    with torch.no_grad():
        #for each dataset then load the data (using same seed as training to get same train test split)
        #get the predicted and actual score this way as easiest way to get values from the torch train test split
        for batch, (X, y) in enumerate(train_dataloader):
            train_predicted[target_gene] = model(X).cpu()
            train_actual[target_gene] = y.cpu()

            results_df.loc[target_gene, 'train_PCC'] = scipy.stats.pearsonr(model(X).cpu(), y.cpu()).statistic
            results_df.loc[target_gene, 'train_SRCC'] = scipy.stats.spearmanr(model(X).cpu(), y.cpu()).statistic

        for batch, (X, y) in enumerate(test_dataloader):
            test_predicted[target_gene] = model(X).cpu()
            test_actual[target_gene] = y.cpu()

            results_df.loc[target_gene, 'test_PCC'] = scipy.stats.pearsonr(model(X).cpu(), y.cpu()).statistic
            results_df.loc[target_gene, 'test_SRCC'] = scipy.stats.spearmanr(model(X).cpu(), y.cpu()).statistic

        for batch, (X, y) in enumerate(female_dataloader):  
            female_actual[target_gene] = np.asarray(y.cpu())
            female_predicted[target_gene] = np.asarray(model(X).cpu())

            results_df.loc[target_gene, 'female_PCC'] = scipy.stats.pearsonr(model(X).cpu(), y.cpu()).statistic
            results_df.loc[target_gene, 'female_SRCC'] = scipy.stats.spearmanr(model(X).cpu(), y.cpu()).statistic

        for batch, (X, y) in enumerate(male_dataloader):  
            male_actual[target_gene] = np.asarray(y.cpu())
            male_predicted[target_gene] = np.asarray(model(X).cpu())

            results_df.loc[target_gene, 'male_PCC'] = scipy.stats.pearsonr(model(X).cpu(), y.cpu()).statistic
            results_df.loc[target_gene, 'male_SRCC'] = scipy.stats.spearmanr(model(X).cpu(), y.cpu()).statistic

    results_df.loc[target_gene, 'train_loss'] = train_loss
    results_df.loc[target_gene, 'test_loss'] = test_loss
    results_df.loc[target_gene, 'in_features'] = len(TF_expression_subset.columns)
    #save models
    torch.save(model, f'../models/log_norm_models/{target_gene}_model.pth')
end_time = time.perf_counter()

#save actual and predicted expression values
#train_actual.to_csv(f'{DATA_ROOT}/Train_dataset_actual_expressions_norm.csv')
#train_predicted.to_csv(f'{DATA_ROOT}/Train_dataset_predicted_expressions_norm.csv')

#test_predicted.to_csv(f'{DATA_ROOT}/Test_dataset_predicted_expressions_norm.csv')
#test_actual.to_csv(f'{DATA_ROOT}/Test_dataset_actual_expressions_norm.csv')


results_df.to_csv('../../data/complete_results.csv')
print(f'Models trained in: {end_time - start_time}')
print('Finished log norm  models')
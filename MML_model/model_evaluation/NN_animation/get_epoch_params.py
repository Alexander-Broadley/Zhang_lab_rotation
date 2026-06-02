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

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

#import early stopping class
from MML_model.model_building.early_stopper import EarlyStopping

#import LEMBAS activation functions
from MML_model.model_building.activation_functions import activation_function_map
#import model
from MML_model.model_building.simpleMMLModel import SimpleMMLModel
#import function to calculate loss per batch (in this case 1 batch is all samples)
from MML_model.model_building.batch_loss import batch_loss
#import function to train a single epoch
from MML_model.model_building.train_one_epoch import train_one_epoch

from MML_model.model_building.filter_dataset import filter_datasets
#import dataset object
from MML_model.model_building.customTFGE_dataset import CustomTFGE

temp = pd.read_csv('/Users/alexanderbroadley/Documents/PhD/Zhang Lab/Zhang_Lab_Code/data/MF_external_results.csv', index_col=0)
print(temp.head())
print(temp[temp['stopped_epoch'] == temp['stopped_epoch'].max()])
#OPALIN is the model that trains for the most epochs, 66 epochs



device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#define root directory
DATA_ROOT = '../../../data'

print('Loading Datasets')
#Load network
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
#Load target gene expressions
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/ARCHS4_healthy_log_noFMExternal.tsv"), sep='\t', header=0)

TF_expressions, gene_expressions = filter_datasets(net, GE_df=gene_expressions)
print(gene_expressions.head())

#define training parameters
learning_rate = 0.5
#run 1 sample at a time, but run through each sample per training epoch
batch_size = TF_expressions.shape[0]
#max 100 epochs
epochs =  200 #12747 * 5

#trying with new loss_fn
from MML_model.model_building.pearsons_loss import PearsonLoss
loss_fn = PearsonLoss()

target_gene = 'OPALIN'

if target_gene in TF_expressions.columns:
    print('Target gene is a TF, removing from TF dataset')
    TF_expression_subset = TF_expressions.drop(target_gene, axis = 1)
else:
    TF_expression_subset = TF_expressions

    #initialise an early stopper to end training if loss on test data does not fall by at least 0.01 MSE for 3 eopochs in a row
    early_stopping = EarlyStopping(patience=3, delta=0.005, verbose=True)

dataset = CustomTFGE(device, TF_expressions=TF_expression_subset, gene_expressions=gene_expressions, network = net, target_gene = target_gene)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = SimpleMMLModel(activation_function_map['MML'], TF_expression_subset.shape[1])

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to(device)

#initialise same optimiser as LEMBAS
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.00001)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

in_weights_by_epoch = pd.DataFrame(columns = list(range(0, 200)))
out_weights_by_epoch = pd.DataFrame( columns = list(range(0, 200)))

for t in range(epochs):
    model.eval()
    with torch.no_grad():
        in_weights_by_epoch[t] = model.linear_in.weight.cpu().flatten()
        out_weights_by_epoch[t] = model.linear_out.weight.cpu().flatten()

    print(t)
    model.train()
    train_loss = train_one_epoch(train_dataloader, model, loss_fn, optimiser)
    test_loss = batch_loss(test_dataloader, model, loss_fn)


print(in_weights_by_epoch)
in_weights_by_epoch.to_csv('./data/in_weights_OPALIN.csv')
out_weights_by_epoch.to_csv('./data/out_weights_OPALIN.csv')
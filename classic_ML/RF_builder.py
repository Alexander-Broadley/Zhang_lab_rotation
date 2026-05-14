#converting notebook into python file for easier running
import os
import sys
import matplotlib.pyplot as plt
import sklearn

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#set seed for reproducibility 
#torch.manual_seed(1475460913)

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

#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
#print(f"Using {device} device")

#define root directory
DATA_ROOT = '../data'

print('Loading Datasets')
#Load network
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
#Load target gene expressions

#try new filtering function
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/ARCHS4_healthy_log_noFMExternal.tsv"), sep='\t', header=0)

#for sensitivity analysis take 5000 random samples 
#gene_expressions = gene_expressions.sample(4000)
#print(f'Subsampled df shape is: {gene_expressions.shape}')

TF_expressions, gene_expressions = filter_datasets(net, GE_df=gene_expressions)

for target_gene in gene_expressions.columns[0:1]:
    print(f'Creating model for {target_gene}')
    
    #make TF_expression_subset an instance of TF_expressions to handle dropping a TF non-permanently if necessary
    TF_expression_subset = TF_expressions
    if target_gene in TF_expressions.columns:
        print('Target gene is a TF, removing from TF dataset')
        TF_expression_subset = TF_expressions.drop(target_gene, axis = 1)

    X_train, y_train, X_test, y_test = sklearn.model_selection.train_test_split(TF_expression_subset, gene_expressions[target_gene], random_state = 42, train_size = 0.8)
    print(X_train.shape, y_train.shape)

    RF_mod = sklearn.ensemble.RandomForestRegressor(n_estimators = 3)

    RF_mod.fit(X_train, y_train)

    y_pred = RF_mod.predict(X_test)

    print(y_pred)
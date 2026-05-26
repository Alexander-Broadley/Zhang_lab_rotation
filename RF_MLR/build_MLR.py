import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

from scipy.stats import pearsonr


#import relevant functions for deep learning approach
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from MML_model.model_building.filter_dataset import filter_datasets

DATA_ROOT = '../data'


gene_expressions = pd.read_csv(f'{DATA_ROOT}/Full data files/ARCHS4_healthy_log_noFMExternal.tsv', sep = '\t', index_col = 0)
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')

print(gene_expressions.head())

TF_expressions, gene_expressions = filter_datasets(net, GE_df=gene_expressions)
print(gene_expressions.shape)

results_df = pd.DataFrame(index = gene_expressions.columns, columns = ['train_loss', 'test_loss'])

for target in gene_expressions.columns[0:5]:
    print(f'Creating model for {target}')
    
    #drop TF from TF inputs to stop using a TFs expression to predict itself
    if target in TF_expressions.columns:
        print('Target gene is a TF, removing from TF dataset')
        TF_expression_subset = TF_expressions.drop(target, axis = 1)
    else:
        TF_expression_subset = TF_expressions

    X = TF_expression_subset
    y = gene_expressions[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42, stratify=y)

    regressor = LinearRegression(
                        n_jobs= 5,
    ).fit(X_train, y_train)


    #put PPC on train and test in a dataframe
    results_df.loc[target, 'train_loss'] = pearsonr(regressor.predict(X_train), y_train)
    results_df.loc[target, 'test_loss'] = pearsonr(regressor.predict(X_test), y_test)

results_df.to_csv('./RF_regressor_results.csv')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

#import relevant functions for deep learning approach
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from MML_model.model_building.filter_dataset import filter_datasets

DATA_ROOT = '../data'

gene_expressions = pd.read_csv(f'{DATA_ROOT}/Full data files/ARCHS4_healthy_log_noFMExternal.tsv', sep = '\t', index_col = 0, header = 0)
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')

TF_expressions, gene_expressions = filter_datasets(net, GE_df=gene_expressions)

RF_results_df = pd.DataFrame(index = gene_expressions.columns, columns = ['train_PCC', 'test_PCC', 'male_PCC', 'female_PCC'])
MLR_results_df = pd.DataFrame(index = gene_expressions.columns, columns = ['train_PCC', 'test_PCC', 'male_PCC', 'female_PCC'])

female_expressions = pd.read_csv(f'{DATA_ROOT}/Full data files/ARCHS4_female_external_expressions.tsv', index_col = 0, sep ='\t')
male_expressions = pd.read_csv(f'{DATA_ROOT}/Full data files/ARCHS4_male_external_expressions.tsv', index_col = 0, sep ='\t')

female_TF_expressions, female_gene_expressions = filter_datasets(net, GE_df=female_expressions)
male_TF_expressions, male_gene_expressions = filter_datasets(net, GE_df=male_expressions)


for target in gene_expressions.columns:
    print(f'Creating model for {target}')
    
    #drop TF from TF inputs to stop using a TFs expression to predict itself
    if target in TF_expressions.columns:
        print('Target gene is a TF, removing from TF dataset')
        TF_expression_subset = TF_expressions.drop(target, axis = 1)
        female_TF_expression_subset = female_TF_expressions.drop(target, axis = 1)
        male_TF_expression_subset = male_TF_expressions.drop(target, axis = 1)
    else:
        TF_expression_subset = TF_expressions
        female_TF_expression_subset = female_TF_expressions
        male_TF_expression_subset = male_TF_expressions

    X = TF_expression_subset
    y = gene_expressions[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

    rf = RandomForestRegressor(max_depth = 3, random_state = 42).fit(X_train, y_train)

    #put PPC on train and test in a dataframe
    RF_results_df.loc[target, 'train_PCC'] = pearsonr(rf.predict(X_train), y_train).statistic
    RF_results_df.loc[target, 'test_PCC'] = pearsonr(rf.predict(X_test), y_test).statistic

    RF_results_df.loc[target, 'male_PCC'] = pearsonr(rf.predict(male_TF_expression_subset), male_gene_expressions[target]).statistic
    RF_results_df.loc[target, 'female_PCC'] = pearsonr(rf.predict(female_TF_expression_subset), female_gene_expressions[target]).statistic
    #save the models
    from pickle import dump
    with open(f'./models/RF/{target}.pkl', 'wb') as f:
        dump(rf, f, protocol=5)

RF_results_df.to_csv('./data/RF_regressor_results.csv')
#RF and MLR trained in separate scripts for easiest parallelisation - ran one per screen
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

from scipy.stats import pearsonr
from scipy.stats import spearmanr

#import relevant functions for deep learning approach
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from MML_model.model_building.filter_dataset import filter_datasets

DATA_ROOT = '../data'
#load log normalised data
gene_expressions = pd.read_csv(f'{DATA_ROOT}/Full data files/ARCHS4_healthy_log_noFMExternal_norm.tsv', sep = '\t', index_col = 0, header = 0)
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
#filter dataset in same way as DL model
TF_expressions, gene_expressions = filter_datasets(net, GE_df=gene_expressions)

#initialise a results df
MLR_results_df = pd.DataFrame(index = gene_expressions.columns, columns = ['train_PCC', 'test_PCC', 'male_PCC', 'female_PCC', 'train_SP', 'test_SP', 'male_SP', 'female_SP'])

#load external male and female samples
female_expressions = pd.read_csv(f'{DATA_ROOT}/Full data files/ARCHS4_female_external_expressions_norm.tsv', index_col = 0, sep ='\t')
male_expressions = pd.read_csv(f'{DATA_ROOT}/Full data files/ARCHS4_male_external_expressions_norm.tsv', index_col = 0, sep ='\t')

#filter these as well
female_TF_expressions, female_gene_expressions = filter_datasets(net, GE_df=female_expressions)
male_TF_expressions, male_gene_expressions = filter_datasets(net, GE_df=male_expressions)

start_time = time.perf_counter()
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

    #create 80/20 split train test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

    #fit the regressor to training data
    regressor = LinearRegression().fit(X_train, y_train)

    #record spearmans and pearsons correlation across the 4 datasets
    MLR_results_df.loc[target, 'train_PCC'] = pearsonr(regressor.predict(X_train), y_train).statistic
    MLR_results_df.loc[target, 'test_PCC'] = pearsonr(regressor.predict(X_test), y_test).statistic

    MLR_results_df.loc[target, 'train_SP'] = spearmanr(regressor.predict(X_train), y_train).statistic
    MLR_results_df.loc[target, 'test_SP'] = spearmanr(regressor.predict(X_test), y_test).statistic

    MLR_results_df.loc[target, 'male_PCC'] = pearsonr(regressor.predict(male_TF_expression_subset), male_gene_expressions[target]).statistic
    MLR_results_df.loc[target, 'female_PCC'] = pearsonr(regressor.predict(female_TF_expression_subset), female_gene_expressions[target]).statistic

    MLR_results_df.loc[target, 'male_SP'] = spearmanr(regressor.predict(male_TF_expression_subset), male_gene_expressions[target]).statistic
    MLR_results_df.loc[target, 'female_SP'] = spearmanr(regressor.predict(female_TF_expression_subset), female_gene_expressions[target]).statistic

    #save the model
    from pickle import dump
    with open(f'./models/MLR_norm/{target}.pkl', 'wb') as f:
        dump(regressor, f, protocol=5)
end_time = time.perf_counter()

print(f'Models trained in: {end_time - start_time}')

#save the results
MLR_results_df.to_csv('./data/MLR_regressor_results_norm.csv')
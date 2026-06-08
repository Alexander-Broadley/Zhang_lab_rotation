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

print(list(male_gene_expressions).index('LBR'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_building.filter_dataset import filter_datasets

DATA_ROOT = '../../data'

net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/ARCHS4_healthy_log_noFMExternal.tsv"), sep='\t', header=0)
TF_expressions, gene_expressions = filter_datasets(net, GE_df=gene_expressions)

#load actual expressions
actual_female_expressions = pd.read_csv('/home/alexanderb/Zhang_lab/data/Full data files/ARCHS4_female_external_expressions.tsv', sep ='\t', index_col = 0)
actual_male_expressions = pd.read_csv('/home/alexanderb/Zhang_lab/data/Full data files/ARCHS4_male_external_expressions.tsv', sep ='\t', index_col = 0)

#create list of DFs to join at the end to minimise memory duplications
df_list = []

#specify abs difference in relative contribution required to consider a TF a differential regulator of a TG
threshold = 0.001

for target in gene_expressions.columns:
    print(f'Processing {target} scores')

    results_df = pd.DataFrame(columns = ['TF','TG','abs_diff', 'Male_mean_exp', 'Female_mean_exp'])

    #for now manually load dist for one target gene
    normalised_cont_differences = pd.read_csv(f'./data/differential_relative_effect_sizes/{target}_DRE.csv', index_col = 0)
    normalised_cont_differences = normalised_cont_differences[normalised_cont_differences["abs_mean_cont_diff"] > threshold]

    TFs_to_add = list(normalised_cont_differences.index)
    results_df['TF'] = TFs_to_add
    results_df['TG'] = target
    results_df['Male_mean_exp'] = np.mean(actual_male_expressions[target])
    results_df['Female_mean_exp'] = np.mean(actual_female_expressions[target])

    print(results_df)

    df_list.append(results_df)

final_results = pd.concat(df_list, axis = 0)

    
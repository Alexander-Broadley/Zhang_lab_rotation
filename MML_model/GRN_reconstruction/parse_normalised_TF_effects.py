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
gene_expressions = pd.read_csv((f"/home/alexanderb/Zhang_lab/data/Full data files/ARCHS4_healthy_log_noFMExternal.tsv"), sep='\t', header=0)
TF_expressions, gene_expressions = filter_datasets(net, GE_df=gene_expressions)

#load actual expressions
actual_female_expressions = pd.read_csv('/home/alexanderb/Zhang_lab/data/Full data files/ARCHS4_female_external_expressions.tsv', sep ='\t', index_col = 0)
actual_male_expressions = pd.read_csv('/home/alexanderb/Zhang_lab/data/Full data files/ARCHS4_male_external_expressions.tsv', sep ='\t', index_col = 0)

#create list of DFs to join at the end to minimise memory duplications
df_list = []

#specify abs difference in relative contribution required to consider a TF a differential regulator of a TG
#for threshold in [0.000000001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3]:
for threshold in [5, 10, 20, 50, 100, 200]:
    print(f'Processing across target genes for a threshold: {threshold}')

    for target in gene_expressions.columns:
        #print(f'Processing {target} scores')

        results_df = pd.DataFrame(columns = ['TF','TG','abs_diff', 'Male_target_mean_exp', 'Female_target_mean_exp', 'target_exp_difference', 'Male_TF_mean_exp', 'Female_TF_mean_exp', 'TF_exp_difference'])

        #for now manually load dist for one target gene
        normalised_cont_differences = pd.read_csv(f'./data/differential_relative_effect_sizes/{target}_DRE.csv', index_col = 0)
        

        results_df = normalised_cont_differences[normalised_cont_differences["abs_mean_cont_diff"] > threshold]
        print(normalised_cont_differences.shape)
        print(results_df.shape)
        results_df['TF'] = list(results_df.index)
        results_df.index = results_df['TF']
        results_df['TG'] = target
        results_df['abs_diff'] = np.asarray(results_df['abs_mean_cont_diff'])
        results_df['Male_target_mean_exp'] = np.mean(actual_male_expressions[target])
        results_df['Female_target_mean_exp'] = np.mean(actual_female_expressions[target])
        results_df['target_exp_difference'] = abs(results_df['Male_target_mean_exp'] - results_df['Female_target_mean_exp'])

        for TF in results_df.index:
            results_df.loc[TF, 'Male_TF_mean_exp']  = np.mean(actual_male_expressions[TF])
            results_df.loc[TF, 'Female_TF_mean_exp'] = np.mean(actual_female_expressions[TF])
            results_df.loc[TF, 'TF_exp_difference']  = abs(results_df.loc[TF, 'Male_TF_mean_exp'] - results_df.loc[TF, 'Female_TF_mean_exp'])

        df_list.append(results_df)

    final_results = pd.concat(df_list, axis = 0, ignore_index=True)
    final_results.to_csv(f'./data/GW_diff_regulatorts_{threshold}.csv')
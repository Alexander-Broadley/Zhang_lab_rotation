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
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/ARCHS4_healthy_log_noFMExternal_norm.tsv"), sep='\t', header=0)
TF_expressions, gene_expressions = filter_datasets(net, GE_df=gene_expressions)

for target in gene_expressions.columns:
    print(f'Processing {target} scores')
    #for now manually load dist for one target gene
    male_target_contribs = pd.read_csv(f'./data/male_contribution_dists_norm/{target}_reg_cont_distributions.csv', index_col = 0)
    female_target_contribs = pd.read_csv(f'./data/female_contribution_dists_norm/{target}_reg_cont_distributions.csv', index_col = 0)
    #iterate through TFs in male_target_contribs (same as in female)
    results_df = pd.DataFrame(columns = ['abs_mean_cont_diff'])
    for TF in male_target_contribs.columns:
        #take the difference between the two columns
        sex_differential = abs(np.mean(male_target_contribs[TF]) - np.mean(female_target_contribs[TF]))
        results_df.loc[TF, 'abs_mean_cont_diff'] = sex_differential
    results_df.to_csv(f'./data/differential_relative_effect_sizes_norm/{target}_DRE_norm.csv')
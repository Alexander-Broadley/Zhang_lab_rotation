import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

threshold = 0.1
file_path = f'./data/GW_diff_regulatorts_{threshold}.csv'
results = pd.read_csv(file_path, index_col = 0)

#remove any infinite values (caused by zero division when expression values where ~0) - checked for maximum absolute diff before this which was ~200
results = results[results['abs_diff'] <= 1000000]

results = results[results['abs_diff'] > threshold]

#overwrite results file with infinite values removed
results.to_csv(file_path)

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(ncols = 1, figsize = (7, 5))
sns.scatterplot(results, x = 'Male_TF_mean_exp', y = 'Female_TF_mean_exp', s = 2, color = '#1b2fb3', legend = False, ax = ax)
plt.ylabel('Mean TF Expression in Female Samples')
plt.xlabel('Mean TF Expression in Male Samples')
plt.savefig(f'./figures/mean_TF_expressions_{threshold}.png', dpi = 300)

fig, ax = plt.subplots(ncols = 1, figsize = (7, 5))
sns.scatterplot(results, x = 'Male_target_mean_exp', y = 'Female_target_mean_exp', s = 2, color = "#0a7126", legend = False, ax = ax)
plt.ylabel('Mean TG Expression in Female Samples')
plt.xlabel('Mean TG Expression in Male Samples')
plt.savefig(f'./figures/mean_TG_expressions_{threshold}.png', dpi = 300)

fig, ax = plt.subplots(ncols = 1, figsize = (7, 5))
((results['TF'].value_counts()/15199)*100).plot(kind = 'hist', color = '#1b2fb3', bins = 10, ax = ax)
ax.set_xlabel('Percentage of Target Genes Differentially Regulated per TF', fontsize = 10)
ax.set_ylabel('Frequency', fontsize = 10)
plt.savefig(f'./figures/per_TF_percent_diff_regged_{threshold}.png', dpi = 300)

fig, ax = plt.subplots(ncols = 1, figsize = (7, 5))
sns.scatterplot(results, x = 'TF_exp_difference', y = 'abs_diff', s = 1.5, color = '#1b2fb3', legend = False, ax = ax)
plt.ylabel('Absolute Difference in Normalised Contribution')
plt.xlabel('Absolute Difference in Mean TF Expression')
plt.savefig(f'./figures/abs_diff_contribution_expression_{threshold}.png', dpi = 300)

diff_regulated = set(results['TG'])
diff_regulators = set(results['TF'])

with open(f'./data/{threshold}_differential_regulated.txt', 'w+') as f:
    for gene in diff_regulated:
        f.write(f'{gene}\n')
f.close

with open(f'./data/{threshold}_differential_regulators.txt', 'w+') as f:
    for gene in diff_regulators:
        f.write(f'{gene}\n')
f.close

#get target gene specific info at given threshold

for target in ['MBOAT7', 'CYP7B1', 'JUND']:
    target_res = results[results['TF'] == target]
    target_res.to_csv(f'./data/{target}_diff_reg_{threshold}.csv')
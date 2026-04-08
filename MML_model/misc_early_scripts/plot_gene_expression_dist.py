import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

DATA_ROOT = './data'

gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/Geneexpression (full).tsv"), sep='\t', header=0)

merged_results = pd.read_csv((f"{DATA_ROOT}/MSE_results_TPM.csv"), header=0, index_col=0)
merged_results['mean_ex'] = 0.0

for gene in merged_results.index:
    merged_results.loc[gene, 'mean_ex'] = np.mean(np.asarray(gene_expressions[gene]))

merged_results.to_csv(f'{DATA_ROOT}/TPM_res_with_MeanEx.csv')

sns.scatterplot(x = merged_results['test_score'], y= merged_results['mean_ex'], s=7, color = 'darkblue')
plt.xlabel('Test Dataset MSE')
plt.ylabel('Mean Gene Expression')
plt.show()
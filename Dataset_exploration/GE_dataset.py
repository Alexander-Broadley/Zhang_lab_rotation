# plots to check distribution of TF dataset - should already be TPM normalised and log10 transformed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#change this to folder in which 'Full data files' folder is in
DATA_ROOT = '/home/alexanderb/LEMBAS-RNN-benchmark'

#load TF dataset
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/Geneexpression (full).tsv"), sep='\t', header=0)

fig, ax = plt.subplots(ncols=1)

ax = plt.hist(gene_expressions.to_numpy().flatten(), color='red')
fig.suptitle('Gene Expression values distribution')

print(f'There are {len(gene_expressions.to_numpy().flatten())} expression values in the dataset')

plt.savefig('../figures/GE_express_dist')
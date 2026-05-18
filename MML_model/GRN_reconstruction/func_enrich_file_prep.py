import sys
import os
import pandas as pd
import numpy as np

DATA_ROOT = '/Users/alexanderbroadley/Documents/PhD/Zhang Lab/Zhang_Lab_Code/data'

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_building.filter_dataset import filter_datasets

#Load datasets

net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')

#try new filtering function
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/ARCHS4_healthy_log_noFMExternal.tsv"), sep='\t', header=0, index_col=0)
print(gene_expressions.head)

TF_expressions, gene_expressions = filter_datasets(net, GE_df=gene_expressions)

#get list of background genes
background_TGs = set(gene_expressions.columns)
background_TFs = set(TF_expressions.columns)

background_genes = background_TFs | background_TGs

print(background_genes)
print(len(background_genes), len(background_TFs), len(background_TGs))


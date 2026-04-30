#code adapted from pyDESEQ tutorial: https://pydeseq2.readthedocs.io/en/latest/auto_examples/plot_minimal_pydeseq2_pipeline.html
import os
import pickle as pkl

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data

SAVE = True 
if SAVE:
    OUTPUT_PATH = "./data"
    os.makedirs(OUTPUT_PATH, exist_ok=True)

import pandas as pd
import numpy as np

DATA_ROOT = '../../../data'
fem_meta = pd.read_csv(f'{DATA_ROOT}/ARCHS4_female_healthy_meta.csv', index_col=0)
male_meta = pd.read_csv(f'{DATA_ROOT}/ARCHS4_male_healthy_meta.csv', index_col=0)

Gene_expression_data = pd.read_csv(f'{DATA_ROOT}/Full data files/ARCHS4_healthy_RAW.tsv', sep = '\t', index_col=0)

#deseq needs samples as row index
counts_df = Gene_expression_data

#put metadata in correct format for DESEQ (sample IDs are row index, column with condition (here sex) 
fem_meta['condition'] = 'F'
male_meta['condition'] = 'M'

fem_concat = fem_meta['condition'] 
male_concat = male_meta['condition']
concat_list = [fem_concat, male_concat]

#convert from a series after vertically joining
meta_df = pd.DataFrame(pd.concat(concat_list))


#filter samples and gene data as per tutorial
samples_to_keep = ~meta_df.condition.isna()
metadata = meta_df.loc[samples_to_keep]


#subset counts to just the male and female samples
print(counts_df.head())
counts_df = counts_df.loc[meta_df.index]

genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
counts_df = counts_df[genes_to_keep]

inference = DefaultInference(n_cpus=8)
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~condition",
    refit_cooks=True,
    inference=inference,
    # n_cpus=8, # n_cpus can be specified here or in the inference object
)



dds.deseq2()


ds = DeseqStats(dds, contrast=["condition", "F", "M"], inference=inference)
ds.summary()


if SAVE:
    with open(os.path.join(OUTPUT_PATH, "dds.pkl"), "wb") as f:
        pkl.dump(dds, f)
    print('Saved base dds object')

if SAVE:
    with open(os.path.join(OUTPUT_PATH, "ds.pkl"), "wb") as f:
        pkl.dump(ds, f)
    print('Saved DGE summary object')
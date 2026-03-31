import torch
import pandas as pd
import numpy as np

DATA_ROOT = './data'
MODEL_ROOT = './models'

merged_results = pd.read_csv(f'{DATA_ROOT}/merged_results.csv', index_col= 0, header = 0)
merged_results['in_features'] = 0

for gene in merged_results.index:
    model = torch.load(f'{MODEL_ROOT}/TPM_models/{gene}_TPM_model.pth', weights_only = False)
    merged_results.loc[gene, 'in_features'] = model.linear_in.in_features

merged_results.to_csv(f'{DATA_ROOT}/merged_results.csv')
import torch
import pandas as pd
import numpy as np

DATA_ROOT = './data'
MODEL_ROOT = './models'

TPM_res = pd.read_csv(f'{DATA_ROOT}/MSE_results_TPM.csv', index_col= 0, header = 0)
TPM_res['in_features'] = 0

for gene in TPM_res.index:
    model = torch.load(f'{MODEL_ROOT}/TPM_models/{gene}_TPM_model.pth', weights_only = False)
    TPM_res.loc[gene, 'in_features'] = model.linear_out.in_features
print(TPM_res)

TPM_res.to_csv(f'{DATA_ROOT}/TPM_res_in_features.csv')
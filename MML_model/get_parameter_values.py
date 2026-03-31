import torch
import pandas as pd
import numpy as np
from torch import nn

DATA_ROOT = './data'
MODEL_ROOT = './models'

merged_results = pd.read_csv(f'{DATA_ROOT}/merged_results.csv', index_col= 0, header = 0)
merged_results['in_features'] = 0

for gene in merged_results.index[0:1]:
    print('New gene \n')
    model = torch.load(f'{MODEL_ROOT}/TPM_models/{gene}_TPM_model.pth', weights_only = False)
    print(model)
    print(model.linear_in.weight.data)
    for parameter in model.parameters():
        pass
        
        #print(parameter.data.shape)
        #print(parameter.data)

#merged_results.to_csv(f'{DATA_ROOT}/merged_results.csv')
import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import DeepLift
from captum.attr import visualization as viz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


from model_building.filter_dataset import filter_datasets

#define root directory
DATA_ROOT = '/home/alexanderb/Zhang_lab/data'

print('Loading Datasets')
#Load network
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
#Load target gene expressions


#gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/Geneexpression (full).tsv"), sep='\t', header=0)
gene_expressions = pd.read_csv((f"{DATA_ROOT}/archs4/ARCHS4_healthy_TPM_stricter.tsv"), sep='\t', header=0, index_col= 0)
gene_expressions.reset_index()
gene_expressions = gene_expressions + 1
gene_expressions = np.log10(gene_expressions)

#load TF expressions
#TF_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/TF(full).tsv"), sep='\t', header=0)
TF_expressions = pd.read_csv((f"{DATA_ROOT}/archs4/ARCHS4_healthy_TPM_stricter.tsv"), sep='\t', header=0, index_col=0)
TF_expressions.reset_index()
TF_expressions = TF_expressions + 1
TF_expressions = np.log10(TF_expressions)


gene_expressions = gene_expressions.T
TF_expressions = TF_expressions.T

#filter network to only include TFs that are in the dataset
net = net[net['TF'].isin(TF_expressions.columns)]

print('Filtering genes in datasets')
#filter genes to nodes in network
network_tfs = set(net['TF'].unique())      # TFs
network_genes = set(net['Gene'].unique())  # target genes
#network_nodes = network_tfs | network_genes
print(TF_expressions)
TF_expressions = TF_expressions[[gene for gene in TF_expressions.columns if gene in list(network_tfs)]]
gene_expressions = gene_expressions[[gene for gene in gene_expressions.columns if gene in list(network_genes)]] 


MODEL_ROOT = '../../model_building/models'
model = torch.load(f'/home/alexanderb/Zhang_lab/MML_model/models/PEARSONS_L2/MBOAT7_model.pth', weights_only = False)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model.eval()
model.to(device)

one_sample = TF_expressions.iloc[500, :]
two_sample = TF_expressions.iloc[501, :]

male_tensor = torch.tensor([np.asarray(one_sample)], dtype = torch.float32).to(device)
female_tensor = torch.tensor([np.asarray(two_sample)], dtype = torch.float32).to(device)

deep_lift = DeepLift(model)
attributions = deep_lift.attribute(male_tensor, baselines=female_tensor)

features = TF_expressions
attr = attributions.detach().cpu().numpy()[0]
plt.bar(features, attr, color='darkblue')
plt.title("DeepLIFT Attributions: MBOAT7")
plt.ylabel("Attribution Score (Zeros vs Sample 1100)")
plt.xlabel("Input Features")
plt.xticks(rotation = 90, fontsize = 9)
plt.show()
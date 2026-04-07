import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import DeepLift
from captum.attr import visualization as viz
import numpy as np
import matplotlib.pyplot as plt

from filter_dataset import filter_datasets

#adapted code from DEEPLIFT tutorial: https://medium.com/@pysquad/deeplift-explained-python-techniques-for-ai-transparency-93d7ca64832d

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

def TF_subset(net, target_gene):
    #simpler version as per discussion w/ Cheng
    #returns all the TFs in the network that directly connect to the target gene
    return(list(net['TF'][net['Gene'] == target_gene])) 

MODEL_ROOT = './models'
DATA_ROOT = './data'

TF_expressions, net = filter_datasets(DATA_ROOT, load_genes=False)

#run same target gene specific pre-processing (retrieves list of genes that will correspond to the input features)
TF_expression_subset = list(TF_expressions[TF_subset(net, 'MBOAT7')].columns)

#take one sample of TF expression subset
one_sample = TF_expressions[TF_expression_subset].iloc[1100, :]


model = torch.load(f'{MODEL_ROOT}/TPM_models/MBOAT7_TPM_model.pth', weights_only = False)
model.eval()
model.to(device)

print(torch.tensor(np.asarray(one_sample), dtype = torch.float32).to(device))

#with torch.no_grad():
#    print(model(torch.tensor(np.asarray(one_sample), dtype = torch.float32).to(device)))


inputs = torch.tensor([np.asarray(one_sample)], dtype = torch.float32).to(device)
#inputs = torch.tensor([[0.5, 1.0, 0.0 , 1.0, 2.0, 3.0, 5.0, 1.0, 0.5, 0.85, 0.1, 0.7, 0.6]], dtype=torch.float32).to(device)
#inputs = torch.tensor([[1.8607, 1.3876, 0.9777, 1.5550, 1.4897, 0.9294, 1.7975, 0.4720, 1.1801, 1.3441, 0.0327, 1.3613, 1.1974]], dtype = torch.float32).to(device)

#baseline = torch.tensor([[0.0, 0.0, 0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32).to(device)
baseline = torch.tensor([np.asarray([0.0] * len(TF_expression_subset)) +1], dtype=torch.float32).to(device)
print(baseline)
print(inputs.shape)


deep_lift = DeepLift(model)
attributions = deep_lift.attribute(inputs, baselines=baseline)

features = TF_expression_subset
attr = attributions.detach().cpu().numpy()[0]
plt.bar(features, attr, color='darkblue')
plt.title("DeepLIFT Attributions: MBOAT7")
plt.ylabel("Attribution Score (Zeros vs Sample 1100)")
plt.xlabel("Input Features")
plt.xticks(rotation = 90, fontsize = 9)
plt.show()
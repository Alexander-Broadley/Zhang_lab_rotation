import torch
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

#import relevant functions
from MML_model.model_building.customTFGE_dataset import CustomTFGE
from MML_model.model_building.gene_MSE_all_samples import gene_MSE_all_samples

#define device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


#model = torch.load('data/model.pth', weights_only=False)

#set seed for reproducibility 
torch.manual_seed(1475460913)

#define file with models in
models_path = '/home/alexanderb/Zhang_lab/MML_model/models/TPM_models'

#define data root directory
DATA_ROOT = '/home/alexanderb/LEMBAS-RNN-benchmark'

### Recreate same data preprocessing as when creating model so can evaluate fairly

#load datasets
print('Loading Datasets')
#Load network
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
print('Loaded Network')
#Load target gene expressions
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/Geneexpression (full).tsv"), sep='\t', header=0)
print('Loaded Gene Expressions')
#load TF expressions
print('Loaded TF expressions')
TF_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/TF(full).tsv"), sep='\t', header=0)


print('Filtering genes in datasets')

#filter network to only include TFs that are in the dataset
net = net[net['TF'].isin(TF_expressions.columns)]
#filter genes to nodes in network
network_tfs = set(net['TF'].unique())      # TFs
network_genes = set(net['Gene'].unique())  # target genes
network_nodes = network_tfs | network_genes

TF_expressions = TF_expressions[[gene for gene in TF_expressions.columns if gene in list(network_nodes)]]
gene_expressions = gene_expressions[[gene for gene in gene_expressions.columns if gene in list(network_nodes)]] 

#define function to subset transcription factors to only those that directly regulate the target gene
def TF_subset(net, target_gene):
    #simpler version as per discussion w/ Cheng
    #returns all the TFs in the network that directly connect to the target gene
    return(list(net['TF'][net['Gene'] == target_gene]))

#define loss fn and batch size (other hyperparams relevant to training only)
loss_fn = nn.MSELoss()
batch_size = TF_expressions.shape[0]

#create a results dataframe
results_df = pd.DataFrame(index = gene_expressions.columns, columns = ['train_score', 'test_score'])

for target_gene in list(gene_expressions.columns)[0:5]:
    print(target_gene)
    #load model for target gene
    model_path = f"{models_path}/{target_gene}_TPM_model.pth"

    #subset to TFs relevant for target gene
    TF_expression_subset = TF_expressions[TF_subset(net, target_gene)]

    dataset = CustomTFGE(device, TF_expressions=TF_expression_subset, gene_expressions=gene_expressions, network = net, target_gene = target_gene)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    model = torch.load(model_path, weights_only=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    results_df.loc[target_gene, 'train_score'] = gene_MSE_all_samples(test_dataloader, model, loss_fn, target_gene)
    results_df.loc[target_gene, 'test_score'] = gene_MSE_all_samples(train_dataloader, model, loss_fn, target_gene)

results_df.to_csv('data/recon_MSE_results_TPM.csv')
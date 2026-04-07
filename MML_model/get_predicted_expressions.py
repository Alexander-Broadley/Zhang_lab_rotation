import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch import nn

from gene_MSE_all_samples import gene_MSE_all_samples
from customTFGE_dataset import CustomTFGE

#define device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#define function to subset transcription factors to only those that directly regulate the target gene
def TF_subset(net, target_gene):
    #simpler version as per discussion w/ Cheng
    #returns all the TFs in the network that directly connect to the target gene
    return(list(net['TF'][net['Gene'] == target_gene]))

#---------------------------------------------------------------
#Load Datasets
#---------------------------------------------------------------

print('Loading Datasets')

MODEL_ROOT = './models'
DATA_ROOT = './data'

external_expressions = pd.read_csv(f'{DATA_ROOT}/Full data files/Liver_bulk_external.tsv', index_col = 0, sep = '\t')

#Load network
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
#Load target gene expressions
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/Geneexpression (full).tsv"), sep='\t', header=0)
#load TF expressions
TF_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/TF(full).tsv"), sep='\t', header=0)

#only TFs that are not in external dataset are SHOX and ZBED1, for now inserted these as zeros as Christian did
external_expressions['SHOX'] = 0
external_expressions['ZBED1'] = 0

#---------------------------------------------------------------
#filtering genes as in model creation pipeline
#---------------------------------------------------------------

#filter network to only include TFs that are in the dataset
net = net[net['TF'].isin(TF_expressions.columns)]

#filter genes to nodes in network
network_tfs = set(net['TF'].unique())      # TFs
network_genes = set(net['Gene'].unique())  # target genes
network_nodes = network_tfs | network_genes

#filter TF and gene expressions to only those in network
TF_expressions = TF_expressions[[gene for gene in TF_expressions.columns if gene in list(network_nodes)]]
gene_expressions = gene_expressions[[gene for gene in gene_expressions.columns if gene in list(network_nodes)]] 

orig_dataset_TFs = list(TF_expressions.columns)
orig_dataset_GEs = list(gene_expressions.columns)

#---------------------------------------------------------------
#get external dataset TF and gene expressions
#---------------------------------------------------------------

#subset to just TFs used when building model
external_TF = external_expressions[orig_dataset_TFs]
#only include target genes for which there exists a model
external_genes = external_expressions[[gene for gene in list(external_expressions.columns) if gene in orig_dataset_GEs]]

#refilter to only those in network now using external dataset
external_TF = external_TF[[gene for gene in external_TF.columns if gene in list(network_nodes)]]
external_genes = external_genes[[gene for gene in external_genes.columns if gene in list(network_nodes)]] 

#---------------------------------------------------------------
#load models per target gene
#---------------------------------------------------------------

batch_size = TF_expressions.shape[0]

loss_fn = nn.MSELoss()

#initialise external dataset predicted values df
external_predicted = pd.DataFrame(index = external_expressions.index, columns= external_expressions.columns)

#intialise training dataset predicted values df
train_predicted = pd.DataFrame(columns=gene_expressions.columns)
test_predicted = pd.DataFrame(columns=gene_expressions.columns)

#initialise training dataset actual values df - easier to do it this way as can get values after torch train-test split
train_actual = pd.DataFrame(columns=gene_expressions.columns)
test_actual = pd.DataFrame(columns=gene_expressions.columns)

missing_models = []

def reverse_log_transorm(tensor_to_transform):
    new_tensor = tensor_to_transform.expm1()
    return(new_tensor)

#set to true if getting predicted expressions for a log model
log_model = False

for target_gene in gene_expressions.columns:
    print(f'Generating scores for {target_gene} model')

    external_TFs = external_TF[TF_subset(net, target_gene)]
    TF_expression_subset = TF_expressions[TF_subset(net, target_gene)]


    if target_gene in external_expressions.columns:
        external_dataset = CustomTFGE(device, TF_expressions=external_TFs, gene_expressions=external_expressions, network = net, target_gene = target_gene)
        eval_dataloader = DataLoader(external_dataset, batch_size=len(external_dataset), shuffle=False)

    original_dataset = CustomTFGE(device, TF_expressions=TF_expression_subset, gene_expressions=gene_expressions, network = net, target_gene = target_gene)
    train_dataset, test_dataset = torch.utils.data.random_split(original_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    model = torch.load(f"{MODEL_ROOT}/logTPM_models/{target_gene}_logTPM_model.pth", weights_only = False)
    

    model.eval()
    with torch.no_grad():
        if target_gene in external_expressions.columns:
            for batch, (X, y) in enumerate(eval_dataloader):     
                #only if running log model reverse the transformation to make error metrics comparable
                if log_model == True:
                    #create a prediction - this will be 1 value for given sample and target gene
                    external_predicted[target_gene] = reverse_log_transorm(model(X)).cpu()
                else:
                    external_predicted[target_gene] = model(X).cpu()
            
        for batch, (X, y) in enumerate(train_dataloader):
            
             #only if running log model reverse the transformation to make error metrics comparable
            if log_model == True:
                train_predicted[target_gene] = reverse_log_transorm(model(X)).cpu()
                train_actual[target_gene] = y.cpu()

            else:
                train_predicted[target_gene] = model(X).cpu()
                train_actual[target_gene] = y.cpu()


        for batch, (X, y) in enumerate(test_dataloader):
             #only if running log model reverse the transformation to make error metrics comparable
            if log_model == True:
                test_predicted[target_gene] = reverse_log_transorm(model(X)).cpu()
                test_actual[target_gene] = y.cpu()
            else:
                test_predicted[target_gene] = model(X).cpu()
                test_actual[target_gene] = y.cpu()

#train_actual.to_csv(f'{DATA_ROOT}/Train_dataset_actual_expressions.csv')
train_predicted.to_csv(f'{DATA_ROOT}/Train_dataset_predicted_expressions_LOG.csv')

#test_actual.to_csv(f'{DATA_ROOT}/Test_dataset_actual_expressions.csv')
test_predicted.to_csv(f'{DATA_ROOT}/Test_dataset_predicted_expressions_LOG.csv')

external_predicted.to_csv(f'{DATA_ROOT}/external_dataset_predicted_expressions_LOG.csv')



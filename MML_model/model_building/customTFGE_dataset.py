from torch.utils.data import Dataset
import torch
import numpy as np

#define pytorch dataset object
class CustomTFGE(Dataset):
    def __init__(self, device, TF_expressions, gene_expressions, target_gene, network, transform=None, target_transform=None):
        '''
        Custom dataset for loading expression data across all TFs and samples and one target gene expression across all samples
        For a given index will return all TF expressions in one sample and target gene expression across all samples

        Parameters
        --------------
        device : torch device
            torch device to put dataset on, must be the same device as the model
        TF_expressions : Pandas dataframe
            pandas dataframe of TF expressions where columns are gene names and rows are samples. Expressions should be TPM normalised and log10 transformed.
        gene_expressions : Pandas dataframe
            pandas dataframe of target gene expressions, columns are gene names and rows are samples. Expressions should be TPM normalised and log10 transformed.
        target_gene : String
            String containing name of target gene for given model.
        network : networkx network
            Undirected network of the singalling pathway of interest. Used to filter TFs used when predicting target gene expression.
        '''
        #no transforms needed so set to none
        self.transform = transform
        self.target_transform = target_transform

        #load network
        self.network = network
        self.TF_expressions = TF_expressions
        self.gene_expressions = gene_expressions

        #subset to just target gene of interest
        self.target_gene = target_gene
        self.gene_expressions = self.gene_expressions[self.target_gene]

        #convert to torch tensors
        self.TF_expressions = torch.tensor(np.asarray(self.TF_expressions).T, dtype = torch.float32, device = device)
        self.gene_expressions = torch.tensor(np.asarray(self.gene_expressions), dtype = torch.float32, device = device)

    def __len__(self):
        #length of the dataset is the number of samples (not TFs in the dataset) - 15935
        return self.TF_expressions.shape[1]

    def __getitem__(self, idx):
        #get all TFs from sample correspinding to index
        TFs_exp = self.TF_expressions[:, idx]
        #get the target gene expression for the target model
        Gene_exp = self.gene_expressions[idx]
        #returns TFs for sample idx and target gene for sample idx
        return TFs_exp, Gene_exp
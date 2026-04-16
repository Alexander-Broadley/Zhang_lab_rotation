import numpy as np
import pandas as pd

DATA_ROOT = '../../data/'

#==================================================================
# Load Datasets
#==================================================================

#Load datasets for original model building to filter the GEP genes 
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
#Load target gene expressions
gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/Geneexpression (full).tsv"), sep='\t', header=0)
#load TF expressions
TF_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/TF(full).tsv"), sep='\t', header=0)

#Load GEP liver expressions and metadata
meta = pd.read_csv(f'{DATA_ROOT}GEP_Liver_bulk/Hbulk_meta.txt', sep = '\t')
TPM_gep = pd.read_csv(f'{DATA_ROOT}GEP_Liver_bulk/Hbulk_expr.txt', sep = '\t')

#==================================================================
# Filter Datasets
#==================================================================

#get only NAFLD sample IDs
NAFLD_sample_IDs = list(meta[meta['Phenotype'] == 'Normal']['Run'])

#make indexes the gene names
TPM_gep.index = TPM_gep.iloc[:, 0]
#drop the gene name column
TPM_exp = TPM_gep.drop('gene', axis = 1)

#get NAFLD samples and transpose to make genes columns
GEP_exp = TPM_exp[NAFLD_sample_IDs].T

#filter genes to nodes in network
network_tfs = set(net['TF'].unique())      # TFs
network_genes = set(net['Gene'].unique())  # target genes
network_nodes = network_tfs | network_genes

TF_expressions = TF_expressions[[gene for gene in TF_expressions.columns if gene in list(network_nodes)]]
gene_expressions = gene_expressions[[gene for gene in gene_expressions.columns if gene in list(network_nodes)]] 

#Get TFs used in model building from GEP dataset - all from original in GEP
GEP_TFs = GEP_exp[list(TF_expressions.columns)]
#15643 of the 16,1000 target genes in the dataset
GEP_target_genes = GEP_exp[[gene for gene in gene_expressions.columns if gene in GEP_exp.columns]]

print(pd.Series(GEP_TFs.columns == TF_expressions.columns).value_counts())

GEP_target_genes.to_csv(f'{DATA_ROOT}/GEP_Liver_bulk/GEP_Normal_gene_expressions.csv')
GEP_TFs.to_csv(f'{DATA_ROOT}/GEP_Liver_bulk/GEP_Normal_TF_expressions.csv')


#script for standard dataset filtering done for all models
#removal of TF if TF is the target gene is done in a model specific manner
import pandas as pd

def filter_datasets(net, GE_df):
    '''
    Takes a network file with two columns, 'Gene' and 'TF' and filters a gene expression dataframe (where column names are gene names), splitting it into two dataframes consisting of TF and gene expressions.
    '''
    #Identify genes in network
    network_tfs = set(net['TF'].unique())
    network_genes = set(net['Gene'].unique())

    #subset TF and gene expressions to just those that appear in respecive columns of the network
    TF_expressions = GE_df[[gene for gene in GE_df.columns if gene in list(network_tfs)]]
    gene_expressions = GE_df[[gene for gene in GE_df.columns if gene in list(network_genes)]] 
    
    return(TF_expressions, gene_expressions)
  
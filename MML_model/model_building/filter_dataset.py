#script for standard dataset filtering done for all models
# Target gene specific filtering done later in model construction
import pandas as pd

def filter_datasets(DATA_ROOT, load_genes = True):
    #loading genes takes 40+ seconds so added option to skip for speedup when possible

    #Load network
    net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
    
    #Load target gene expressions if specific
    if load_genes == True:
        gene_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/Geneexpression (full).tsv"), sep='\t', header=0)
    #load TF expressions
    TF_expressions = pd.read_csv((f"{DATA_ROOT}/Full data files/TF(full).tsv"), sep='\t', header=0)

    #filter network to only include TFs that are in the expression dataset
    net = net[net['TF'].isin(TF_expressions.columns)]

    #Identify genes in network
    network_tfs = set(net['TF'].unique())
    network_genes = set(net['Gene'].unique())
    network_nodes = network_tfs | network_genes

    #subset TF and gene expressions to just those that appear in the network
    TF_expressions = TF_expressions[[gene for gene in TF_expressions.columns if gene in list(network_nodes)]]
    if load_genes == True:
        gene_expressions = gene_expressions[[gene for gene in gene_expressions.columns if gene in list(network_nodes)]] 
        return(TF_expressions, gene_expressions, net)
    else:
        return(TF_expressions, net)
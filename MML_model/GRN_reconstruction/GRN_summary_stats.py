import pandas as pd
import numpy as np
import networkx as nx
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

GRN_ROOT = './data'

#set sex you want to get and plot GRN statistics for
#sex = '_FEMALE'
#sex = '_MALE'
sex = ''

GRN = pd.read_csv(f'{GRN_ROOT}/100per_act_inferred_GRN{sex}_norm.csv', index_col = 0)
print(GRN.head())

#add a regulatory nature column
GRN['Reg_name'] = np.where(GRN['Reg'] < 0, 'Inhibitory', 'Excitatory')

#create graph from edgelist with edge labels
graph = nx.from_pandas_edgelist(GRN, source = 'TF', target='Gene', edge_attr='Reg_name', create_using = nx.DiGraph)

#get list of gene names that are TFs only, TGs only, and both
DATA_ROOT = '../../data'
network = pd.read_csv(f'{DATA_ROOT}/Full data files/network(full).tsv', sep='\t')
TFs = set(GRN['TF'])
TGs = set(GRN['Gene'])
TF_TGs = TFs & TGs
#mkae TFs set the TFs that are not also TGs
TFs = TFs - TF_TGs
TGs = TGs - TF_TGs
print(f'There are {len(TFs)} TFs and {len(TGs)} TGs. {len(TF_TGs)} TFs are also TGs')

in_degree_dict = {}
out_degree_dict = {}
#get node in and out degrees and put in dictionary where keys are gene names
for node in graph.nodes():
    in_degree_dict[node] = graph.in_degree(node)
    out_degree_dict[node] = graph.out_degree(node)

#make TF specific dictionaries
TF_in_degrees = {k: in_degree_dict[k] for k in TFs}
TF_out_degrees = {k: out_degree_dict[k] for k in TFs}

#TG specific dictionaries
TG_in_degrees = {k: in_degree_dict[k] for k in TGs}
TG_out_degrees = {k: out_degree_dict[k] for k in TGs}

#TF_TG specific dictionaries
TF_TG_in_degrees = {k: in_degree_dict[k] for k in TF_TGs}
TF_TG_out_degrees = {k: out_degree_dict[k] for k in TF_TGs}

#plot out degree distributions (of nodes that will have it)
fig, ax = plt.subplots(ncols = 2, figsize = (15, 5))
ax[0].hist(TF_out_degrees.values(), bins = 15, color = '#b00000')
ax[0].set_title('TFs (Only)', fontsize = 15)
ax[1].hist(TF_TG_out_degrees.values(), bins = 15, color = '#6b00d6')
ax[1].set_title('TF and Target Genes', fontsize = 15)
fig.suptitle('Out Degree Distributions', fontsize = 20)
plt.savefig('./figures/OutDistributions_norm', dpi = 300)

#plot in degree distributions (of nodes that will have it)
fig, ax = plt.subplots(ncols = 2, figsize = (15, 5))
ax[0].hist(TG_in_degrees.values(), bins = 15, color = '#03038f')
ax[0].set_title('Target Genes (Only)', fontsize = 15)
ax[1].hist(TF_TG_in_degrees.values(), bins = 15, color = '#6b00d6')
ax[1].set_title('TF and Target Genes', fontsize = 15)
fig.suptitle('In Degree Distributions', fontsize = 20)
plt.savefig('./figures/InDistributions_norm', dpi = 300)

#calculate centralities of TF_TGs
degree_centralities = nx.degree_centrality(graph)
TF_TG_centralities = {k: degree_centralities[k] for k in TF_TGs}

#calculate betweenes of TF_TGs
node_betweeness = nx.betweenness_centrality(graph)
TF_TG_node_betweeness = {k: node_betweeness[k] for k in TF_TGs}

#create results df
TFTG_df = pd.DataFrame([TF_TG_in_degrees, TF_TG_out_degrees, TF_TG_centralities, TF_TG_node_betweeness], index = ['in degree', 'out degree', 'Centrality', 'Betweeness']).T

#plot in and out degree coloured by centrality
fig, ax = plt.subplots(1)
sns.scatterplot(TFTG_df, x = 'in degree', y = 'out degree', hue = 'Centrality', s = 7, palette = 'icefire')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set_ylabel('Out Degree', fontsize = 15)
ax.set_xlabel('In Degree', fontsize = 15)
ax.set_title('TF-TG Connectedness', fontsize = 20)
plt.savefig(f'./figures/InOutConnectednessTFTG{sex}_norm', dpi = 300, bbox_inches = 'tight')

#plot in and out degree coloured by betweenes
fig, ax = plt.subplots(1)
sns.scatterplot(TFTG_df, x = 'in degree', y = 'out degree', hue = 'Betweeness', s = 7, palette = 'icefire')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set_ylabel('Out Degree', fontsize = 15)
ax.set_xlabel('In Degree', fontsize = 15)
ax.set_title('TF-TG Betweeness', fontsize = 20)
plt.savefig(f'./figures/InOutBetweenessTFTG{sex}_norm', dpi = 300, bbox_inches = 'tight')
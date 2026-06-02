import networkx as nx

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import animation

epoch_index = 0


in_weights = pd.read_csv('./data/in_weights_OPALIN.csv', index_col=0)
out_weights = pd.read_csv('./data/out_weights_OPALIN.csv', index_col=0)

print(in_weights.head())

fig, ax = plt.subplots(ncols = 1, figsize = (10, 10))
fig.patch.set_facecolor('#0f0f0f')
ax.set_facecolor('#0f0f0f')

'''
G = nx.DiGraph()
G.add_node('OPALIN')

for row_index in range(0, 20):
    G.add_node(f"TF{row_index}", label = f"TF{row_index}")
    G.add_node(f"H{row_index}", label = f"H{row_index}")
    G.add_edge(f"TF{row_index}", f"H{row_index}", weight = in_weights.iloc[row_index, epoch_index])
    G.add_edge(f"H{row_index}", f"OPALIN", weight = out_weights.iloc[row_index, epoch_index])

edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
pos = graphviz_layout(G, prog='dot', args="-Grankdir=LR")

max_x = max(x for x, y in pos.values())
mid_y = np.mean([y for x, y in pos.values()])
pos['OPALIN'] = (max_x, mid_y)

nx.draw(G, pos = pos, edge_color=weights, edge_cmap=plt.cm.plasma, ax = ax)
'''
all_in  = in_weights.iloc[:,  :199].values.flatten()
all_out = out_weights.iloc[:, :199].values.flatten()
vmin = min(all_in.min(), all_out.min())
vmax = max(all_in.max(), all_out.max())

def update(epoch_index):
    ax.clear()
    fig.patch.set_facecolor('#0f0f0f')
    ax.set_facecolor('#0f0f0f')
    G = nx.DiGraph()
    G.add_node('OPALIN')

    for row_index in range(0, 30):
        G.add_node(f"TF{row_index}", label = f"TF{row_index}")
        G.add_node(f"H{row_index}", label = f"H{row_index}", in_weight = in_weights.iloc[row_index, epoch_index])
        G.add_edge(f"TF{row_index}", f"H{row_index}", weight = in_weights.iloc[row_index, epoch_index])
        G.add_edge(f"H{row_index}", f"OPALIN", weight = out_weights.iloc[row_index, epoch_index])

    for node in G.nodes():
        print(node.in_weight)
        


    edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
    pos = graphviz_layout(G, prog='dot', args="-Grankdir=LR")

    max_x = max(x for x, y in pos.values())
    mid_y = np.mean([y for x, y in pos.values()])
    pos['OPALIN'] = (max_x, mid_y)

    nx.draw(G, pos = pos, edge_color=weights, edge_cmap=plt.cm.plasma, ax = ax, node_color= ["#f1070b" if name == 'OPALIN'
                                                                                             else "#3D03EC" if name.startswith('TF')
                                                                                             else "#FEFE13" for name in G.nodes()])

ani = animation.FuncAnimation(
    fig, update,
    interval = 10,
    frames=range(0, 199),
    repeat=True
)

plt.show()
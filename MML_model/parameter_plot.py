#script to create a plot displaying various important parameters for a given model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model_params = pd.read_csv('/Users/alexanderbroadley/Documents/PhD/Zhang Lab/Learning_PyTorch/MML_model/data/TPM_model_parameters/MBOAT7_TPM_MPs.csv', header = 0, index_col=0)

gene_features = list(model_params.index)
fig, ax = plt.subplots(ncols=1)
ax = plt.bar(x = gene_features, height = model_params["out_weight"], color='darkblue')


#iterate through genes and color bar according to whether outweight agrees with network reg direction
for i in range(0, len(gene_features)):
    bar = ax.patches[i]
    bar_val = ax.patches[i].get_height()
    outweight = model_params.loc[gene_features[i], 'reg_direction']
    if (bar_val > 0 ) & (outweight > 0):
        print(f"bar val {bar_val} outweight {outweight}")
        bar.set_color('green')
    elif (bar_val < 0 ) & (outweight < 0):
        print(f"bar val {bar_val} outweight {outweight}")
        bar.set_color('green')
    else:
        bar.set_color('red')


plt.xticks(rotation = 90, size = 8)
plt.title('Outweights for MBOAT7 model')
plt.show()
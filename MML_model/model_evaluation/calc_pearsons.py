import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

DATA_ROOT = '../../data'
FIGURE_ROOT = './figures'

#determines which set of models to use
suffix = '_MF_external'

#======================================================================
#load expressions datasets
#======================================================================

print('Loading Datasets')

train_predicted_TPM = pd.read_csv(f"{DATA_ROOT}/Train_dataset_predicted_expressions{suffix}.csv", index_col=0, header=0)
train_actual_TPM = pd.read_csv(f"{DATA_ROOT}/Train_dataset_actual_expressions{suffix}.csv", index_col=0, header=0)

test_predicted_TPM = pd.read_csv(f"{DATA_ROOT}/Test_dataset_predicted_expressions{suffix}.csv", index_col=0, header=0)
test_actual_TPM = pd.read_csv(f"{DATA_ROOT}/Test_dataset_actual_expressions{suffix}.csv", index_col=0, header=0)

female_actual = pd.read_csv(f"{DATA_ROOT}/female_actual_gene_expressions.csv", index_col = 0)
female_predicted = pd.read_csv(f"{DATA_ROOT}/female_predicted_gene_expressions.csv", index_col = 0)

male_actual = pd.read_csv(f"{DATA_ROOT}/male_actual_gene_expressions.csv", index_col = 0)
male_predicted = pd.read_csv(f"{DATA_ROOT}/male_predicted_gene_expressions.csv", index_col = 0)
'''
external_actual_TPM = pd.read_csv(f"{DATA_ROOT}/Full data files/Liver_bulk_external.tsv", sep = '\t', index_col=0, header=0)
external_predicted_TPM = pd.read_csv(f"{DATA_ROOT}/external_predicted_expressions{suffix}.csv", index_col=0, header=0)
'''
#remove columns for genes that no models exist for
train_predicted_TPM = train_predicted_TPM.dropna(axis=1, how='all')
train_actual_TPM = train_actual_TPM.dropna(axis=1, how='all')
test_predicted_TPM = test_predicted_TPM.dropna(axis=1, how='all')
test_actual_TPM = test_actual_TPM.dropna(axis=1, how='all')
#external_predicted_TPM = external_actual_TPM.dropna(axis=1, how='all')
#external_actual_TPM = external_actual_TPM.dropna(axis=1, how='all')


#remove columns for genes that no models exist for
#external_predicted_TPM.drop(['SHOX', 'ZBED1'], axis = 1, inplace = True)

#temporary before fixing code for external model creation - drop all NA columns
#external_predicted_TPM = external_predicted_TPM.dropna(axis=1, how='all')
#external_actual_TPM = external_actual_TPM[external_predicted_TPM.columns]

#ensure column indexes (genes) are in the same order across datasets
train_actual_TPM = test_actual_TPM[train_predicted_TPM.columns]
test_actual_TPM = test_actual_TPM[test_predicted_TPM.columns]
female_actual = female_actual[female_predicted.columns]
male_actual = male_actual[male_predicted.columns]
#external_actual_TPM = external_actual_TPM[external_predicted_TPM.columns]
print('Loaded Datasets')

#print(external_actual_TPM.head())
#print(external_predicted_TPM.head())

#======================================================================
#calculate sample wise correlation values
#======================================================================

def create_pearsons_violin(results_df, y_col, title, file_title, figure_root = './figures'):
    '''
    Taking a results df with two columns ['pearsons', 'dataset'] 
    Plots the pearsons values by dataset of origin as violin plots and saves figure with specified file_title at figure_root
    '''
    palette = 'viridis'

    savepath = f'{figure_root}/{file_title}'

    fig, ax = plt.subplots()
    
    ax = sns.violinplot(data = results_df, x = 'dataset', y= y_col, hue = 'dataset', palette=palette)

    if y_col == 'pearsons':
        ax.set_ylabel('Pearsons R', fontsize = 15)
    elif y_col == 'spearmans':
        ax.set_ylabel('Spearmans Rank', fontsize = 15)
    ax.set_xlabel('Dataset', fontsize = 15)
    #fig.canvas.draw()
    #labels = [item.get_text() for item in ax.get_xticklabels()]
    #labels[0], labels[1], labels[2] = ['Training', 'Test', 'External']
    #ax.set_xticklabels(['Training', 'Test', 'External'], fontsize = 10)
    fig.suptitle(title, fontsize = 20)
    plt.savefig(savepath, dpi = 300, bbox_inches = 'tight')

def get_correlations(df1, df2, rowise = True):
    '''
    Function that takes two dataframes of the same dimensions and uses np.corrcoef to get correlation values
    Returns only the bottom left of the resulting similiarity matrix (excluding diagonal)

    Calculates correlation rowwise when rowwise == True, else does columnwise
    '''

    #if calculating columnwise values just transpose the dataframes beforehand
    if rowise != True:
        df1 = df1.T
        df2 = df2.T

    n = df1.shape[0]

    corr_vals = []

    for df1_row, df2_row in zip(df1.itertuples(), df2.itertuples()):
        corr_vals.append(scipy.stats.pearsonr(np.asarray(df1_row), np.asarray(df2_row)).statistic)

    #calculate pearsons between all rows
    #corr_mat = np.corrcoef(df1, df2)
    #return just the values correspinding to row1 df1 vs row1 df2 etc.
    #return(np.diag(corr_mat[:n, n:]))

    return(corr_vals)

def get_correlations_df(df1, df2, dataset_label, rowise = True):
    '''
    Uses get_correlations to calculate the row or columnwise perasons correlations between two dataframes with the same dimensions
    
    Returns the result as a df suitable for later concatenating and use with seaborn violin plots where the columns are ['value', 'dataset']
    The resulting dataframe has dimensions (n, 2) where n is the number of rows in df1 
    '''

    #initialise df
    results_df = pd.DataFrame(columns = ['pearsons', 'spearmans','MSE', 'dataset'])
    #add pearsons correlation between df1, df2 to the pearsons column

    if rowise == False:
        df1 = df1.T
        df2 = df2.T

    for i in range(0, df1.shape[0]):
        results_df.loc[i, "pearsons"] = scipy.stats.pearsonr(df1.iloc[i, :], df2.iloc[i, :]).statistic
        results_df.loc[i, 'spearmans'] = scipy.stats.spearmanr(df1.iloc[i, :], df2.iloc[i, :]).statistic
        #results_df.loc[i, 'MSE'] = mean_squared_error(df1.iloc[i, :], df2.iloc[i, :])
        results_df.loc[i, "dataset"] = dataset_label
    
    #results_df['pearsons'] = get_correlations(df1, df2, rowise = rowise)
    #create a matching dataset
    #results_df['dataset'] = dataset_label

    return(results_df)

print('Calculating Samplewise Correlations')
samplewise_df_list = []
samplewise_df_list.append(get_correlations_df(train_actual_TPM, train_predicted_TPM, dataset_label='Train', rowise = True))
samplewise_df_list.append(get_correlations_df(test_actual_TPM, test_predicted_TPM, dataset_label='Test', rowise = True))
samplewise_df_list.append(get_correlations_df(female_actual, female_predicted, dataset_label='Female', rowise = True))
samplewise_df_list.append(get_correlations_df(male_actual, male_predicted, dataset_label='Male', rowise = True))

samplewise_correlations = pd.concat(samplewise_df_list, ignore_index = True)
create_pearsons_violin(results_df=samplewise_correlations, y_col='pearsons', title = f'Samplewise Pearsons Correlations', file_title= f'samplewise_pearsons{suffix}', figure_root=f'./figures/{suffix}')
create_pearsons_violin(results_df=samplewise_correlations, y_col='spearmans', title = f'Samplewise Spearmans Correlations', file_title=f'samplewise_spearmans{suffix}', figure_root=f'./figures/{suffix}')
#create_pearsons_violin(results_df=samplewise_correlations, y_col='MSE', title = f'Samplewise MSE', file_title=f'samplewise_MSE{suffix}', figure_root=f'./figures/{suffix}')

print('Finished Samplewise Calculations')

print('Calculating Genewise Correlations')
genewise_df_list = []
genewise_df_list.append(get_correlations_df(train_actual_TPM, train_predicted_TPM, dataset_label='Train', rowise = False))
genewise_df_list.append(get_correlations_df(test_actual_TPM, test_predicted_TPM, dataset_label='Test', rowise = False))
genewise_df_list.append(get_correlations_df(female_actual, female_predicted, dataset_label='Female', rowise = False))
genewise_df_list.append(get_correlations_df(male_actual, male_predicted, dataset_label='Male', rowise = False))

genewise_correlations = pd.concat(genewise_df_list, ignore_index=True)

create_pearsons_violin(results_df=genewise_correlations, y_col='pearsons', title = f'Genewise Pearsons Correlations', file_title = f'genewise_pearsons{suffix}', figure_root=f'./figures/{suffix}')
create_pearsons_violin(results_df=genewise_correlations, y_col='spearmans', title = f'Genewise Spearmans Correlations', file_title = f'genewise_spearmans{suffix}', figure_root=f'./figures/{suffix}')
#create_pearsons_violin(results_df=genewise_correlations, y_col='MSE', title = f'Genewise MSE', file_title = f'genewise_MSE{suffix}', figure_root=f'./figures/{suffix}')

print('Finished Genewise Calculations')


#======================================================================
#calculate gene wise correlation values
#======================================================================

samplewise_correlations.to_csv(f'data/samplewise_correlations{suffix}.csv')
genewise_correlations.to_csv(f'data/genewise_correlations{suffix}.csv')
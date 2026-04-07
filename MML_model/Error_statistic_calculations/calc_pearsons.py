import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt


DATA_ROOT = '../data'
FIGURE_ROOT = './figures'

#======================================================================
#load expressions datasets
#======================================================================


train_predicted_TPM = pd.read_csv(f"{DATA_ROOT}/Train_dataset_predicted_expressions.csv", index_col=0, header=0)
train_actual_TPM = pd.read_csv(f"{DATA_ROOT}/Train_dataset_actual_expressions.csv", index_col=0, header=0)

test_predicted_TPM = pd.read_csv(f"{DATA_ROOT}/Test_dataset_predicted_expressions.csv", index_col=0, header=0)
test_actual_TPM = pd.read_csv(f"{DATA_ROOT}/Test_dataset_actual_expressions.csv", index_col=0, header=0)

external_actual_TPM = pd.read_csv(f"{DATA_ROOT}/Full data files/liver_bulk_external.tsv", sep = '\t', index_col=0, header=0)
external_predicted_TPM = pd.read_csv(f"{DATA_ROOT}/external_dataset_predicted_expressions.csv", index_col=0, header=0)

#remove columns for genes that no models exist for
external_predicted_TPM.drop(['SHOX', 'ZBED1'], axis = 1, inplace = True)

#temporary before fixing code for external model creation - drop all NA columns
external_predicted_TPM = external_predicted_TPM.dropna(axis=1, how='all')

#ensure column indexes (genes) are in the same order across datasets
test_actual_TPM = test_actual_TPM[train_predicted_TPM.columns]
test_actual_TPM = test_actual_TPM[test_predicted_TPM.columns]
external_actual_TPM = external_actual_TPM[external_predicted_TPM.columns]

print('Loaded Datasets')

#======================================================================
#calculate sample wise correlation values
#======================================================================

def create_pearsons_violin(results_df, title, file_title, figure_root = './figures'):
    '''
    Taking a results df with two columns ['pearsons', 'dataset'] 
    Plots the pearsons values by dataset of origin as violin plots and saves figure with specified file_title at figure_root
    '''
    palette = 'viridis'

    savepath = f'{figure_root}/{file_title}'

    fig, ax = plt.subplots()

    results_df['dataset'] = results_df.dataset.astype('category')
    
    ax = sns.violinplot(data = results_df, x = 'dataset', y="pearsons", hue = 'dataset', palette=palette)
    ax.set_ylabel('Pearsons R', fontsize = 15)
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
    results_df = pd.DataFrame(columns = ['pearsons', 'spearmans', 'dataset'])
    #add pearsons correlation between df1, df2 to the pearsons column

    if rowise == False:
        df1 = df1.T
        df2 = df2.T

    for i in range(0, df1.shape[0]):
        results_df.loc[i, "pearsons"] = scipy.stats.pearsonr(df1.iloc[i, :], df2.iloc[i, :]).statistic
        results_df.loc[i, 'spearmans'] = scipy.stats.spearmanr(df1.iloc[i, :], df2.iloc[i, :]).statistic
        results_df.loc[i, "dataset"] = dataset_label
    
    #results_df['pearsons'] = get_correlations(df1, df2, rowise = rowise)
    #create a matching dataset
    #results_df['dataset'] = dataset_label

    return(results_df)


samplewise_df_list = []
samplewise_df_list.append(get_correlations_df(train_actual_TPM, train_predicted_TPM, dataset_label='Train', rowise = True))
samplewise_df_list.append(get_correlations_df(test_actual_TPM, test_predicted_TPM, dataset_label='Test', rowise = True))
samplewise_df_list.append(get_correlations_df(external_actual_TPM, external_predicted_TPM, dataset_label='External', rowise = True))

samplewise_correlations = pd.concat(samplewise_df_list, ignore_index = True)
#create_pearsons_violin(results_df=samplewise_correlations, title = 'Samplewise Pearsons Correlations for TPM models', file_title='samplewise_pearsons_TPM')
create_pearsons_violin(results_df=samplewise_correlations, title = 'Samplewise Pearsons Correlations for TPM models', file_title='samplewise_pearsons_TPM_iterations2')
#print('Finished Samplewise Calculations')

genewise_df_list = []
genewise_df_list.append(get_correlations_df(train_actual_TPM, train_predicted_TPM, dataset_label='Train', rowise = False))
genewise_df_list.append(get_correlations_df(test_actual_TPM, test_predicted_TPM, dataset_label='Test', rowise = False))
genewise_df_list.append(get_correlations_df(external_actual_TPM, external_predicted_TPM, dataset_label='External', rowise = False))

genewise_correlations = pd.concat(genewise_df_list, ignore_index=True)

genewise_correlations.to_csv('temp_file.csv')
create_pearsons_violin(results_df=genewise_correlations, title = 'Genewise Pearsons Correlations for TPM models', file_title = 'genewise_pearsons_TPM_iterations2')
print('Finished Genewise Calculations')


'''
df_list = []
temp_df = pd.DataFrame(columns = ['pearsons', 'dataset'])
#test_pearsons = get_correlations(train_actual_TPM, train_predicted_TPM)
temp_df['pearsons'] = get_correlations(train_actual_TPM, train_predicted_TPM)
temp_df['dataset'] = 'train'

df_list.append(temp_df)

temp_df = pd.DataFrame(columns = ['pearsons', 'dataset'])
#test_pearsons = get_correlations(train_actual_TPM, train_predicted_TPM)
temp_df['pearsons'] = get_correlations(test_actual_TPM, test_predicted_TPM)
temp_df['dataset'] = 'test'

df_list.append(temp_df)

temp_df = pd.DataFrame(columns = ['pearsons', 'dataset'])
#test_pearsons = get_correlations(train_actual_TPM, train_predicted_TPM)
temp_df['pearsons'] = get_correlations(external_actual_TPM, external_predicted_TPM)
temp_df['dataset'] = 'external'


results_df = pd.concat(df_list)

print(results_df)
print(results_df['dataset'].value_counts())


print('Calculating samplewise pearsons for test dataset')
df_list = []
temp_df = pd.DataFrame(columns = ['pearsons', 'dataset'])
for i in range(0, test_predicted_TPM.shape[0]):
    temp_df.loc[i, "pearsons"] = scipy.stats.pearsonr(test_predicted_TPM.iloc[i, :], test_actual_TPM.iloc[i, :]).statistic
    temp_df.loc[i, "dataset"] = 'Test'
df_list.append(temp_df)

print('Calculating samplewise pearsons for train dataset')
temp_df = pd.DataFrame(columns = ['pearsons', 'dataset'])
for i in range(0, train_predicted_TPM.shape[0]):
    temp_df.loc[i, "pearsons"] = scipy.stats.pearsonr(train_predicted_TPM.iloc[i, :], train_actual_TPM.iloc[i, :]).statistic
    temp_df.loc[i, "dataset"] = 'Train'
df_list.append(temp_df)

print('Calculating samplewise external for test dataset')
temp_df = pd.DataFrame(columns = ['pearsons', 'dataset'])
for i in range(0, external_predicted_TPM.shape[0]):
    temp_df.loc[i, "pearsons"] = scipy.stats.pearsonr(external_actual_TPM.iloc[i, :], external_predicted_TPM.iloc[i, :]).statistic
    temp_df.loc[i, "dataset"] = 'External'
df_list.append(temp_df)


results_df = pd.concat(df_list)
print(results_df['dataset'].value_counts())
'''


#======================================================================
#calculate gene wise correlation values
#======================================================================


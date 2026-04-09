import seaborn as sns
import matplotlib.pyplot as plt

def create_pearsons_violin(results_df, y_col, title, file_title, figure_root = './figures'):
    '''
    Taking a results df with two columns ['pearsons', 'dataset'] 
    Plots the pearsons values by dataset of origin as violin plots and saves figure with specified file_title at figure_root

    Parameters
    ---------------
    results_df : pandas dataframe
        dataframe containg the results to plot. Must contain a categorical column called dataset to serve as x axis categories
    y_col : string
        name of column in results df with statistic to make violin plot for 
    title : string
        title to display on plot
    file_title : string
        filename of saved figure
    figure_root : string
        path to directory in which to save the figure

    Returns
    --------------
    None
        Saves figure to desired location
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

def feature_error_scatter(df, y_col, x_col, hue_col, x_label, y_label, title, file_title, figure_root = './figures', palette = 'icefire'):
    '''
    Creates a scatter plot where points are coloured by a third column. Original intended use was for two error scores against model size.
    '''
    
    fig, ax = plt.subplots()

    #ensures that legend is coloured by order of hue_col values
    sorted_results = df.sort_values('hue_col')
    
    sns.scatterplot(data=sorted_results, x = x_col, y = y_col, legend=True, ax = ax, hue = hue_col, palette=palette, s=10)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    sns.despine()
    ax.set_xlabel(x_label, fontsize = 10)
    ax.set_ylabel(y_label, fontsize = 10)
    fig.suptitle(title)
    plt.savefig(f'{figure_root}/file_title', bbox_inches='tight')
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def draw_bar_importance(col, *args):
    '''This function draw three plots in row. Each presents a barplot of feature importance
    from dataset. The first one should be Decision tree feature importance, second-Random Forest
    and lastly XGBoost.
       
       Parameters
       -------------------
       col:  list or iterable
           list of dataframe column names
       args:  list or iterable
           list of vectors containing value of feature importance

       Returns
       ------------------
       None'''

    # creating figure
    fig, ax = plt.subplots(3, 1, figsize=(18, 15))
    
    # barplots
    sns.barplot(x=col, y=args[0], palette='Blues_d', ax=ax[0])
    sns.barplot(x=col, y=args[1], palette='Reds_d', ax=ax[1])
    sns.barplot(x=col, y=args[2], palette='Greens_d', ax=ax[2])
    
    # rotating ticks because they overlap
    ax[0].set_xticklabels(col, rotation=45)
    ax[1].set_xticklabels(col, rotation=45)
    ax[2].set_xticklabels(col, rotation=45)
    
    # setting titles on plots
    ax[0].set_title('Decision tree feature importance', fontsize=6)
    ax[1].set_title('Random forest feature importance', fontsize=6)
    ax[2].set_title('XGBoost feature importance', fontsize=6)

    plt.tight_layout()
    plt.show()


def plot_result_point(df, w_time=True, result_best=None):
    '''This function have two tribes. If w_time is True then it will draw
    two columns with four plots in each one. They will be pointplots. First column
    compare results from different metrics between five best performing models. In the 
    second time there is mean evaluation time.

    If w_time is false it will draw one column of 4 plots. It will need result_best dataframe
    in similar form as df. Each plot will be a point plot from two different dataframes comparing
    their results. 
       
       Parameters
       -------------------
       df: pandas DataFrame
           DataFrame of metrics it should contain name of model in first column and
           metrics: accuracy, recall, precision, F1_score in this order
       w_time:  boolean
           parameter from which depends function's tribe
       result_best: pandas DataFrame
            similar dataframe as df

       Returns
       ------------------
       None'''
    
    # number of metrics
    n = 4
    
    if w_time:
        # creating figure
        fig, ax = plt.subplots(n, 2, figsize=(15, 10))
        # iterating over index and metric name
        for idx, col in enumerate(df.columns[1:-1]):
            # choosing 5 best models
            subset = df.nlargest(5, [col])
            # time pointplot
            sns.pointplot(x='model', y='time', data=subset, color='red', ax=ax[idx][1])
            # metrics pointplot
            sns.pointplot(x='model', y=col, data=subset, ax=ax[idx][0])

    else:
        fig, ax = plt.subplots(n, 1, figsize=(12, 10))
     
        # iterating over index and metric name
        for idx, col in enumerate(df.columns[1:5]):
            # choosing 5 best models
            subset = df.nlargest(5, [col]).copy()
            # extracting the name of the model by splitting and
            # taking first element of list
            subset.model = subset['model'].str.split('_')
            subset.model = subset['model'].apply(lambda x: x[0])
            
            # for example if model is KNN4 we change it to simply KNN
            if any(subset.model.str.contains('KNN')):
                subset.loc[subset.model.str.contains('KNN'), 'model'] = 'KNN'
            
               
            subset['Legend'] = 'Previous results'
            subset = subset.append(result_best.iloc[:, :5], ignore_index=True, sort=True)
            subset['Legend'] = subset['Legend'].fillna('Current results')

            
            # metrics pointplot
            sns.pointplot(x='model', y=col, data=subset, hue='Legend', ax=ax[idx])
         
    plt.tight_layout()
    plt.show()

def draw_point_importance(col, *args):
    '''This function draw three plots in row. Each presents a pointplot of feature importance
    from dataset. The first one should be Decision tree feature importance, second-Random Forest
    and lastly XGBoost. It is like draw_bar_importance but it is prefered when len(col) is large.
       
       Parameters
       -------------------
       col:  list or iterable
           list of dataframe column names
       args:  list or iterable
           list of vectors containing value of feature importance

       Returns
       ------------------
       None'''
    
    # creating figure
    fig, ax = plt.subplots(3, 1, figsize=(18, 15))
    
    # barplots
    sns.pointplot(x=col, y=args[0], palette='Blues_d', ax=ax[0])
    sns.pointplot(x=col, y=args[1], palette='Reds_d', ax=ax[1])
    sns.pointplot(x=col, y=args[2], palette='Greens_d', ax=ax[2])
    
    # rotating ticks because they overlap
    ax[0].set_xticklabels(col, rotation=90)
    ax[1].set_xticklabels(col, rotation=90)
    ax[2].set_xticklabels(col, rotation=90)
    
    # setting titles
    ax[0].set_title('Decision tree feature importance', fontsize=6)
    ax[1].set_title('Random forest feature importance', fontsize=6)
    ax[2].set_title('XGBoost feature importance', fontsize=6)
    
    plt.tight_layout()
    plt.show()
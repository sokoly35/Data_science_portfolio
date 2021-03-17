import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np



import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly


def drawing_time_series(x, y, z):
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    g = sns.lineplot(x=x.index, y=x, ax=ax[0])
    h = sns.lineplot(x=y.index, y=y, ax=ax[1])
    i = sns.lineplot(x=z.index, y=z, ax=ax[2])

def rollings_plot(x, y, z):
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    moving_avg = x.rolling(12).mean().dropna().plot(ax=ax[0])
    moving_std = x.rolling(12).std().dropna().plot(ax=ax[0])
    x.plot(ax=ax[0])
    
    moving_avg = y.rolling(12).mean().dropna().plot(ax=ax[1])
    moving_std = y.rolling(12).std().dropna().plot(ax=ax[1])
    y.plot(ax=ax[1])

    
    moving_avg = z.rolling(12).mean().dropna().plot(ax=ax[2])
    moving_std = z.rolling(12).std().dropna().plot(ax=ax[2])
    z.plot(ax=ax[2])

def seasonal_decompose_my(df):
    result = seasonal_decompose(df, model='additive')
    fig= plt.figure();
    fig = result.plot()
    fig.set_size_inches(10,10);
    plt.show();

def test_stationarity(timeseries, name, window = 12, cutoff = 0.01):
    
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()
    t = timeseries.index

    #  Plot rolling statistics:
    fig, ax = plt.subplots(1, 1,figsize=(14, 7))
    orig = sns.lineplot(x=t, y=timeseries,label='Original',ax=ax)
    mean = sns.lineplot(x=t, y=rolmean, color='orange', label='Rolling Mean', ax=ax)
    std = sns.lineplot(x=t, y=rolstd, color='green', label = 'Rolling Std', ax=ax)
    ax.legend(loc='best')
    ax.set_title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    dftest = adfuller(timeseries, autolag='AIC', maxlag = 20 )
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    pvalue = dftest[1]

  
    fig = go.Figure(data=[go.Table(
        header=dict(values=['', 'Values'],
                    align='left'),
        cells=dict(values=[dfoutput.index, dfoutput.values],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(height=200, width=850, margin=dict(r=5, l=5, t=5, b=5))
    fig.show()

def plot_acf_pacf(series):
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    
    plot_acf(series, zero=True, ax=ax[0][0], lags=np.arange(30))
    plot_pacf(series, zero=True, ax=ax[0][1], lags=np.arange(30))
    
    plot_acf(series, zero=True, ax=ax[1][0], lags=np.arange(0, 60, 7))
    plot_pacf(series, zero=True, ax=ax[1][1], lags=np.arange(0, 60, 7))
    
    ax[0][0].set_xlabel('lag')
    ax[0][1].set_xlabel('lag')
    ax[1][0].set_xlabel('lag')
    ax[1][1].set_xlabel('lag')
    ax[1][0].set_title('Seasonal autocorrelation')
    ax[1][1].set_title('Seasonal partial autocorrelation')
    
    plt.tight_layout()
    plt.show()


def plot_forecast(model, series):
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    
    n = len(series)
    preds_in_sample = model.get_prediction(start=-n+1, end=-30)
    
    last_preds_in_sample = model.get_prediction(start=-30, end=-7)
    forecast = model.get_prediction(start=-7, dynamic=True)
    
    
    sns.lineplot(x=preds_in_sample.row_labels, y=series[:-30], ax=ax[0][0])
    sns.lineplot(x=preds_in_sample.row_labels, y=preds_in_sample.predicted_mean, ax=ax[0][0], 
                 color='chocolate')
    
    sns.lineplot(x=series.index[-30:], y=series[-30:], ax=ax[0][1])
    sns.lineplot(x=last_preds_in_sample.row_labels, y=last_preds_in_sample.predicted_mean, ax=ax[0][1],
                 color='chocolate')
    sns.lineplot(x=forecast.row_labels, y=forecast.predicted_mean, ax=ax[0][1], color='darkgreen')
    ax[0][1].fill_between(forecast.row_labels, forecast.conf_int().iloc[:, 0], forecast.conf_int().iloc[:, 1],
                         color='mediumseagreen', alpha=0.3)

    
    sns.lineplot(x=preds_in_sample.row_labels, y=np.exp(series[:-30]), ax=ax[1][0])
    sns.lineplot(x=preds_in_sample.row_labels, y=np.exp(preds_in_sample.predicted_mean), ax=ax[1][0], 
                 color='chocolate')
    
    sns.lineplot(x=series.index[-30:], y=np.exp(series[-30:]), ax=ax[1][1])
    sns.lineplot(x=last_preds_in_sample.row_labels, y=np.exp(last_preds_in_sample.predicted_mean), ax=ax[1][1],
                 color='chocolate')
    sns.lineplot(x=forecast.row_labels, y=np.exp(forecast.predicted_mean), ax=ax[1][1], color='darkgreen')
    ax[1][1].fill_between(forecast.row_labels, np.exp(forecast.conf_int().iloc[:, 0]), 
                          np.exp(forecast.conf_int().iloc[:, 1]),
                         color='mediumseagreen', alpha=0.3)
    
    ax[0][0].tick_params(axis='x', labelrotation=15)
    ax[0][1].tick_params(axis='x', labelrotation=15)
    ax[1][0].tick_params(axis='x', labelrotation=15)
    ax[1][1].tick_params(axis='x', labelrotation=15)

    ax[0][0].set_title('Predict in sample to september', fontsize=13, weight='bold')
    ax[0][1].set_title('Forecast for 30 days of september', fontsize=13, weight='bold')
    ax[1][0].set_title('Predict in sample to september after exponent', fontsize=13, weight='bold')
    ax[1][1].set_title('Forecast for 30 days of september after exponent', fontsize=13, weight='bold')

        
    plt.tight_layout()